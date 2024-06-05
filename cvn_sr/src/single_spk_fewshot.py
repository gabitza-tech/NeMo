import numpy as np
import torch
import time
from tqdm import tqdm
from omegaconf import OmegaConf
from nemo.core.config import hydra_runner
import os 

from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from utils.utils import CL2N_embeddings, data_SQ_from_pkl, analyze_data
from methods.paddle import PADDLE
from methods.simpleshot import Simpleshot
from utils.task_generator import Tasks_Generator

def run_paddle_new(enroll_embs,enroll_labels,test_embs,test_labels,method_info):
    x_q = torch.tensor(test_embs)
    y_q = torch.tensor(test_labels).long().unsqueeze(2)
    x_s = torch.tensor(enroll_embs)
    y_s = torch.tensor(enroll_labels).long().unsqueeze(2)
    
    task_dic = {}
    task_dic['y_s'] = y_s
    task_dic['y_q'] = y_q
    task_dic['x_s'] = x_s
    task_dic['x_q'] = x_q
    
    method = PADDLE(**method_info)
    logs = method.run_task(task_dic)
    #acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])

    return logs['acc'][:,-1].tolist()

#@hydra_runner(config_path="../conf", config_name="speaker_identification_fewshot.yaml")
def main():
    log_file = get_log_file(log_path='log_single_speaker10k', backbone='ecapa', dataset='voxceleb1', method='all_nq_5_neff_5_kshot_3')
    logger = Logger(__name__, log_file)

    n_tasks = 10000
    batch_size = 50

    normalize = True
    args={}
    args['iter']=20
    args['alpha']=20
    args['maj_vote'] = False
    method_info = {'device':'cuda','log_file':log_file,'args':args}
    methods = ['inductive','inductive_maj','transductive_centroid','transductive_L2_sum','EM','paddle']

    input_dir = "out_sampled_tasks_multispk"
    for file in os.listdir(input_dir):
        logger.info(f"Performing evaluation with n_query_per_class=5,n_eff=5,k_shot=3 with seed {file}.")
        filepath = os.path.join(input_dir,file)
        
        test_embs,test_labels,test_audios,enroll_embs,enroll_labels,enroll_audios = data_SQ_from_pkl(filepath) 
        enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,normalize)
        
        if 'paddle' in methods:
            logger.info(f"Alpha is equal to {args['alpha']}.")
        
        run_acc = {}
        for method in methods:
            run_acc[method] = []
        
        for start in tqdm(range(0,n_tasks,batch_size)):
            end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

            x_q,y_q,x_s,y_s = (test_embs[start:end],
                              test_labels[start:end],
                              enroll_embs[start:end],
                              enroll_labels[start:end])

            for method in methods:
                if method=="paddle":
                    acc_list = run_paddle_new(x_s, y_s, x_q, y_q,method_info)
                else:
                    eval = Simpleshot(avg="mean",backend="L2",method=method)
                    acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
                
                run_acc[method].extend(acc_list)
                
        for key in run_acc.keys():
            acc = sum(run_acc[key])/len(run_acc[key])*100
            logger.info(f"{key} acc: {acc}%")

if __name__ == "__main__":
    main()