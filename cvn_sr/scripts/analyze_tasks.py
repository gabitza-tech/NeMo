import numpy as np
#from utils.task_generator import Tasks_Generator
import torch
import time
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from utils.utils import load_pickle, embedding_normalize, analyze_data
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.paddle import PADDLE
from methods.em_dirichlet import HARD_EM_DIRICHLET
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from omegaconf import OmegaConf
from nemo.core.config import hydra_runner
import os 

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
    audios = []
    labels = []
    input_dir = "out_sampled_tasks_multispk"
    for file in os.listdir(input_dir):
        filepath = os.path.join(input_dir,file)
        data_dict = load_pickle(filepath)
        
        test_embs = data_dict['test_embs']
        test_labels = data_dict['test_labels']
        test_audios = data_dict['test_audios']
        enroll_embs = data_dict['enroll_embs']
        enroll_labels = data_dict['enroll_labels']
        enroll_audios = data_dict['enroll_audios']

        audios.append(test_audios)
        labels.append(test_labels)
        #analyze_data(test_labels)
        #analyze_data(test_audios)

    audios = np.array(audios)
    labels = np.array(labels)
    print(audios.shape)
    print(labels.shape)
    audios = np.concatenate(audios,axis=0)
    labels = np.concatenate(labels,axis=0)
    print(audios.shape)
    analyze_data(audios)
    print(labels.shape)
    analyze_data(labels)
        
        
        

if __name__ == "__main__":
    main()