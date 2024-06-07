import numpy as np
import sys
from collections import Counter
import random
import torch
import json
import time
from utils.task_generator import Tasks_Generator
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from methods.methods import run_paddle_new
import os 
from utils.utils import save_pickle
from utils.utils import CL2N_embeddings
from methods.paddle import PADDLE
from utils.paddle_utils import get_log_file,Logger

query_file = "saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl"
support_file = "saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl"
enroll_dict = np.load(support_file, allow_pickle=True)
test_dict = np.load(query_file, allow_pickle=True)
merged_dict = {}
for key in enroll_dict.keys():
    merged_dict[key] = np.concatenate((enroll_dict[key],test_dict[key]),axis=0)

seed = 42
n_tasks = 500
batch_size = 20
normalize = True
args={}
args['iter']=20

alphas = [i for i in range(0, 76) if i % 3 == 0 or i % 5 == 0]
n_queries = [15,10,5,3,1]
k_shots = [5,3,1]
n_ways_effs = [5,3,1]

uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))

out_dir = "log_alpha_experiments_500_ways"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for k_shot in k_shots:
    for n_ways_eff in n_ways_effs:
        for n_query in n_queries:
            
            out_filename = f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}.json'
            out_file = os.path.join(out_dir,out_filename)
            
            #log_file = get_log_file(log_path='log_alpha_experiments', method=f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}', backbone='ecapa', dataset='voxceleb1')
            #logger = Logger(__name__, log_file)
            
            task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                                n_tasks=n_tasks,
                                                n_ways=500,
                                                n_ways_eff=n_ways_eff,
                                                n_query=n_query,
                                                k_shot=k_shot,
                                                seed=seed)

            test_embs, test_labels, test_audios,enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(merged_dict)
            enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,normalize)

            acc = {}
            acc["simpleshot"] = []
            acc["paddle"] = {}
            for alpha in alphas:
                acc["paddle"][str(alpha)] = []

            for start in tqdm(range(0,n_tasks,batch_size)):
                end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

                x_q,y_q,x_s,y_s = (test_embs[start:end],
                                test_labels[start:end],
                                enroll_embs[start:end],
                                enroll_labels[start:end])

                if n_ways_eff == 1:
                    eval = Simpleshot(avg="mean",backend="L2",method="transductive_centroid")
                    acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                else:
                    eval = Simpleshot(avg="mean",backend="L2",method="inductive")
                    acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
                acc["simpleshot"].extend(acc_list)

                if n_ways_eff == 1:
                    args['maj_vote'] = True
                else:
                    args['maj_vote'] = False

                for alpha in alphas:
                    args['alpha'] = alpha
                    method_info = {'device':'cuda','args':args}#'log_file':log_file,'args':args}
                    acc_list = run_paddle_new(x_s, y_s, x_q, y_q,method_info)
                    acc["paddle"][str(alpha)].extend(acc_list)

            final_json = {}
            final_json['simpleshot'] = 100*sum(acc["simpleshot"])/len(acc["simpleshot"])
            final_json['paddle'] = {}
            for alpha in alphas:
                final_json['paddle'][str(alpha)] = 100*sum(acc["paddle"][str(alpha)])/len(acc["paddle"][str(alpha)])

            with open(out_file,'w') as f:
                json.dump(final_json,f)
                
