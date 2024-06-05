import numpy as np
import sys
from collections import Counter
from utils.task_generator import Tasks_Generator
import random
import torch
import time
from methods.paddle import PADDLE
from tqdm import tqdm
from methods.simpleshot import Simpleshot
import os 
from utils.utils import save_pickle

file1 = sys.argv[1]
file2 = sys.argv[2]
seeds = [sys.argv[3]]
enroll_dict = np.load(file1, allow_pickle=True)
test_dict = np.load(file2, allow_pickle=True)

#seeds = [42, 56, 100, 24, 51]

for seed in seeds:
    
    n_tasks = 10000

    run_acc_ind=[]
    run_acc_trans_centroid=[]
    run_acc_trans_l2_sum=[]
    run_acc_EM = []
    uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))

    task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                         n_tasks=n_tasks,
                                         n_ways=1251,
                                         n_ways_eff=1,
                                         n_query=5,
                                         k_shot=2,
                                         seed=int(seed))
        
    start_sample_support = time.time()
    
    test_embs, test_labels, test_audios = task_generator.sampler(test_dict, mode='query')
    enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(enroll_dict, mode='support')

    out_dict = {}
    out_dict['test_embs'] = test_embs
    out_dict['test_labels'] = test_labels
    out_dict['test_audios'] = test_audios
    out_dict['enroll_embs'] = enroll_embs
    out_dict['enroll_labels'] = enroll_labels
    out_dict['enroll_audios'] = enroll_audios

    out_dir = "out_sampled_tasks_nq_5_kshot_2"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    file = os.path.splitext(os.path.basename(file1))[0].split("_support")[0]+"_seed_"+str(seed)+"_tasks_"+str(n_tasks)+".pkl"
    out_file = os.path.join(out_dir,file)
    save_pickle(out_file,out_dict)


