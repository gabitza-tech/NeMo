import numpy as np
import sys
from collections import Counter
from utils.task_generator import Tasks_Generator
import random
import torch
import time
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.methods import run_algo_transductive, run_paddle,simpleshot_inductive
from utils.utils import load_pickle, sampler_windows_query, sampler_windows_support, embedding_normalize,compute_acc
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.paddle import PADDLE
from tqdm import tqdm
from methods.simpleshot import Simpleshot

def run_paddle_transductive(enroll_embs,enroll_labels,test_embs,test_labels,method_info):
    
    x_q = torch.tensor(test_embs)
    y_q = torch.tensor(test_labels).long().unsqueeze(2)
    x_s = torch.tensor(enroll_embs)
    y_s = torch.tensor(enroll_labels).long().unsqueeze(2)
    
    task_dic = {}
    task_dic['y_s'] = y_s
    task_dic['y_q'] = y_q
    task_dic['x_s'] = x_s
    task_dic['x_q'] = x_q

    #print(x_q.shape)
    #print(y_q.shape)
    #print(x_s.shape)
    #print(y_s.shape)
    
    method = PADDLE(**method_info)
    logs = method.run_task(task_dic)
    acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])

    return acc_sample

file1 = sys.argv[1]
file2 = sys.argv[2]
enroll_dict = np.load(file1, allow_pickle=True)
test_dict = np.load(file2, allow_pickle=True)

feat = enroll_dict['concat_features']
labels = np.expand_dims(np.asarray(enroll_dict['concat_labels']),axis=1)
slices = np.expand_dims(np.asarray(enroll_dict['concat_slices']),axis=1)
patchs = np.expand_dims(np.asarray(enroll_dict['concat_patchs']),axis=1)

"""
print(feat.shape)
print(labels.shape)
print(slices.shape)
print(patchs.shape)
print(feat[10])
for i in range(100):
    print([labels[i],slices[i],patchs[i]])

labels = enroll_dict['concat_labels']
label_counts = Counter(labels)
# Number of unique labels
num_unique_labels = len(label_counts)

# Minimum and maximum occurrences
min_occurrences = min(label_counts.values())
max_occurrences = max(label_counts.values())

print(f"Number of unique labels: {num_unique_labels}")
print(f"Minimum number of occurrences: {min_occurrences}")
print(f"Maximum number of occurrences: {max_occurrences}")
"""

n_runs = 1
seed = 42

log_file = get_log_file(log_path='tests', backbone='ecapa', dataset='voxceleb1', method='balba')
logger = Logger(__name__, log_file)


for run in range(n_runs):
    
    n_tasks = 10
    batch_size = 2

    run_acc_ind=[]
    run_acc_trans_centroid=[]
    run_acc_trans_l2_sum=[]
    run_acc_EM = []
    run_acc_PADDLE = []
    uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))

    task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                         n_tasks=n_tasks,
                                         n_ways=1251,
                                         n_ways_eff=1,
                                         n_query=3,
                                         k_shot=1,
                                         seed=seed)
        
    start_sample_support = time.time()
    
    test_embs, test_labels, test_audios = task_generator.sampler(test_dict, mode='query')
    enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(enroll_dict, mode='support')
    normalize = True
    # Choose to normalize embeddings or not
    if normalize == True:
        
        all_embs = np.concatenate((enroll_embs,test_embs),axis=1)
        all_embs = embedding_normalize(all_embs)
        enroll_embs = all_embs[:,:enroll_embs.shape[1]]
        test_embs = all_embs[:,enroll_embs.shape[1]:]

        #enroll_embs=embedding_normalize(enroll_embs)
        #test_embs=embedding_normalize(test_embs)

    min_value_enroll = np.min(enroll_embs)
    max_value_enroll = np.max(enroll_embs)
    min_value_test = np.min(test_embs)
    max_value_test = np.max(test_embs)

    #print(min_value_enroll)
    #print(min_value_test)
    #print(max_value_enroll)
    #print(max_value_test)

    #print(test_embs.shape)
    #print(enroll_embs.shape)
    duration_sampling = time.time() - start_sample_support
    print(f"Duration {duration_sampling}s for batch size {batch_size}")

    for start in tqdm(range(0,n_tasks,batch_size)):
        #start = i*batch_size
        end = start+batch_size
        if end > n_tasks:
            end = n_tasks

        x_q = test_embs[start:end]
        y_q = test_labels[start:end]
        x_s = enroll_embs[start:end]
        y_s = enroll_labels[start:end]

        args={}
        args['iter']=30
        args['alpha']=x_q.shape[1]
        args['maj_vote'] = True
        method_info = {'device':'cuda','log_file':log_file,'args':args}
        method = 'simpleshot'
        #print(f"Alpha is equal to {args['alpha']}.")

        if method == "simpleshot":
            
            #pred_labels, pred_labels_top5 = simpleshot_inductive(enroll_embs, enroll_labels, test_embs, avg='mean', backend='ecapa')
            eval = Simpleshot(avg="mean",backend="L2",method='inductive')
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
            #print(acc_list)
            run_acc_ind.extend(acc_list)
            """
            eval = Simpleshot(avg="mean",backend="L2",method='transductive_centroid')
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
            #print(acc_list)
            run_acc_trans_centroid.extend(acc_list)
            eval = Simpleshot(avg="mean",backend="L2",method='transductive_L2_sum')
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
            #print(acc_list)
            run_acc_trans_l2_sum.extend(acc_list)
            eval = Simpleshot(avg="mean",backend="L2",method='EM')
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end])
            #print(acc_list)
            run_acc_EM.extend(acc_list)
            """
            acc_list = run_paddle_transductive(x_s, y_s, x_q, y_q,method_info)
            run_acc_PADDLE.append(acc_list)

    
    ind_acc = sum(run_acc_ind)/len(run_acc_ind)
    print(f"Inductive Simpleshot acc: {ind_acc}")

    paddle_acc = sum(run_acc_PADDLE)/len(run_acc_PADDLE)
    print(f"Paddle acc: {paddle_acc}")
    """
    trans_centroid_acc = sum(run_acc_trans_centroid)/len(run_acc_trans_centroid)
    print(f"Transductive centroid Simpleshot acc:{trans_centroid_acc}")
    trans_l2_sum_acc = sum(run_acc_trans_l2_sum)/len(run_acc_trans_l2_sum)
    print(f"Transductive L2 sum Simpleshot acc:{trans_l2_sum_acc}")
    em_acc = sum(run_acc_EM)/len(run_acc_EM)
    print(f"EM Simpleshot acc: {em_acc}")
    """
    