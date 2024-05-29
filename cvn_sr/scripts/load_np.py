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

def run_paddle_transductive(enroll_embs,enroll_labels,test_embs,test_labels,k_shot,method_info, batch_size):
    """
    This function predicts using the PADDLE algorithm over a SINGLE TASK. 
    We can also iterate over the query with a batch_size! 

    INPUT:
    test_embs: (Q_samples, n_patches_sample, feature_dim) shape, Q_samples represents the total number of query samples
    test_labels: (Q_samples,) shape, enrollment labels
    enroll_embs: (S_samples, feature_dim) shape, S_samples represents the total number of support samples
    enroll_labels: (S_samples,) shape, enrollment labels
    k_shot: depending on k_shot, we choose the batch-size
    method_info: arguments for paddle!

    INTERMEDIATE:
    x_q: [len_batch, n_patches_sample, feature_dim]
    y_q: [len_batch, n_patches_sample, 1] (OBSERVATION: it could also be [len_batch,1,1]])
    x_s: [len_batch, S_samples, feature_dim]
    y_s: [len_batch, S_samples, 1]

    RETURN:
    avg_acc_task: average accuracy over the task
    """

    """
    if k_shot == 1:
        query_batch = 512
    elif k_shot == 3:
        query_batch = 256
    elif k_shot == 5:
        query_batch = 128
    else: 
        query_batch = 50
    """  

    query_batch = int(batch_size)
    acc_mean_list = []
    acc_mean_list_top5 = []
    print(query_batch)
    print(test_labels.shape[0])
    for j in tqdm(range(0,test_labels.shape[0],query_batch)):
        
        end = j+query_batch
        if end>test_labels.shape[0]:
            end = test_labels.shape[0]
        if end == j:
            end = j+1

        len_batch = end - j
        print(len_batch)

        x_q = torch.tensor(test_embs[j:end])
        y_q = torch.tensor(test_labels[j:end]).long().unsqueeze(2)#.unsqueeze(2).repeat(1,x_q.shape[1],1) # It can also work with a shape of [len_batch,1,1] (repeat operation is not needed, but it is nicer and clearer in this way)
        x_s = torch.tensor(enroll_embs)#.unsqueeze(0)#.repeat(len_batch,1,1)
        y_s = torch.tensor(enroll_labels).long().unsqueeze(2)#.unsqueeze(0).unsqueeze(2)#.repeat(len_batch,1,1)
        
        task_dic = {}
        task_dic['y_s'] = y_s
        task_dic['y_q'] = y_q
        task_dic['x_s'] = x_s
        task_dic['x_q'] = x_q

        print(x_q.shape)
        print(y_q.shape)
        print(x_s.shape)
        print(y_s.shape)
        
        method = PADDLE(**method_info)
        logs = method.run_task(task_dic)
        acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])
        acc_sample_top5, _ = compute_confidence_interval(logs['acc_top5'][:, -1])
        
        #if acc_sample < 1:
        #    print(acc_sample)
        # Mean accuracy per batch
        print(f"Accuracy of the sample/samples is:{acc_sample}\n")
        acc_mean_list.append(acc_sample*len_batch)
        acc_mean_list_top5.append(acc_sample_top5*len_batch)
    
    avg_acc_task = 100*sum(acc_mean_list)/test_labels.shape[0]
    avg_acc_task_top5 = 100*sum(acc_mean_list_top5)/test_labels.shape[0]
    
    return avg_acc_task, avg_acc_task_top5

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
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_tasks = 100
    batch_size = 20

    run_acc_ind=[]
    run_acc_trans_centroid=[]
    run_acc_trans_l2_sum=[]
    run_acc_5 = []
    uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))
    #print(len(uniq_classes))
    for i in range(int(n_tasks/batch_size)):
        task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                         n_tasks=batch_size,
                                         n_ways=1251,
                                         n_ways_eff=1,
                                         n_query=3,
                                         k_shot=1)
        
        start_sample_support = time.time()
        enroll_embs, enroll_labels = task_generator.sampler(enroll_dict, mode='support')
        test_embs, test_labels = task_generator.sampler(test_dict, mode='query')

        duration_sampling = time.time() - start_sample_support
        print(f"Duration {duration_sampling}s for batch size {batch_size}")

        args={}
        args['iter']=20
        args['alpha']=test_embs.shape[1]
        args['maj_vote'] = True
        method_info = {'device':'cuda','log_file':log_file,'args':args}
        method = 'simpleshot'
        print(f"Alpha is equal to {args['alpha']}.")

        if method == "simpleshot":
            #pred_labels, pred_labels_top5 = simpleshot_inductive(enroll_embs, enroll_labels, test_embs, avg='mean', backend='ecapa')
            eval = Simpleshot(avg="mean",backend="L2",method='inductive')
            acc_list = eval.eval(enroll_embs, enroll_labels, test_embs, test_labels)
            print(acc_list)
            run_acc_ind.extend(acc_list)
            eval = Simpleshot(avg="mean",backend="L2",method='transductive_centroid')
            acc_list = eval.eval(enroll_embs, enroll_labels, test_embs, test_labels)
            print(acc_list)
            run_acc_trans_centroid.extend(acc_list)
            eval = Simpleshot(avg="mean",backend="L2",method='transductive_L2_sum')
            acc_list = eval.eval(enroll_embs, enroll_labels, test_embs, test_labels)
            print(acc_list)
            run_acc_trans_l2_sum.extend(acc_list)

    ind_acc = sum(run_acc_ind)/len(run_acc_ind)
    print(f"Inductive Simpleshot acc: {ind_acc}")
    trans_centroid_acc = sum(run_acc_trans_centroid)/len(run_acc_trans_centroid)
    print(f"Transductive centroid Simpleshot acc:{trans_centroid_acc}")
    trans_l2_sum_acc = sum(run_acc_trans_l2_sum)/len(run_acc_trans_l2_sum)
    print(f"Transductive L2 sum Simpleshot acc:{trans_l2_sum_acc}")

    """ 
            exit(0)
        
        elif method == "paddle":
            acc,acc_top5= run_paddle_transductive(enroll_embs,   
                                  enroll_labels,
                                  test_embs,
                                  test_labels,
                                  5,
                                  method_info,
                                  batch_size)
            acc_latex,acc_top5_latex = run_algo_transductive(enroll_embs,
                                  enroll_labels,
                                  test_embs,
                                  test_labels,
                                  batch_size)
            
            
            
        elif method == "latex":
            acc,acc_top5 = run_algo_transductive(enroll_embs,
                                  enroll_labels,
                                  test_embs,
                                  test_labels,
                                  batch_size)
        
        task_accs.append(acc)
        task_accs5.append(acc_top5)
        task_accs_latex.append(acc_latex)
        task_accs5_latex.append(acc_top5_latex)

        print("---")
        print(acc)
        print(acc_latex)

    print(task_accs)        
    final_avg_acc,final_conf_score = compute_confidence_interval(task_accs)
    final_avg_acc5,final_conf_score5 = compute_confidence_interval(task_accs5)
    logger.info(f"Final Acc for all tasks is {final_avg_acc} and confidence interval:{final_conf_score}")
    logger.info(f"Final Top 5 Acc for all tasks is {final_avg_acc5} and confidence interval:{final_conf_score5}")

       
    all_query = np.concatenate(np.array(all_query),axis=0).reshape(-1)
    print(all_query.shape)

    label_counts = Counter(all_query.tolist())
    # Number of unique labels
    num_unique_labels = len(label_counts)

    # Minimum and maximum occurrences
    min_occurrences = min(label_counts.values())
    max_occurrences = max(label_counts.values())

    print(f"Number of unique labels: {num_unique_labels}")
    print(f"Minimum number of occurrences: {min_occurrences}")
    print(f"Maximum number of occurrences: {max_occurrences}")
    
    avg_dur = sum(all_dur)/len(all_dur)
    print(avg_dur)
    """

    