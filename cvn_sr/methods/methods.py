import os
import numpy as np
from tqdm import tqdm
import torch
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.paddle import PADDLE
from methods.tim import ALPHA_TIM, TIM_GD

def simpleshot(enroll_embs,enroll_labels,test_embs,sampled_classes,method="mean"):

    print("Using SimpleShot method")
    # Calculate the mean embeddings for each class in the support
    avg_enroll_embs = []

    for label in range(len(sampled_classes)):
        
        indices = np.where(enroll_labels == label)
        if method == "normal":
            embedding = (enroll_embs[indices].sum(axis=0).squeeze()) / len(indices[0])
        if method == "median":
            embedding = np.median(enroll_embs[indices], axis=0)

        avg_enroll_embs.append(embedding)
    
    avg_enroll_embs = np.asarray(avg_enroll_embs)
    
    # Calculate cosine similarity between test embeddings and the transpose of the averaged class embeddings
    scores = np.matmul(test_embs, avg_enroll_embs.T)
    pred_labels = scores.argmax(axis=-1)

    return pred_labels

def run_paddle(enroll_embs,enroll_labels,test_embs,test_labels,k_shot,method_info):
    """
    label_dict = {}
    for i,label in enumerate(sampled_classes):
        label_dict[label]=i
    
    new_test_labels = []
    for label in test_labels:
        new_test_labels.append(label_dict[label])
    new_test_labels = np.asarray(new_test_labels)

    new_enroll_labels = []
    for label in enroll_labels:
        new_enroll_labels.append(label_dict[label])
    new_enroll_labels = np.asarray(new_enroll_labels)
    """
    acc_mean_list = []
    acc_conf_list = []

    if k_shot == 1:
        query_batch = 512
    elif k_shot == 3:
        query_batch = 256
    elif k_shot == 5:
        query_batch = 128
    else: # k_shot == 10
        query_batch = 50
    
    for j in tqdm(range(0,test_labels.shape[0],query_batch)):
    #for j in tqdm(range(test_labels.shape[0])):
        
        end = j+query_batch
        if end>test_labels.shape[0]-1:
            end = test_labels.shape[0]-1

        len_batch = end - j

        #x_q = torch.tensor([test_embs[j]]).unsqueeze(0)
        x_q = torch.tensor(test_embs[j:end]).unsqueeze(1)
        #y_q = torch.tensor([new_test_labels[j]]).long().unsqueeze(0).unsqueeze(2)
        y_q = torch.tensor([test_labels[j:end]]).long().view(-1,1).unsqueeze(2)
        #x_s = torch.tensor(enroll_embs).unsqueeze(0)
        x_s = torch.tensor(enroll_embs).unsqueeze(0).repeat(len_batch,1,1)
        #y_s = torch.tensor(new_enroll_labels).long().unsqueeze(0).unsqueeze(2)
        y_s = torch.tensor(enroll_labels).long().unsqueeze(0).unsqueeze(2).repeat(len_batch,1,1)

        task_dic = {}
        task_dic['y_s'] = y_s
        task_dic['y_q'] = y_q
        task_dic['x_s'] = x_s
        task_dic['x_q'] = x_q

        method = PADDLE(**method_info)
        logs = method.run_task(task_dic)
        acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])

        # Mean accuracy per batch
        print(acc_sample)
        acc_mean_list.append(acc_sample*len_batch)
        
    avg_acc_task = sum(acc_mean_list)/test_labels.shape[0]

    return avg_acc_task

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
    for j in tqdm(range(0,test_labels.shape[0],query_batch)):
        
        end = j+query_batch
        if end>test_labels.shape[0]-1:
            end = test_labels.shape[0]-1
        if end == j:
            end = j+1

        len_batch = end - j

        x_q = torch.tensor(test_embs[j:end])
        y_q = torch.tensor(test_labels[j:end]).long().unsqueeze(1).unsqueeze(2).repeat(1,x_q.shape[1],1) # It can also work with a shape of [len_batch,1,1] (repeat operation is not needed, but it is nicer and clearer in this way)
        x_s = torch.tensor(enroll_embs).unsqueeze(0).repeat(len_batch,1,1)
        y_s = torch.tensor(enroll_labels).long().unsqueeze(0).unsqueeze(2).repeat(len_batch,1,1)

        task_dic = {}
        task_dic['y_s'] = y_s
        task_dic['y_q'] = y_q
        task_dic['x_s'] = x_s
        task_dic['x_q'] = x_q
    
        method = PADDLE(**method_info)
        logs = method.run_task(task_dic)
        acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])

        #if acc_sample < 1:
        #    print(acc_sample)
        # Mean accuracy per batch
        print(f"Accuracy of the sample/samples is:{acc_sample}\n")
        acc_mean_list.append(acc_sample*len_batch)
        #acc_mean_list.append(acc_sample)
        
    avg_acc_task = sum(acc_mean_list)/test_labels.shape[0]
    
    return avg_acc_task

def run_tim(enroll_embs,enroll_labels,test_embs,test_labels,k_shot,method_info):
    """
    label_dict = {}
    for i,label in enumerate(sampled_classes):
        label_dict[label]=i
    
    new_test_labels = []
    for label in test_labels:
        new_test_labels.append(label_dict[label])
    new_test_labels = np.asarray(new_test_labels)

    new_enroll_labels = []
    for label in enroll_labels:
        new_enroll_labels.append(label_dict[label])
    new_enroll_labels = np.asarray(new_enroll_labels)
    """
    acc_mean_list = []
    acc_conf_list = []

    if k_shot == 1:
        query_batch = 512
    elif k_shot == 3:
        query_batch = 256
    elif k_shot == 5:
        query_batch = 128
    else: # k_shot == 10
        query_batch = 50
    
    query_batch = 128
    for j in tqdm(range(0,test_labels.shape[0],query_batch)):
    #for j in tqdm(range(test_labels.shape[0])):
        
        end = j+query_batch
        if end>test_labels.shape[0]-1:
            end = test_labels.shape[0]-1

        len_batch = end - j

        #x_q = torch.tensor([test_embs[j]]).unsqueeze(0)
        x_q = torch.tensor(test_embs[j:end]).unsqueeze(1)
        #y_q = torch.tensor([new_test_labels[j]]).long().unsqueeze(0).unsqueeze(2)
        y_q = torch.tensor([test_labels[j:end]]).long().view(-1,1).unsqueeze(2)
        #x_s = torch.tensor(enroll_embs).unsqueeze(0)
        x_s = torch.tensor(enroll_embs).unsqueeze(0).repeat(len_batch,1,1)
        #y_s = torch.tensor(new_enroll_labels).long().unsqueeze(0).unsqueeze(2)
        y_s = torch.tensor(enroll_labels).long().unsqueeze(0).unsqueeze(2).repeat(len_batch,1,1)

        task_dic = {}
        task_dic['y_s'] = y_s
        task_dic['y_q'] = y_q
        task_dic['x_s'] = x_s
        task_dic['x_q'] = x_q

        method = ALPHA_TIM(**method_info)
        logs = method.run_task(task_dic)
        acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])

        # Mean accuracy per batch
        print(acc_sample)
        acc_mean_list.append(acc_sample*len_batch)
        
    avg_acc_task = sum(acc_mean_list)/test_labels.shape[0]

    return avg_acc_task