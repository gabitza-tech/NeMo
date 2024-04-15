import os
import numpy as np
from tqdm import tqdm
import torch
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.paddle import PADDLE

def simpleshot(enroll_embs,enroll_labels,test_embs,sampled_classes,method="mean"):

    print("Using SimpleShot method")
    # Calculate the mean embeddings for each class in the support
    avg_enroll_embs = []

    for label in sampled_classes:
        
        indices = np.where(enroll_labels == label)
        if method == "normal":
            embedding = (enroll_embs[indices].sum(axis=0).squeeze()) / len(indices)
        if method == "median":
            embedding = np.median(enroll_embs[indices], axis=0)

        avg_enroll_embs.append(embedding)
    
    avg_enroll_embs = np.asarray(avg_enroll_embs)
    
    # Calculate cosine similarity between test embeddings and the transpose of the averaged class embeddings
    scores = np.matmul(test_embs, avg_enroll_embs.T)
    matched_labels = scores.argmax(axis=-1)
    pred_labels = [sampled_classes[label] for label in matched_labels]

    return pred_labels

def run_paddle(enroll_embs,enroll_labels,test_embs,test_labels,sampled_classes,k_shot,method_info):
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
        y_q = torch.tensor([new_test_labels[j:end]]).long().view(-1,1).unsqueeze(2)
        #x_s = torch.tensor(enroll_embs).unsqueeze(0)
        x_s = torch.tensor(enroll_embs).unsqueeze(0).repeat(len_batch,1,1)
        #y_s = torch.tensor(new_enroll_labels).long().unsqueeze(0).unsqueeze(2)
        y_s = torch.tensor(new_enroll_labels).long().unsqueeze(0).unsqueeze(2).repeat(len_batch,1,1)

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