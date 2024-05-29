import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import compute_acc
import torch.nn.functional as F
import random

class Simpleshot():
    def __init__(self,avg="mean",backend="l2", majority="True",device='cpu', method="inductive"):
        self.avg = avg
        self.backend = backend
        self.majority = majority
        self.device = torch.device(device)
        self.method = method

    def eval(self,enroll_embs,enroll_labels,test_embs,test_labels):

        if self.method == "inductive":
            pred_labels, pred_labels_5 = self.inductive(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "transductive_centroid":
            pred_labels, pred_labels_5 = self.transductive_centroid(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "transductive_L2_sum":
            pred_labels, pred_labels_5 = self.transductive_L2_sum(enroll_embs,enroll_labels,test_embs,test_labels)
        elif self.method == "EM":
            pred_labels, pred_labels_5 = self.estimation_maximization(enroll_embs,enroll_labels,test_embs,test_labels)
    
        test_labels = torch.from_numpy(test_labels).long()
        acc_tasks = compute_acc(pred_labels, test_labels)

        return acc_tasks

    def calculate_centrois(self,enroll_embs,enroll_labels):
        # Returns [n_tasks,n_ways,192] tensor with the centroids
        # sampled_classes: [n_tasks,n_ways]
        
        sampled_classes=[]
        for task in enroll_labels:
            sampled_classes.append(sorted(list(set(task))))

        avg_enroll_embs = []
        for i,task_classes in enumerate(sampled_classes):
            task_enroll_embs = []
            for label in task_classes:
                indices = np.where(enroll_labels[i] == label)
                if self.avg == "mean":
                    embedding = (enroll_embs[i][indices].sum(axis=0).squeeze()) / len(indices[0])
                if self.avg == "median":
                    embedding = np.median(enroll_embs[i][indices], axis=0)
                task_enroll_embs.append(embedding)
            avg_enroll_embs.append(task_enroll_embs)

        avg_enroll_embs = np.asarray(avg_enroll_embs)

        return avg_enroll_embs

    def inductive(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        # Calculate the mean embeddings for each class in the support
        avg_enroll_embs = self.calculate_centrois(enroll_embs, enroll_labels)

        test_embs = torch.from_numpy(test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
      
        if self.backend == "cosine":
            print("Using SimpleShot inductive method with cosine similarity backend")

            scores = torch.einsum('ijk,ilk->ijl', test_embs, avg_enroll_embs)

            pred_labels = torch.argmax(scores, dim=-1).long()#.tolist()
            _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=True)
            
        else:
            print("Using SimpleShot inductive method with L2 norm backend")
            test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

            # Class distance
            dist = (test_embs-avg_enroll_embs)**2
            C_l = torch.sum(dist,dim=-1) # [n_tasks,n_query,1251]

            pred_labels = torch.argmin(C_l, dim=-1).long()#.tolist()
            _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

        return pred_labels, pred_labels_top5
    
    def transductive_centroid(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        # Calculate the mean embeddings for each class in the support

        n_query = test_embs.shape[1]
        avg_enroll_embs = self.calculate_centrois(enroll_embs, enroll_labels)
        avg_test_embs = self.calculate_centrois(test_embs, test_labels)

        avg_test_embs = torch.from_numpy(avg_test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
        
        if self.backend == "cosine":
            print("Using SimpleShot transductive centroid method with cosine similarity backend.")

            scores = torch.einsum('ijk,ilk->ijl', avg_test_embs, avg_enroll_embs).repeat(1,n_query,1)
            pred_labels = torch.argmax(scores, dim=-1).long()
            _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=True)
            
        else:
            print("Using SimpleShot transductive centroid method with L2 norm backend.")
            avg_test_embs = torch.unsqueeze(avg_test_embs,2) # [n_tasks,n_query,1,emb_shape]
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

            # Class distance
            dist = (avg_test_embs-avg_enroll_embs)**2
            C_l = torch.sum(dist,dim=-1).repeat(1,n_query,1) # [n_tasks,n_query,1251]

            pred_labels = torch.argmin(C_l, dim=-1).long()
            _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

        return pred_labels, pred_labels_top5
    
    def transductive_L2_sum(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        n_query = test_embs.shape[1]
        # Calculate the mean embeddings for each class in the support

        avg_enroll_embs = self.calculate_centrois(enroll_embs, enroll_labels)
        test_embs = torch.from_numpy(test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
      
        print("Using SimpleShot transductive L2_sum method with L2 norm backend")
        test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]
        avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

        # Class distance
        dist = torch.sum((test_embs-avg_enroll_embs)**2,dim=-1)
        # We sum the distance of all the samples in the query, then repeat it in order to have the same n_query as the test labels
        C_l = torch.unsqueeze(torch.sum(dist,dim=1),dim=1).repeat(1,n_query,1) # [n_tasks, 1251]

        pred_labels = torch.argmin(C_l, dim=-1).long()
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)
        
        return pred_labels, pred_labels_top5
    
    def estimation_maximization(self,enroll_embs,enroll_labels,test_embs,test_labels):
        print("Using Estimation maximization method")
        n_query = test_embs.shape[1]
        x_sq = np.concatenate((enroll_embs,test_embs),1)
        y_sq = np.concatenate((enroll_labels,test_labels),1)

        w_s = self.calculate_centrois(enroll_embs, enroll_labels)
        w_sq = self.calculate_centrois(x_sq, y_sq)
        
        device_here = torch.device('cuda:0')
        w_s = torch.from_numpy(w_s).float().to(self.device).unsqueeze(1).to(device_here)
        w_sq = torch.from_numpy(w_sq).float().to(self.device).unsqueeze(1).to(device_here)
        z_s = torch.from_numpy(np.expand_dims(enroll_embs,2)).float().to(device_here)
        z_q = torch.from_numpy(np.expand_dims(test_embs,2)).float().to(device_here)

        #print(w_sq.shape)
        #print(w_s.shape)
        #print(z_s.shape)
        #print(z_q.shape)
        #print("-"*5)
        # Calculate sum_s and sum_q in PyTorch
        sum_s = torch.sum(((w_sq - z_s)**2 -(w_s - z_s)**2),dim=1)
        sum_q = torch.sum((w_sq - z_q)**2,dim=1)
        #print(sum_s.shape)
        #print(sum_q.shape)
        # Calculate C_l
        C_l = torch.sum(sum_s+sum_q,dim=-1)
        #print(C_l.shape)
        pred_labels = torch.argmin(C_l, dim=-1).unsqueeze(1).repeat(1,n_query).to(torch.device('cpu'))
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=1, largest=False)

        return pred_labels, pred_labels_top5

def compute_acc(pred_labels, test_labels):

    # Check if the input tensors have the same shape
    assert pred_labels.shape == test_labels.shape, "Shape mismatch between predicted and groundtruth labels"
    # Calculate accuracy for each task
    acc_list = (pred_labels == test_labels).float().mean(dim=1).tolist()
    
    return acc_list

