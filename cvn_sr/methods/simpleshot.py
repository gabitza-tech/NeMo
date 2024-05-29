import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import compute_acc
import torch.nn.functional as F

class Simpleshot():
    def __init__(self,avg="mean",backend="l2", majority="True",device='cpu'):
        self.avg = avg
        self.backend = backend
        self.majority = majority
        self.device = torch.device(device)

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
        print("Using SimpleShot inductive method")
        # Calculate the mean embeddings for each class in the support

        avg_enroll_embs = self.calculate_centrois(enroll_embs, enroll_labels)
        test_embs = torch.from_numpy(test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
      
        if self.backend == "cosine":
            print("Using cosine similarity")

            scores = torch.einsum('ijk,ilk->ijl', test_embs, avg_enroll_embs)

            pred_labels = torch.argmax(scores, dim=-1)#.tolist()
            _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=True)
            
        else:
            print("Using L2 norm")
            test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

            # Class distance
            dist = (test_embs-avg_enroll_embs)**2
            C_l = torch.sum(dist,dim=-1) # [n_tasks,n_query,1251]

            pred_labels = torch.argmin(C_l, dim=-1)#.tolist()
            _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

        print(test_labels)
        print(pred_labels)
        #print(pred_labels_top5)

        return pred_labels, pred_labels_top5
    
    def transductive_centroid(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        print("Using SimpleShot transductive centroid method")
        # Calculate the mean embeddings for each class in the support

        n_query = test_embs.shape[1]
        avg_enroll_embs = self.calculate_centrois(enroll_embs, enroll_labels)
        avg_test_embs = self.calculate_centrois(test_embs, test_labels)

        avg_test_embs = torch.from_numpy(avg_test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
        
        if self.backend == "cosine":
            print("Using cosine similarity")

            scores = torch.einsum('ijk,ilk->ijl', avg_test_embs, avg_enroll_embs).repeat(1,n_query,1)
            pred_labels = torch.argmax(scores, dim=-1)#.tolist()
            _,pred_labels_top5 = torch.topk(scores, k=5, dim=-1, largest=True)
            
        else:
            print("Using L2 norm")
            test_embs = torch.unsqueeze(avg_test_embs,2) # [n_tasks,n_query,1,emb_shape]
            avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

            # Class distance
            dist = (avg_test_embs-avg_enroll_embs)**2
            C_l = torch.sum(dist,dim=-1).repeat(1,n_query,1) # [n_tasks,n_query,1251]

            pred_labels = torch.argmin(C_l, dim=-1)#.tolist()
            _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

        print(test_labels)
        print(pred_labels)
        #print(pred_labels_top5)

        return pred_labels, pred_labels_top5
    
    def transductive_L2_sum(self,enroll_embs,enroll_labels,test_embs,test_labels):
        """
        enroll_embs: [n_tasks,k_shot*n_ways,192]
        enroll_labels: [n_tasks,k_shot*n_ways]
        test_embs: [n_tasks,n_query,192]
        test_labels: [n_tasks,n_query]
        """
        n_query = test_embs.shape[1]
        print("Using SimpleShot transductive L2_dist sum method")
        # Calculate the mean embeddings for each class in the support

        avg_enroll_embs = self.calculate_centrois(enroll_embs, enroll_labels)
        test_embs = torch.from_numpy(test_embs).float().to(self.device)
        avg_enroll_embs = torch.from_numpy(avg_enroll_embs).float().to(self.device)
      
        print("Using L2 norm")
        test_embs = torch.unsqueeze(test_embs,2) # [n_tasks,n_query,1,emb_shape]
        avg_enroll_embs = torch.unsqueeze(avg_enroll_embs,1) # [n_tasks,1,1251,emb_shape]

        # Class distance
        dist = torch.sum((test_embs-avg_enroll_embs)**2,dim=-1)
        # We sum the distance of all the samples in the query, then repeat it in order to have the same n_query as the test labels
        C_l = torch.unsqueeze(torch.sum(dist,dim=1),dim=1).repeat(1,5,1) # [n_tasks, 1251]

        pred_labels = torch.argmin(C_l, dim=-1)#.tolist()
        _,pred_labels_top5 = torch.topk(C_l, k=5, dim=-1, largest=False)

        print(test_labels)
        print(pred_labels)
        #print(pred_labels_top5)

        return pred_labels, pred_labels_top5