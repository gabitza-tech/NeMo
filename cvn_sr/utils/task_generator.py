import torch
import numpy as np
from omegaconf import OmegaConf
import random
from utils.utils import find_matching_positions
from tqdm import tqdm
import time

class Tasks_Generator:
    def __init__(self, uniq_classes, n_tasks=1, n_ways=1251, n_ways_eff=1, n_query=1, k_shot=1):
        """
        uniq_classes: all labels of classes in the support set [List]
        n_tasks: number of generated tasks
        n_ways: number of sampled classes in the support (number of support classes)
        n_ways_eff: number of sampled classes in the query, are part of the support classes (closed set) and much fewer (number of query classes)
        n_query: number of samples per query class
        k_shot: number of samples per support class
        """
        self.uniq_classes=sorted(uniq_classes)
        # Convert all class ids to integer values.

        self.n_tasks = n_tasks 
        self.n_ways= n_ways
        self.n_ways_eff = n_ways_eff
        self.n_query = n_query
        self.k_shot = k_shot

        self.support_classes = []
        self.query_classes = []
        for i in range(n_tasks):
            sampled_classes=sorted(random.sample(self.uniq_classes, n_ways))
            self.support_classes.append(sampled_classes)
            # Query classes must be part of the sampled support classes
            self.query_classes.append(sorted(random.sample(sampled_classes, n_ways_eff)))
        
        #print(self.support_classes)
        #print(self.query_classes)
    
    def sampler(self, data_dict, mode):
        """
        There are 2 modes: query and support. Depending on the mode, we either load the sampled support/query classes for n_tasks
        """
        out_embs = []
        out_labels = []

        if mode == "support":
            tasks_classes = self.support_classes
            no_samples = self.k_shot
        else:
            tasks_classes = self.query_classes
            no_samples = self.n_query
            

        for task, sampled_classes in tqdm(enumerate(tasks_classes)):
            
            # Get indices of samples that are part of the sampled classes in the support for this task.
            # The query must use the same indices as the support!
            
            self.label_dict = {label:i for i,label in enumerate(self.support_classes[task])}

            # Get the indices where elements in concat_labels are in sampled_classes
            data_label_indices = np.where(np.isin(np.array(data_dict['concat_labels']), sampled_classes))[0].tolist()
            
            all_labels = np.asarray(data_dict['concat_labels'])[data_label_indices]
            all_slices = np.asarray(data_dict['concat_slices'])[data_label_indices]
            all_patchs = np.asarray(data_dict['concat_patchs'])[data_label_indices]
            all_embs = np.asarray(data_dict['concat_features'])[data_label_indices]

            combined_array = np.column_stack((all_labels, all_slices))
            unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)
            
            random_pairs = [(label, np.random.choice(unique_pairs[unique_pairs[:, 0] == label, 1], size=no_samples, replace=False)) for label in sorted(sampled_classes)]
            random_pairs_array = np.concatenate([[[label, id_] for id_ in ids] for label, ids in random_pairs])

            data_indices = np.array(find_matching_positions(combined_array, random_pairs_array))

            data_embs = all_embs[data_indices]
            data_labels = all_labels[data_indices]
            data_labels = np.asarray([self.label_dict[label] for label in data_labels])
            
            out_embs.append(data_embs)
            out_labels.append(data_labels)

            #print("Sampled classes: "+str([self.label_dict[label] for label in sampled_classes]))
            #print("Labels: " + str(data_labels))

        out_embs = np.array(out_embs)
        out_labels = np.array(out_labels)

        return out_embs, out_labels
    