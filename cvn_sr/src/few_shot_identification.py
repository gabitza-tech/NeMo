# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
import torch
from omegaconf import OmegaConf
import os

from nemo.core.config import hydra_runner
from nemo.utils import logging
from scripts.utils import load_pickle
import random
import logging
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from scripts.utils import majority_element
# Set the seed to a fixed value (for reproducibility)



def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger("short_audios_logger")
    logger.setLevel(logging.INFO)

    # Create a file handler and set the level to INFO
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set the format for the handler
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


"""
concat_features -- features extracted from an audio
concat_labels -- class of an audio
concat_slices -- root name of an audio, ex: p255_01_window_part_4.wav -> p255_01
concat_patchs -- name of an audio containing the window part too, ex: p255_01_window_part_4.wav -> p255_01_window_part_4
"""
"""
This script only needs data.enrollment_embs, data.test_embs, n_way, k_shot
"""
@hydra_runner(config_path="../conf", config_name="speaker_identification_fewshot")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.data.out_file is not None:
        log_name = cfg.data.out_file+"_"+str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
    else:
        log_name = "logger"+"_"+str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)

    logger = setup_logger(f"{log_name}.log")

    enroll_dict = load_pickle(cfg.data.enrollment_embs)
    test_dict = load_pickle(cfg.data.test_embs)


    # We check how many classes we have in support (closed set problem -> we assume that query classes are part of support classes)
    uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))
    
    if cfg.n_way > len(uniq_classes):
        cfg.n_way = len(uniq_classes)
    
    print(f"Number of ways is :{cfg.n_way}")
    
    # We want to be able to reproduce the classes selected and the ids selected throughout experiments
    # For each task, we have a certain seed => for same enroll, test, k-shot and n-way, we should obtain same results!
    random.seed(42)
    random_numbers = [int(random.uniform(0,10000)) for _ in range(cfg.n_tasks)]

    task_accs = []
    for i in range(cfg.n_tasks):
        # Randomly select the classes to be analyzed and the files from each class
        # For each task, we use a certain seed
        task_seed = random_numbers[i]
        random.seed(task_seed)

        # We sample cfg.n_way classes out of the total number of classes
        sampled_classes = sorted(random.sample(uniq_classes, cfg.n_way))
        #logger.info(f"For task {i}, number of classes are {len(sampled_classes)} and selected classes are: {sampled_classes}")
        
        # We find which indices in the lists are part of the sampled classes
        test_indices = [index for index, element in enumerate(test_dict['concat_labels']) if element in sampled_classes]
        enroll_indices = [index for index, element in enumerate(enroll_dict['concat_labels']) if element in sampled_classes]
        
        print(f"There are {len(test_indices)} test samples that belong to the sampled classes")
        print(f"There are {len(enroll_indices)} enroll samples that belong to the sampled classes")
        
        """
        We first construct the test/enroll, as it is easier and we always use all of it, one file at a time
        We will need a tensor with the embeddings of the files that are part of the sampled_classes: [n,192]
        We will also need a ref_labels vector [n], where each element represents a class from sampled_classes
        """
        print("Creating test embeddings vector and the reference labels")
        test_embs = test_dict['concat_features'][test_indices] #np.asarray([embs for index,embs in enumerate(test_dict['concat_features']) if index in test_indices])
        test_ids = [test_dict['concat_slices'][index] for index in test_indices]
        ref_labels = [test_dict['concat_labels'][index] for index in test_indices] #[label for index,label in enumerate(test_dict['concat_labels']) if index in test_indices]
        

        """
        We sample the embeddings from k_shot audios in each sampled class
        If audios are normal, it wil sample exactly k_shot audios per class
        If audios are split, it will sample a variable number of window_audios, but still coming from k_shot audios
        We don't oversample embs in a class to equalize the classes lengths at the moment, as we calculate the mean anyway
        """
        print("Creating enroll embeddings vector")
        enroll_embs = []
        enroll_labels = []
        max_class_ids = 0
        for label in sampled_classes:
            
            # We get all the uniq audio filenames in the sampled support class (label)
            # In case the audios are split, that's not a problem, because it will get the windows too.
            k_shot = cfg.k_shot
            uniq_ids_enroll_class = sorted(set([enroll_dict['concat_slices'][index] for index in enroll_indices if enroll_dict["concat_labels"][index] == label]))        
            
            # In case we do not have enough audio files per class, we lower k_shot for this class at the moment            
            if k_shot > len(uniq_ids_enroll_class):
                k_shot = len(uniq_ids_enroll_class)
            
            # We extract the class embs as the embeddings coming from the sampled audios
            sampled_enroll_class_ids = sorted(random.sample(uniq_ids_enroll_class,k_shot))
            
            class_embs = np.asarray([enroll_dict['concat_features'][index] for index, enroll_id in enumerate(enroll_dict['concat_slices']) if ((enroll_id in sampled_enroll_class_ids) and (enroll_dict['concat_labels'][index]==label))])
            
            # We don't use it yet, as we do not oversample in the case of using windows yet
            if len(class_embs) > max_class_ids:
                max_class_ids = len(class_embs)
            
            enroll_embs.extend(class_embs)
            enroll_labels.extend([label]*len(class_embs))
        
        enroll_embs = np.asarray(enroll_embs)
        enroll_labels = np.asarray(enroll_labels)

        # Choose to normalize embeddings or not

        if cfg.normalize == True:
            #enroll_embs = enroll_embs / (np.linalg.norm(enroll_embs, ord=2, axis=-1, keepdims=True))
            enroll_embs = embedding_normalize(enroll_embs)
            #test_embs = test_embs / (np.linalg.norm(test_embs, ord=2, axis=-1, keepdims=True))
            test_embs = embedding_normalize(test_embs)


        print("Calculating the mean class embeddings")
        # Calculate the mean embeddings for each class in the support
        avg_enroll_embs = []

        for label in sampled_classes:
            
            indices = np.where(enroll_labels == label)
            if cfg.method == "normal":
                embedding = (enroll_embs[indices].sum(axis=0).squeeze()) / len(indices)
            if cfg.method == "median":
                embedding = np.median(enroll_embs[indices], axis=0)
            
            

            avg_enroll_embs.append(embedding)
        
        avg_enroll_embs = np.asarray(avg_enroll_embs)

        print(avg_enroll_embs.shape)
        
        # Calculate cosine similarity between test embeddings and the transpose of the averaged class embeddings
        scores = np.matmul(test_embs, avg_enroll_embs.T)
        matched_labels = scores.argmax(axis=-1)

        """
        Calculate score per audio, it's only one audio per file id in case of normal audios
        In case of split audios (windows) there are multiple audios per file id, so majority voting must be performed
        """

        """
        uniq_test_ids = sorted(list(set(test_ids))) # We go through each audio and get the majority class for it
        
        test_results = {}
        for uniq_id in uniq_test_ids:
            test_results[uniq_id] = {}
            ids_indices = [index for index,id1 in enumerate(test_ids) if id1==uniq_id]

            test_results[uniq_id]["ref"] = ref_labels[ids_indices[0]]
            test_results[uniq_id]["pred"] = sampled_classes[majority_element(matched_labels[ids_indices])]

        total_preds = 0
        correct_preds = 0
        for key in test_results.keys():
            total_preds += 1
            if test_results[key]["ref"] == test_results[key]["pred"]:
                correct_preds += 1
        """
        total_preds = 0
        correct_preds = 0

        class_acc = {}
        for cls in sampled_classes:
            class_acc[cls] = {}
            class_acc[cls]['total_preds'] = 0
            class_acc[cls]['correct_preds'] = 0
            class_acc[cls]['preds'] = []

        # label in matched_labels is the position of a class in sampled_classes, from argmax
        # matched_labels and ref_labels have the same size.
        for (j,label) in enumerate(matched_labels):
            total_preds += 1
            class_acc[ref_labels[j]]['total_preds'] +=1

            pred_class = sampled_classes[label]
            class_acc[ref_labels[j]]['preds'].append(pred_class)

            if pred_class == ref_labels[j]:
                correct_preds += 1
                class_acc[ref_labels[j]]['correct_preds'] +=1

        acc = 100*(correct_preds/total_preds)
        task_accs.append(acc)
        logger.info(f"Accuracy for task {i} is {acc}%.")

    final_acc = sum(task_accs)/len(task_accs)
    logger.info(f"Final accuracy over {cfg.n_tasks} tasks is {final_acc}%.")


if __name__ == '__main__':
    main()
