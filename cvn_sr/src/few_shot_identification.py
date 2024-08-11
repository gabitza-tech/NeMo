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
from tqdm import tqdm
from nemo.core.config import hydra_runner
import random
import logging
import time
from utils.utils import load_pickle,sampler_query,sampler_support, sampler_windows_query,sampler_windows_support, compute_acc,embedding_normalize
from methods.methods import simpleshot

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

    #logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # Create logger file
    if cfg.data.out_file is not None:
        log_name = cfg.data.out_file+"_"+str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
    else:
        log_name = "logger"+"_"+str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)

    logger = setup_logger(f"{log_name}.log")

    # Load embeddings and labels dictionaries for enrollment and test
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
        task_start_time = time.time()
        task_seed = random_numbers[i]
        np.random.seed(task_seed)
        random.seed(task_seed)

        # We sample cfg.n_way classes out of the total number of classes
        sampled_classes = sorted(random.sample(uniq_classes, cfg.n_way))

        test_embs,test_labels = sampler_windows_query(test_dict, sampled_classes)

        enroll_embs, enroll_labels = sampler_windows_support(enroll_dict,sampled_classes,k_shot=cfg.k_shot)
        print(enroll_embs.shape)
        if len(test_embs.shape) == 3:
            test_embs = test_embs.squeeze(1)
        
        #print(test_embs[0])
        #print(enroll_embs[0])

        """
        # Choose to normalize embeddings or not
        if cfg.normalize == True:
            #enroll_embs = enroll_embs / (np.linalg.norm(enroll_embs, ord=2, axis=-1, keepdims=True))
            enroll_embs = embedding_normalize(enroll_embs)
            #test_embs = test_embs / (np.linalg.norm(test_embs, ord=2, axis=-1, keepdims=True))
            test_embs = embedding_normalize(test_embs)
        """
        if cfg.normalize == True:
            
            all_embs = np.concatenate((enroll_embs,test_embs),axis=0)
            all_embs = embedding_normalize(all_embs)
            enroll_embs = all_embs[:enroll_embs.shape[0]]
            test_embs = all_embs[enroll_embs.shape[0]:]
        

        pred_labels = simpleshot(enroll_embs, enroll_labels, test_embs, sampled_classes, avg=cfg.avg, backend=cfg.backend)
        acc = compute_acc(test_labels,pred_labels,sampled_classes)

        task_accs.append(acc)
        logger.info(f"Accuracy for task {i} is {acc}%.")
        computing_duration = time.time() - task_start_time
        print(f"Time taken by task {i} is {computing_duration} s")

    final_acc = sum(task_accs)/len(task_accs)
    logger.info(f"Final accuracy over {cfg.n_tasks} tasks is {final_acc}%.")


if __name__ == '__main__':
    main()
