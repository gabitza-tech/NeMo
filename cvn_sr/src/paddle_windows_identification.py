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
import random
import logging
from tqdm import tqdm
import time
from utils.utils import load_pickle, sampler_windows_query, sampler_windows_support, embedding_normalize
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.methods import run_paddle_transductive, run_paddle

"""
concat_features -- features extracted from an audio
concat_labels -- class of an audio
concat_slices -- root name of an audio, ex: p255_01_window_part_4.wav -> p255_01
concat_patchs -- name of an audio containing the window part too, ex: p255_01_window_part_4.wav -> p255_01_window_part_4
"""
"""
This script only needs data.enrollment_embs, data.test_embs, n_way, k_shot
"""
@hydra_runner(config_path="../conf", config_name="paddle_identification_fewshot.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # init logger
    if cfg.data.out_file is None:
        method = str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
    else: 
        method = cfg.data.out_file + "_" + str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
        
    log_file = get_log_file(log_path="logs_paddle", backbone='ecapa', dataset='voxceleb1', method=method)
    logger = Logger(__name__, log_file)

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

        test_embs,test_labels  = sampler_windows_query(test_dict, sampled_classes)
        print(f"Output sampler shape for test embeddings: {test_embs.shape}")
        enroll_embs, enroll_labels = sampler_windows_support(enroll_dict,sampled_classes,k_shot=cfg.k_shot)
        print(f"Output sampler shape for enroll embeddings: {enroll_embs.shape}")

        duration_sampling = time.time() - task_start_time     
        print(duration_sampling)

        # Choose to normalize embeddings or not
        if cfg.normalize == True:
            #enroll_embs = enroll_embs / (np.linalg.norm(enroll_embs, ord=2, axis=-1, keepdims=True))
            enroll_embs = embedding_normalize(enroll_embs)
            #test_embs = test_embs / (np.linalg.norm(test_embs, ord=2, axis=-1, keepdims=True))
            test_embs = embedding_normalize(test_embs)

        args={}
        args['iter']=20
        args['alpha']=1
        method_info = {'device':'cuda:0','log_file':log_file,'args':args}
        
        avg_acc_task = run_paddle_transductive(enroll_embs,
                                  enroll_labels,
                                  test_embs,
                                  test_labels,
                                  cfg.k_shot,
                                  method_info)

        logger.info(f"Accuracy for task {i} is {avg_acc_task}%.")
        computing_duration = time.time() - task_start_time
        print(f"Time taken by task {i} is {computing_duration} s")
        task_accs.append(avg_acc_task)
        
    final_avg_acc,final_conf_score = compute_confidence_interval(task_accs)
    logger.info(f"Final Acc for all tasks is {final_avg_acc} and confidence interval:{final_conf_score}")


if __name__ == '__main__':
    main()
