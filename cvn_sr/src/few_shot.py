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

import numpy as np

from omegaconf import OmegaConf

import numpy as np
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
import random
import logging
from tqdm import tqdm
import time
from utils.utils import load_pickle, sampler_windows_query, sampler_windows_support, embedding_normalize,compute_acc
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.methods import run_algo_transductive, run_paddle_transductive, run_paddle,simpleshot

"""
concat_features -- features extracted from an audio
concat_labels -- class of an audio
concat_slices -- root name of an audio, ex: p255_01_window_part_4.wav -> p255_01
concat_patchs -- name of an audio containing the window part too, ex: p255_01_window_part_4.wav -> p255_01_window_part_4

This script only needs data.enrollment_embs, data.test_embs, n_way, k_shot
"""
@hydra_runner(config_path="../conf", config_name="speaker_identification_fewshot.yaml")
def main(cfg):

    #logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # init logger
    if cfg.data.out_file is None:
        setup = str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
    else: 
        setup = cfg.data.out_file + "_" + str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
        
    log_file = get_log_file(log_path=cfg.log_dir, backbone=cfg.backbone, dataset='voxceleb1', method=setup)
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
    task_accs5 = []
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
        print(f"Output sampler shape for test embeddings: {test_embs.shape},{test_labels.shape}")
        enroll_embs, enroll_labels = sampler_windows_support(enroll_dict,sampled_classes,k_shot=cfg.k_shot)
        print(f"Output sampler shape for enroll embeddings: {enroll_embs.shape}, {enroll_labels.shape}")

        duration_sampling = time.time() - task_start_time     
        print(duration_sampling)

        """
        # Choose to normalize embeddings or not
        if cfg.normalize == True:
            initial_shape = test_embs.copy().shape
            if len(initial_shape) == 3 and cfg.method=="simpleshot":
                test_embs =test_embs.reshape((-1, test_embs.shape[-1]))
            #enroll_embs = enroll_embs / (np.linalg.norm(enroll_embs, ord=2, axis=-1, keepdims=True))
            enroll_embs = embedding_normalize(enroll_embs)
            #test_embs = test_embs / (np.linalg.norm(test_embs, ord=2, axis=-1, keepdims=True))
            test_embs = embedding_normalize(test_embs)
        """

        # Choose to normalize embeddings or not
        if cfg.normalize == True:
            initial_shape = test_embs.copy().shape
            if len(initial_shape) == 3:
                test_embs =test_embs.reshape((-1, test_embs.shape[-1]))
            
            all_embs = np.concatenate((enroll_embs,test_embs),axis=0)
            all_embs = embedding_normalize(all_embs)
            enroll_embs = all_embs[:enroll_embs.shape[0]]
            test_embs = all_embs[enroll_embs.shape[0]:]

            if len(initial_shape) == 3 and cfg.method != "simpleshot":
                test_embs = test_embs.reshape(initial_shape)
        
        
        args={}
        args['iter']=20
        args['alpha']=test_embs.shape[1]
        print(f"Alpha is equal to {args['alpha']}.")

        args['maj_vote'] = True
        
        method_info = {'device':'cuda','log_file':log_file,'args':args}
        if cfg.method == "simpleshot":
            pred_labels, pred_labels_top5 = simpleshot(enroll_embs, enroll_labels, test_embs, sampled_classes, avg=cfg.avg, backend=cfg.backend)
            acc,acc_top5 = compute_acc(pred_labels,pred_labels_top5,test_labels,sampled_classes)
        elif cfg.method == "paddle":
            acc,acc_top5= run_paddle_transductive(enroll_embs,  
                                  enroll_labels,
                                  test_embs,
                                  test_labels,
                                  cfg.k_shot,
                                  method_info,
                                  cfg.batch_size)
        elif cfg.method == "latex":
            acc,acc_top5 = run_algo_transductive(enroll_embs,
                                  enroll_labels,
                                  test_embs,
                                  test_labels,
                                  cfg.k_shot,
                                  method_info,
                                  cfg.batch_size)
        
        logger.info(f"Accuracy for task {i} is {acc}%") #and top 5 is {acc_top5}%.")
        computing_duration = time.time() - task_start_time
        print(f"Time taken by task {i} is {computing_duration} s")
        task_accs.append(acc)
        task_accs5.append(acc_top5)
        
    final_avg_acc,final_conf_score = compute_confidence_interval(task_accs)
    final_avg_acc5,final_conf_score5 = compute_confidence_interval(task_accs5)
    logger.info(f"Final Acc for all tasks is {final_avg_acc} and confidence interval:{final_conf_score}")
    logger.info(f"Final Top 5 Acc for all tasks is {final_avg_acc5} and confidence interval:{final_conf_score5}")

if __name__ == '__main__':
    main()
