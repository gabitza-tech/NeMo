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
from methods.paddle_old import PADDLE
from methods.utils import get_log_file,Logger,compute_confidence_interval
from tqdm import tqdm
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
@hydra_runner(config_path="../conf", config_name="paddle_identification_fewshot.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    """
    if cfg.data.out_file is not None:
        log_name = cfg.data.out_file+"_"+str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
    else:
        log_name = "logger"+"_"+str(cfg.n_way)+"_"+str(cfg.k_shot)+"_"+str(cfg.n_tasks)
    """
    # init logger
    log_file = get_log_file(log_path=cfg.data.out_file, dataset='voxceleb1',
                            backbone='ecapa', method='test_'+str(cfg.n_way)+"_"+str(cfg.k_shot))
    logger_paddle = Logger(__name__, log_file)

    #logger = setup_logger(f"{log_name}.log")

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
    for i_task in range(cfg.n_tasks):
        # Randomly select the classes to be analyzed and the files from each class
        # For each task, we use a certain seed
        task_seed = random_numbers[i_task]
        random.seed(task_seed)

        # We sample cfg.n_way classes out of the total number of classes
        sampled_classes = sorted(random.sample(uniq_classes, cfg.n_way))
        logger_paddle.info(f"For task {i_task}, number of classes are {len(sampled_classes)}")# and selected classes are: {sampled_classes}")
        
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
        
        test_labels = np.asarray(ref_labels)

        print(test_labels.shape)
        print(test_embs.shape)
        print(enroll_labels.shape)
        print(enroll_embs.shape)

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

        args={}
        args['iter']=20
        args['alpha']=1
        method_info = {'device':'cuda:0','log_file':log_file,'args':args}
        
        acc_mean_list = []
        acc_conf_list = []
        batch_size = 32
        for j in tqdm(range(0,test_labels.shape[0],batch_size)):
        #for j in tqdm(range(test_labels.shape[0])):
            end = j+batch_size
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
            logs = method.run_task(task_dic,cfg.k_shot)
            acc_sample, _ = compute_confidence_interval(logs['acc'][:, -1])

            # Mean accuracy per batch
            acc_mean_list.append(acc_sample)
            
        avg_acc_task,_ = compute_confidence_interval(acc_mean_list)

        logger_paddle.info(f"Acc for task {i_task} is {avg_acc_task}")
        task_accs.append(avg_acc_task)
        print(avg_acc_task)
    
    final_avg_acc,final_conf_score = compute_confidence_interval(task_accs)
    logger_paddle.info(f"Final Acc for all tasks is {final_avg_acc} and confidence interval:{final_conf_score}")


if __name__ == '__main__':
    main()
