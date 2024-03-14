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
from pytorch_lightning import seed_everything
import os
from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from scripts.utils import load_pickle
import random
import logging
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize

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

@hydra_runner(config_path="../conf", config_name="speaker_identification_fewshot")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    log_name = os.path.basename(cfg.data.enrollment_embs.split("_support")[0].split("manifest_")[1])
    logger = setup_logger(f"{log_name}.log")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    enroll_dict = load_pickle(cfg.data.enrollment_embs)
    test_dict = load_pickle(cfg.data.test_embs)

    uniq_classes = sorted(list(set(enroll_dict['concat_labels'])))
    if cfg.n_way > len(uniq_classes):
        cfg.n_way = len(uniq_classes)

    # We want to be able to reproduce the classes selected and the ids selected throughout experiments
    random.seed(42)
    random_numbers = [int(random.uniform(0,10000)) for _ in range(cfg.n_tasks)]

    task_accs = []
    for i in range(cfg.n_tasks):
        # Randomly select the classes to be analyzed and the files from each class
        # For each task, we use a certain seed
        task_seed = random_numbers[i]
        random.seed(task_seed)

        
        # we sample cfg.n_way classes out of the total number of classes
        sampled_classes = sorted(random.sample(uniq_classes, cfg.n_way))
        logger.info(f"For task {i}, selected classes are: {sampled_classes}")
        enroll_indices = [index for index, element in enumerate(enroll_dict['concat_labels']) if element in sampled_classes]
        test_indices = [index for index, element in enumerate(test_dict['concat_labels']) if element in sampled_classes]
        
        """
        We first construct the test/enroll, as it is easier and we always use all of it, one file at a time
        We will need a tensor with the embeddings of the files that are part of the sampled_classes: [n,192]
        We will also need a ref_labels vector [n], where each element represents the index of the class from sampled_classes
        """
        print("Creating test embeddings vector and the reference labels")
        ref_labels = [test_dict['concat_labels'][index] for index in test_indices] #[label for index,label in enumerate(test_dict['concat_labels']) if index in test_indices]
        test_embs = test_dict['concat_features'][test_indices] #np.asarray([embs for index,embs in enumerate(test_dict['concat_features']) if index in test_indices])
        

        # We sample the embeddings from k_shot audios in a class
        # If audios are normal, it wil sample exactly k_shot audios per class
        # If audios are split, it will sample a variable number of window_audios, but still coming from k_shot audios
        # We don't oversample embs in a class to equalize the classes lengths, as we calculate the mean anyway
        print("Creating enroll embeddings vector")
        enroll_embs = []
        enroll_labels = []
        max_class_ids = 0
        for label in sampled_classes:
            
            k_shot = cfg.k_shot
            uniq_enroll_class_ids = sorted(set([enroll_dict['concat_slices'][index] for index in enroll_indices if enroll_dict["concat_labels"][index] == label]))
            
            if k_shot > len(uniq_enroll_class_ids):
                k_shot = len(uniq_enroll_class_ids)
            
            sampled_enroll_class_ids = sorted(random.sample(uniq_enroll_class_ids,k_shot))
            enroll_class_embs = np.asarray([enroll_dict['concat_features'][index] for index, enroll_id in enumerate(enroll_dict['concat_slices']) if enroll_id in sampled_enroll_class_ids])
            
            if len(enroll_class_embs) > max_class_ids:
                max_class_ids = len(enroll_class_embs)
            
            enroll_embs.extend(enroll_class_embs)
            enroll_labels.extend([label]*len(enroll_class_embs))
        
        enroll_embs = np.asarray(enroll_embs)
        enroll_labels = np.asarray(enroll_labels)

        if cfg.normalize == True:
            enroll_embs = enroll_embs / (np.linalg.norm(enroll_embs, ord=2, axis=-1, keepdims=True))
            #enroll_embs = embedding_normalize(enroll_embs)
            test_embs = test_embs / (np.linalg.norm(test_embs, ord=2, axis=-1, keepdims=True))
            #test_embs = embedding_normalize(test_embs)

        print("Calculating the reference embeddings")
        # Create reference embeddings, for each class by calcuating the mean
        # also apply normalization if the case
        ref_embs = []
        for label in sampled_classes:
            
            indices = np.where(enroll_labels == label)
            embedding = (enroll_embs[indices].sum(axis=0).squeeze()) / len(indices)
            ref_embs.append(embedding)

        ref_embs = np.asarray(ref_embs)
        
        scores = np.matmul(test_embs, ref_embs.T)
        matched_labels = scores.argmax(axis=-1)

        total_preds = 0
        correct_preds = 0
        for (i,label) in enumerate(matched_labels):
            total_preds += 1
            pred_class = sampled_classes[label]
            if pred_class == ref_labels[i]:
                correct_preds += 1
            
        acc = 100*(correct_preds/total_preds)
        task_accs.append(acc)

    final_acc = sum(task_accs)/len(task_accs)
    logger.info(f"Final accuracy over {cfg.n_tasks} tasks is {final_acc}%.")


if __name__ == '__main__':
    main()
