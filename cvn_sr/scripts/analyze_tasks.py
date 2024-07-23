import numpy as np
from utils.task_generator import Tasks_Generator
import torch
import time
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from utils.utils import load_pickle, embedding_normalize, analyze_data
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.paddle import PADDLE
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from omegaconf import OmegaConf
import os 
from collections import Counter, defaultdict
import sys

#@hydra_runner(config_path="../conf", config_name="speaker_identification_fewshot.yaml")
def main():
    audios = []
    labels = []
    input_dir = "out_sampled_tasks_multispk"
    for file in os.listdir(input_dir):
        filepath = os.path.join(input_dir,file)
        data_dict = load_pickle(filepath)
        
        test_embs = data_dict['test_embs']
        test_labels = data_dict['test_labels']
        test_audios = data_dict['test_audios']
        enroll_embs = data_dict['enroll_embs']
        enroll_labels = data_dict['enroll_labels']
        enroll_audios = data_dict['enroll_audios']

        audios.append(test_audios)
        labels.append(test_labels)
        #analyze_data(test_labels)
        #analyze_data(test_audios)

    audios = np.array(audios)
    labels = np.array(labels)
    print(audios.shape)
    print(labels.shape)
    audios = np.concatenate(audios,axis=0)
    labels = np.concatenate(labels,axis=0)
    print(audios.shape)
    analyze_data(audios)
    print(labels.shape)
    analyze_data(labels)
        
def main2():

    #voxceleb1_path = sys.argv[1]
    voxmovies_path = sys.argv[1]
    
    #input_dict2 = np.load(voxceleb1_path,allow_pickle=True)
    input_dict = np.load(voxmovies_path,allow_pickle=True)

    labels = input_dict['concat_labels']
    label_counts = Counter(labels)
    # Number of unique labels
    num_unique_labels = len(label_counts)
    unique_labels = set(labels)

    # Minimum and maximum occurrences
    min_occurrences = min(label_counts.values())
    max_occurrences = max(label_counts.values())
    avg = sum(label_counts.values())/len(label_counts)

    #print(f"Number of samples in dataset: {len(labels)}")
    #rint(f"Number of unique labels: {num_unique_labels}")
    #print(f"Minimum number of occurrences: {min_occurrences}")
    #print(f"Maximum number of occurrences: {max_occurrences}")
    #print(f'Average number of occurences: {avg}')

        # Get the top 10 labels with the least samples
    least_common_labels = label_counts.most_common()[-120:]

    # Create a dictionary to store indices for each label
    label_indices = defaultdict(list)

    # Populate the dictionary with indices for each label
    for index, label in enumerate(labels):
        label_indices[label].append(index)

    # Create a result list containing label, count, and indices
    result = []
    for label, count in least_common_labels:
        result.append({
            'label': label,
            'count': count,
            'indices': label_indices[label]
        })

    # Print the result
    for item in result:
        print(f"Label: {item['label']}, Count: {item['count']}, Indices: {item['indices']}")

    exit(0)

    task_generator = Tasks_Generator(uniq_classes=unique_labels,
                                            n_tasks=10,
                                            n_ways=num_unique_labels,
                                            n_ways_eff=3,
                                            n_query=1,
                                            k_shot=3,
                                            seed=42)
        
    start_sample_support = time.time()
    test_embs, test_labels, test_audios = task_generator.sampler(input_dict, mode="query")
    enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(input_dict2, mode="support")
    #test_embs, test_labels, test_audios,enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(input_dict)

    print(test_audios.shape)
    analyze_data(test_audios)
    print(test_labels.shape)
    analyze_data(test_labels)
    print(enroll_audios.shape)
    analyze_data(enroll_audios)
    print(enroll_labels.shape)
    analyze_data(enroll_labels)

if __name__ == "__main__":
    main2()