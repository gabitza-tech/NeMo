import numpy as np
from utils.task_generator import Tasks_Generator
import torch
import time
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from utils.utils import save_pickle,load_pickle, embedding_normalize, analyze_data
from utils.paddle_utils import get_log_file,Logger,compute_confidence_interval
from methods.paddle import PADDLE
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from omegaconf import OmegaConf
import os 
from collections import Counter, defaultdict
import random

random.seed(42)

#@hydra_runner(config_path="../conf", config_name="speaker_identification_fewshot.yaml")
def create_lists():
    voxmovies_pkl = 'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'#"saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl"
    voxceleb_s_pkl = 'saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'#"saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl"
    voxceleb_q_pkl = 'saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'#"saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl"
    
    voxmovies_dict = np.load(voxmovies_pkl, allow_pickle=True)
    voxceleb_s = np.load(voxceleb_s_pkl, allow_pickle=True)
    voxceleb_q = np.load(voxceleb_q_pkl, allow_pickle=True)
    voxceleb_dict = {}
    for key in voxceleb_s.keys():
        voxceleb_dict[key] = np.concatenate((voxceleb_s[key],voxceleb_q[key]),axis=0) 

    #labels = merged_dict['concat_labels']
    labels_movies = voxmovies_dict['concat_labels']
    labels_voxceleb = voxceleb_dict['concat_labels']

    print(len(set(labels_movies)))
    print(len(set(labels_voxceleb)))

    out_dir = 'datasets_splits'
    labels_movies_file = os.path.join(out_dir,'voxmovies_257_labels.txt')
    with open(labels_movies_file,'a') as f:
        for label in sorted(set(labels_movies)):
            f.write(label+"\n")
    exit(0)
    labels_celeb1_file = os.path.join(out_dir,'voxceleb1_all_labels.txt')
    with open(labels_celeb1_file,'a') as f:
        for label in sorted(set(labels_voxceleb)):
            f.write(label+"\n")

    remaining_voxceleb = [label for label in set(labels_voxceleb) if label not in set(labels_movies)]
    print(len(remaining_voxceleb))    

    validation_voxceleb1 = sorted(random.sample(remaining_voxceleb, 125))
    print(len(validation_voxceleb1))

    val_celeb1_file = os.path.join(out_dir,'voxceleb1_val_labels.txt')
    with open(val_celeb1_file,'a') as f:
        for label in sorted(set(validation_voxceleb1)):
            f.write(label+"\n")

    test_voxceleb = sorted([label for label in set(labels_voxceleb) if label not in set(validation_voxceleb1)])
    print(len(test_voxceleb))    

    test_celeb1_file = os.path.join(out_dir,'voxceleb1_test_labels.txt')
    with open(test_celeb1_file,'a') as f:
        for label in sorted(set(test_voxceleb)):
            f.write(label+"\n")

def get_embeddings():
    voxmovies_pkl = 'datasets_splits/embeddings/voxmovies_5s_ecapa_embs_257.pkl'#"saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl"
    voxceleb_s_pkl = 'saved_embs/voxceleb1_5s/voxceleb1_5s_support_ecapa_embs.pkl'#"saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl"
    voxceleb_q_pkl = 'saved_embs/voxceleb1_5s/voxceleb1_5s_query_ecapa_embs.pkl'#"saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl"
    
    out_val = 'datasets_splits/embeddings/voxceleb1_5s_val_ecapa_embs.pkl'
    out_test = 'datasets_splits/embeddings/voxceleb1_5s_test_ecapa_embs.pkl'
    out_movies = 'datasets_splits/embeddings/voxceleb1_5s_movies_ecapa_embs_257.pkl'

    voxmovies_dict = np.load(voxmovies_pkl, allow_pickle=True)
    voxceleb_s = np.load(voxceleb_s_pkl, allow_pickle=True)
    voxceleb_q = np.load(voxceleb_q_pkl, allow_pickle=True)
    voxceleb_dict = {}
    for key in voxceleb_s.keys():
        voxceleb_dict[key] = np.concatenate((voxceleb_s[key],voxceleb_q[key]),axis=0) 

    with open('datasets_splits/voxmovies_257_labels.txt','r') as f:
        lines = f.readlines()
    voxmovies_labels = sorted([label.strip() for label in lines])

    with open('datasets_splits/voxceleb1_test_labels.txt','r') as f:
        lines = f.readlines()
    voxceleb1_test_labels = sorted([label.strip() for label in lines])

    with open('datasets_splits/voxceleb1_val_labels.txt','r') as f:
        lines = f.readlines()
    voxceleb1_val_labels = sorted([label.strip() for label in lines])

    indices_val = [index for index,label in enumerate(voxceleb_dict['concat_labels']) if label in voxceleb1_val_labels]
    indices_test = [index for index,label in enumerate(voxceleb_dict['concat_labels']) if label in voxceleb1_test_labels]
    indices_movies = [index for index,label in enumerate(voxceleb_dict['concat_labels']) if label in voxmovies_labels]

    voxceleb1_val ={}
    voxceleb1_movies ={}
    voxceleb1_test ={}
    for key in voxceleb_dict.keys():
        voxceleb1_val[key] = voxceleb_dict[key][indices_val]
        voxceleb1_test[key] = voxceleb_dict[key][indices_test]
        voxceleb1_movies[key] = voxceleb_dict[key][indices_movies]

    #save_pickle(out_val,voxceleb1_val)
    #save_pickle(out_test,voxceleb1_test)
    save_pickle(out_movies,voxceleb1_movies)    

def clean_voxmovies():
    voxmovies_pkl = 'saved_embs/voxmovies_5s/voxmovies_5s_ecapa_embs.pkl'#"saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl"
    voxmovies_dict = np.load(voxmovies_pkl, allow_pickle=True)
    
    labels = voxmovies_dict['concat_labels']
    label_counts = Counter(labels)
    
    # Get the top 10 labels with the least samples
    least_common_labels = label_counts.most_common()

    # Create a dictionary to store indices for each label
    label_indices = defaultdict(list)

    # Populate the dictionary with indices for each label
    for index, label in enumerate(labels):
        label_indices[label].append(index)

    # Create a result list containing label, count, and indices
    result = []
    final_indices = []
    for label, count in least_common_labels:
        result.append({
            'label': label,
            'count': count,
            'indices': label_indices[label]
        })

        if count >= 5:
            final_indices.extend(label_indices[label])

    print(len(final_indices))

    new_dict = {}
    for key in voxmovies_dict.keys():
        print(key)
        if type(voxmovies_dict[key])==list:
            new_dict[key] = [voxmovies_dict[key][index] for index in final_indices]
        else:
            new_dict[key] = voxmovies_dict[key][final_indices]

    print(new_dict['concat_features'].shape)
    print(len(set(new_dict['concat_labels'])))
    save_pickle('datasets_splits/embeddings/voxmovies_5s_ecapa_embs_cleaned.pkl',new_dict)

if __name__ == "__main__":
    #create_lists()
    get_embeddings()
    #clean_voxmovies()