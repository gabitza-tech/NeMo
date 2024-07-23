import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter
import json
import time
from utils.task_generator import Tasks_Generator
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from methods.methods import run_paddle_new
import os 
from utils.utils import save_pickle
from utils.utils import tsne_domains_per_class,tsne_domains_multi_class
from utils.utils import CL2N_embeddings, embedding_normalize
from utils.paddle_utils import get_log_file,Logger
from numpy.linalg import norm
from utils.utils import plot_embeddings,class_compute_transform_A,class_compute_diagonal_A,class_compute_sums_A
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

#query_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'#voxmovies_3s_ecapa_embs.pkl'#'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'
query_file = 'saved_embs/voxmovies_3s/voxmovies_3s_ecapa_embs.pkl'
#query_file = 'saved_embs/voxceleb1_3s_dur_1.5_ovl_0.75/voxceleb1_3s_dur_1.5_ovl_0.75_support_ecapa_embs.pkl'
#query_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'
support_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'

classes_file = 'datasets_splits/voxmovies_257_labels.txt'#

test_dict = np.load(query_file, allow_pickle=True)
enroll_dict = np.load(support_file, allow_pickle=True)   

with open(classes_file,'r') as f:
    lines = f.readlines()
    ids = []
    for line in lines: 
        ids.append(line.strip())

labels_count = []
indices_celeb = []
indices_movies = []

for i,label in enumerate(test_dict['concat_labels']):
    if label in ids:
        indices_movies.append(i)

for i,label in enumerate(enroll_dict['concat_labels']):
    if label in ids:
        indices_celeb.append(i)

celeb_dict = {}
celeb_dict['concat_features'] = enroll_dict['concat_features'][indices_celeb]
celeb_feat = celeb_dict['concat_features']
celeb_dict['concat_labels'] = np.array(enroll_dict['concat_labels'])[indices_celeb]
celeb_labels = celeb_dict['concat_labels']
celeb_dict['concat_slices'] = np.array(enroll_dict['concat_slices'])[indices_celeb]
celeb_dict['concat_patchs'] = np.array(enroll_dict['concat_patchs'])[indices_celeb]

movies_dict = {}
movies_dict['concat_features'] = test_dict['concat_features'][indices_movies]
movies_feat = movies_dict['concat_features']
movies_dict['concat_labels'] = np.array(test_dict['concat_labels'])[indices_movies]
movies_labels = movies_dict['concat_labels']
movies_dict['concat_slices'] = np.array(test_dict['concat_slices'])[indices_movies]
movies_dict['concat_patchs'] = np.array(test_dict['concat_patchs'])[indices_movies]

normalize = True
use_mean = True

if normalize == True:
    #movies_feat,celeb_feat = CL2N_embeddings(np.expand_dims(movies_feat,0),np.expand_dims(celeb_feat,0),normalize,use_mean=True)   
    movies_feat = embedding_normalize(movies_feat,use_mean=use_mean)
    celeb_feat = embedding_normalize(celeb_feat,use_mean=use_mean)
    movies_feat = np.squeeze(movies_feat)
    celeb_feat = np.squeeze(celeb_feat)

uniq_classes = sorted(list(set(movies_dict['concat_labels'])))

print(celeb_feat.shape)
print(movies_feat.shape)

#tsne_domains_per_class(celeb_feat,movies_feat,celeb_dict['concat_labels'],movies_dict['concat_labels'])

#exit(0)
for i in range(1):
    # Randomly select 5 classes from uniq_classes
    random_classes = random.sample(list(uniq_classes), 250)
    tsne_domains_multi_class(celeb_feat,movies_feat,celeb_dict['concat_labels'],movies_dict['concat_labels'],random_classes)