import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from collections import Counter
import random
import torch
import json
import time
from utils.task_generator import Tasks_Generator
from tqdm import tqdm
from methods.simpleshot import Simpleshot
from methods.methods import run_paddle_new
import os 
from utils.utils import save_pickle
from utils.utils import CL2N_embeddings
from methods.paddle import PADDLE
from utils.paddle_utils import get_log_file,Logger

no_filter = 'voxmovies_3s_filter_ecapa_embs.pkl'#'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'
filtered = 'voxmovies_3s_filter_no_ecapa_embs.pkl'
celeb_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'
enroll_dict = np.load(no_filter, allow_pickle=True)
test_dict = np.load(filtered, allow_pickle=True)
celeb_dict = np.load(celeb_file,allow_pickle=True)


# Flatten the arrays
no_filter_id = enroll_dict['concat_features'].flatten()
filtered_id = test_dict['concat_features'].flatten()
celeb_id_feat = celeb_dict['concat_features']

indices = []
for i,label in enumerate(celeb_dict['concat_labels']):
    if label == 'id10061':
        indices.append(i)

celeb_id = celeb_id_feat[indices][:9].flatten()
# Plot the distributions
fig = plt.figure(figsize=(21, 7))

print(celeb_id.shape)
print(no_filter_id.shape)
print(filtered_id.shape)

print(celeb_id[0])
print(no_filter_id[0])
print(filtered_id[0])

# Histogram plot
plt.subplot(1, 2, 1)
plt.hist(no_filter_id, bins=50, alpha=0.5, label='No filter Movie')
plt.hist(filtered_id, bins=50, alpha=0.5, label='Filtered Movie')
plt.hist(celeb_id, bins=50, alpha=0.5, label='Celeb')
plt.legend(loc='upper right')
plt.title('Histogram of No filter movie, filtered movie and celeb for id10061')
plt.xlabel('Value')
plt.ylabel('Frequency')

# KDE plot
plt.subplot(1, 2, 2)
sns.kdeplot(no_filter_id, shade=True, label='No filter Movie')
sns.kdeplot(filtered_id, shade=True, label='Filtered Movie')
sns.kdeplot(celeb_id, shade=True, label='Celeb')
plt.legend(loc='upper right')
plt.title('KDE of No filter, Filtered and Celeb')
plt.xlabel('Value')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

fig.savefig('distribution_speaker_id10061.png')