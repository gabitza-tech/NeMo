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
from utils.utils import CL2N_embeddings
from methods.paddle import PADDLE
from utils.paddle_utils import get_log_file,Logger
from numpy.linalg import norm
from utils.utils import plot_embeddings,class_compute_transform_A


#query_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'#voxmovies_3s_ecapa_embs.pkl'#'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'
query_file = 'saved_embs/voxmovies_3s/voxmovies_3s_ecapa_embs.pkl'
support_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'
classes_file = 'datasets_splits/voxmovies_domain_adapt.txt'

test_dict = np.load(query_file, allow_pickle=True)
enroll_dict = np.load(support_file, allow_pickle=True)   

with open(classes_file,'r') as f:
    lines = f.readlines()
    ids = []
    for line in lines:
        ids.append(line.strip())

labels_count = []
indices_celeb=[]
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

celeb_dict_test = {}
celeb_dict_test['concat_features'] = enroll_dict['concat_features'][indices_celeb]#[500:]
celeb_dict_test['concat_labels'] = np.array(enroll_dict['concat_labels'])[indices_celeb]#[500:]
celeb_dict_test['concat_slices'] = np.array(enroll_dict['concat_slices'])[indices_celeb]#[500:]
celeb_dict_test['concat_patchs'] = np.array(enroll_dict['concat_patchs'])[indices_celeb]

movies_dict = {}
movies_dict['concat_features'] = test_dict['concat_features'][indices_movies]
movies_feat = movies_dict['concat_features']
movies_dict['concat_labels'] = np.array(test_dict['concat_labels'])[indices_movies]
movies_labels = movies_dict['concat_labels']
movies_dict['concat_slices'] = np.array(test_dict['concat_slices'])[indices_movies]
movies_dict['concat_patchs'] = np.array(test_dict['concat_patchs'])[indices_movies]

movies_dict_test = {}
movies_dict_test['concat_features'] = test_dict['concat_features'][indices_movies]#[200:]
movies_dict_test['concat_labels'] = np.array(test_dict['concat_labels'])[indices_movies]#[200:]
movies_dict_test['concat_slices'] = np.array(test_dict['concat_slices'])[indices_movies]#[200:]
movies_dict_test['concat_patchs'] = np.array(test_dict['concat_patchs'])[indices_movies]


uniq_classes = sorted(list(set(celeb_dict['concat_labels'])))

out_dir = "test_voxmovies_filter"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

start = time.time()

seed = 42
n_tasks = 100
batch_size = 20
normalize = True
args={}
args['iter']=20

alpha = 3
n_q = 1
k_shot = 3
n_ways_eff = 1

thetas = [1,0]#[20,15,10,9,8,7,6,5,4,3,2,1.5,1,0.5,0]#,100,250,500,1000,2500,5000,10000,50000,250000]
#A_list = []
final_json = {}
final_json['simpleshot'] = {}
final_json['paddle'] = {}
out_file = os.path.join(out_dir,f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_q}.json')
    
for theta in thetas:
    print(f"Theta:{theta}")
    acc = {}
    acc["simpleshot"] = []
    acc["paddle"] = []
    start = time.time()
    
    
    #celeb_feat, movies_feat = CL2N_embeddings(np.expand_dims(celeb_dict['concat_features'],0),np.expand_dims(movies_dict['concat_features'],0),normalize)
    #celeb_feat = np.squeeze(celeb_feat,0)
    #movies_feat = np.squeeze(movies_feat,0)
    celeb_feat = celeb_dict['concat_features']
    movies_feat = movies_dict['concat_features']
    #A_matrix = compute_transform_A(celeb_feat,movies_feat, theta)
    A_matrix = class_compute_transform_A(movies_feat,movies_labels,celeb_feat,celeb_labels, theta)
    #A_list.append(A_matrix)

    #exit(0)
    movie_avg = np.average(movies_feat,axis=0)
    celeb_avg = np.average(celeb_feat,axis=0)
    movies_feat_adapted = np.matmul(movies_feat,A_matrix.T)
    
    if theta == 0:
        movies_feat_adapted = movies_feat
    
    movies_avg_adapted = np.average(movies_feat_adapted,axis=0)
    plot_embeddings(celeb_avg,movie_avg, movies_avg_adapted, theta)
    
    movies_dict_adapted = {}
    movies_dict_adapted['concat_features'] = movies_feat_adapted
    movies_dict_adapted['concat_labels'] = movies_dict['concat_labels']
    movies_dict_adapted['concat_slices'] = movies_dict['concat_slices']
    movies_dict_adapted['concat_patchs'] = movies_dict['concat_patchs']
    
    celeb_dict['concat_features'] = celeb_feat

    dur = time.time()-start
    print(f"Time taken to compute A:{dur} seconds")

    task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                                n_tasks=n_tasks,
                                                n_ways=len(set(movies_dict['concat_labels'])),
                                                n_ways_eff=n_ways_eff,
                                                n_query=n_q,
                                                k_shot=k_shot,
                                                seed=seed)


    test_embs, test_labels, test_audios = task_generator.sampler(movies_dict_adapted,mode='query')
    enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(celeb_dict,mode='support')
            
    #enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,normalize)

    """"""
    print(len(set(movies_dict['concat_labels'])))
    print(enroll_embs.reshape(-1,192).shape)
    print(test_embs.reshape(-1,192).shape)
    enroll_avg = np.average(enroll_embs.reshape(-1,192),0)
    test_avg = np.average(test_embs.reshape(-1,192),0)
    
    fig = plt.figure(figsize=(21, 7))
    plt.subplot(1, 2, 1)
    plt.hist(test_avg, bins=50, alpha=0.5, label='Movie new')
    plt.hist(enroll_avg, bins=50, alpha=0.5, label='Celeb')
    plt.legend(loc='upper right')
    plt.title(f'Histogram of Movie and Celeb for alpha:{theta}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # KDE plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(test_avg, shade=True, label='Movie new')
    sns.kdeplot(enroll_avg, shade=True, label='Celeb')
    plt.legend(loc='upper right')
    plt.title(f'KDE of Movie and Celeb for alpha:{theta}')
    plt.xlabel('Value')
    plt.ylabel('Density')


    plt.tight_layout()
    plt.show()
    #fig.savefig(f'plot_normalized_alpha_{theta}.png')
    for start in tqdm(range(0,n_tasks,batch_size)):
        end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks
        
        x_q,y_q,x_s,y_s = (test_embs[start:end],
                        test_labels[start:end],
                        enroll_embs[start:end],
                        enroll_labels[start:end])

        if n_ways_eff == 1:
            eval = Simpleshot(avg="mean",backend="L2",method="transductive_centroid")
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
        else:
            eval = Simpleshot(avg="mean",backend="L2",method="inductive")
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 

        acc["simpleshot"].extend(acc_list)

        if n_ways_eff == 1:
            args['maj_vote'] = True
        else:
            args['maj_vote'] = False

        args['alpha'] = alpha
        method_info = {'device':'cuda','args':args}#'log_file':log_file,'args':args}
        acc_list = run_paddle_new(x_s, y_s, x_q, y_q,method_info)
        acc["paddle"].extend(acc_list)

    
    final_json['simpleshot'][str(theta)] = 100*sum(acc["simpleshot"])/len(acc["simpleshot"])

    final_json['paddle'][str(theta)] = 100*sum(acc["paddle"])/len(acc["paddle"])

with open(out_file,'w') as f:
    json.dump(final_json,f)

exit(0)


fig = plt.figure(figsize=(21, 7))
# Histogram plot
plt.subplot(1, 2, 1)
plt.hist(movie_avg, bins=50, alpha=0.5, label='Movie')
plt.hist(movies_avg_adapted, bins=50, alpha=0.5, label='Movie new')
plt.hist(celeb_avg, bins=50, alpha=0.5, label='Celeb')
plt.legend(loc='upper right')
plt.title(f'Histogram of Movie and Celeb for alpha:{theta}')
plt.xlabel('Value')
plt.ylabel('Frequency')

# KDE plot
plt.subplot(1, 2, 2)
sns.kdeplot(movie_avg, shade=True, label='Movie')
sns.kdeplot(movies_avg_adapted, shade=True, label='Movie new')
sns.kdeplot(celeb_avg, shade=True, label='Celeb')
plt.legend(loc='upper right')
plt.title(f'KDE of Movie and Celeb for alpha:{theta}')
plt.xlabel('Value')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
fig.savefig(f'plot_alpha_{alpha}.png')

out_filename = f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}.json'
out_file = os.path.join(out_dir,out_filename)

#log_file = get_log_file(log_path='log_alpha_experiments', method=f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_query}', backbone='ecapa', dataset='voxceleb1')
#logger = Logger(__name__, log_file)

task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                    n_tasks=n_tasks,
                                    n_ways=len(set(enroll_dict['concat_labels'])),
                                    n_ways_eff=n_ways_eff,
                                    n_query=n_query,
                                    k_shot=k_shot,
                                    seed=seed)

#test_embs, test_labels, test_audios = task_generator.sampler(test_dict,mode='query')
enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(enroll_dict,mode='support')
#test_embs, test_labels, test_audios,enroll_embs, enroll_labels, enroll_audios = task_generator.sampler_unified(merged_dict)
test_embs = np.expand_dims(test_dict['concat_features'],0).reshape((9,1,192))
test_labels =[]
label_dict = {label:i for i,label in enumerate(sorted(uniq_classes))}
for label in test_dict['concat_labels']:
    test_labels.append(label_dict[label])
    
test_labels = np.expand_dims(np.array(test_labels),0).reshape((9,1))
test_audios = test_dict['concat_slices']
#enroll_embs, test_embs = CL2N_embeddings(enroll_embs,test_embs,normalize)

acc = {}
acc["simpleshot"] = []
acc["paddle"] = {}
for alpha in alphas:
    acc["paddle"][str(alpha)] = []

for start in tqdm(range(0,n_tasks,batch_size)):
    end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks

    x_q,y_q,x_s,y_s = (test_embs[start:end],
                    test_labels[start:end],
                    enroll_embs[start:end],
                    enroll_labels[start:end])

    if n_ways_eff == 1:
        eval = Simpleshot(avg="mean",backend="L2",method="transductive_centroid")
        acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
    else:
        eval = Simpleshot(avg="mean",backend="L2",method="inductive")
        acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
    acc["simpleshot"].extend(acc_list)

    if n_ways_eff == 1:
        args['maj_vote'] = True
    else:
        args['maj_vote'] = False

    for alpha in alphas:
        args['alpha'] = alpha
        method_info = {'device':'cuda','args':args}#'log_file':log_file,'args':args}
        acc_list = run_paddle_new(x_s, y_s, x_q, y_q,method_info)
        acc["paddle"][str(alpha)].extend(acc_list)

final_json = {}
final_json['simpleshot'] = 100*sum(acc["simpleshot"])/len(acc["simpleshot"])
final_json['paddle'] = {}
print(acc['simpleshot'])
for alpha in alphas:
    print(acc['paddle'])
    final_json['paddle'][str(alpha)] = 100*sum(acc["paddle"][str(alpha)])/len(acc["paddle"][str(alpha)])

with open(out_file,'w') as f:
    json.dump(final_json,f)
                
