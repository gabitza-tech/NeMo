import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter
import json
import time
from utils.task_generator import Tasks_Generator
from tqdm import tqdm
from methods.simpleshot import Simpleshot,compute_acc
from methods.methods import run_paddle_new
import os 
from utils.utils import save_pickle
from utils.utils import CL2N_embeddings, embedding_normalize
from utils.paddle_utils import get_log_file,Logger
from numpy.linalg import norm
from utils.utils import coral,calculate_centroids,ana_A,iterative_A,plot_embeddings,class_compute_transform_A,class_compute_diagonal_A,class_compute_sums_A
import sys
import random
import torch
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.utils import tsne_query_support
from numpy.linalg import norm


method = sys.argv[1]

#query_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'#voxmovies_3s_ecapa_embs.pkl'#'datasets_splits/embeddings/voxmovies_3s_ecapa_embs_257.pkl'
query_file = 'saved_embs/voxmovies_3s/voxmovies_3s_ecapa_embs.pkl'
support_file = 'saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'

classes_file = 'datasets_splits/voxmovies_domain_adapt.txt'#'datasets_splits/voxmovies_257_labels.txt'#'datasets_splits/voxceleb1_test_labels.txt'#
test_classes_file = 'datasets_splits/voxmovies_257_labels.txt'#'datasets_splits/voxmovies_domain_adapt.txt'#

test_dict = np.load(query_file, allow_pickle=True)
enroll_dict = np.load(support_file, allow_pickle=True)   

with open(classes_file,'r') as f:
    lines = f.readlines()
    ids = []
    for line in lines: 
        ids.append(line.strip())


with open(test_classes_file,'r') as f:
    lines = f.readlines()
    test_ids = []
    for line in lines:
        test_ids.append(line.strip())

labels_count = []
indices_celeb = []
indices_movies = []

indices_celeb_test = []
indices_movies_test = []

for i,label in enumerate(test_dict['concat_labels']):
    if label in ids:
        indices_movies.append(i)
    if label in test_ids:
        indices_movies_test.append(i)

for i,label in enumerate(enroll_dict['concat_labels']):
    if label in ids:
        indices_celeb.append(i)
    if label in test_ids:
        indices_celeb_test.append(i)

celeb_dict = {}
celeb_dict['concat_features'] = enroll_dict['concat_features'][indices_celeb]
celeb_feat = celeb_dict['concat_features']
celeb_dict['concat_labels'] = np.array(enroll_dict['concat_labels'])[indices_celeb]
celeb_labels = celeb_dict['concat_labels']
celeb_dict['concat_slices'] = np.array(enroll_dict['concat_slices'])[indices_celeb]
celeb_dict['concat_patchs'] = np.array(enroll_dict['concat_patchs'])[indices_celeb]

celeb_dict_test = {}
celeb_dict_test['concat_features'] = enroll_dict['concat_features'][indices_celeb_test]
celeb_dict_test['concat_labels'] = np.array(enroll_dict['concat_labels'])[indices_celeb_test]
celeb_dict_test['concat_slices'] = np.array(enroll_dict['concat_slices'])[indices_celeb_test]
celeb_dict_test['concat_patchs'] = np.array(enroll_dict['concat_patchs'])[indices_celeb_test]

movies_dict = {}
movies_dict['concat_features'] = test_dict['concat_features'][indices_movies]
movies_feat = movies_dict['concat_features']
movies_dict['concat_labels'] = np.array(test_dict['concat_labels'])[indices_movies]
movies_labels = movies_dict['concat_labels']
movies_dict['concat_slices'] = np.array(test_dict['concat_slices'])[indices_movies]
movies_dict['concat_patchs'] = np.array(test_dict['concat_patchs'])[indices_movies]

movies_dict_test = {}
movies_dict_test['concat_features'] = test_dict['concat_features'][indices_movies_test]
movies_dict_test['concat_labels'] = np.array(test_dict['concat_labels'])[indices_movies_test]
movies_dict_test['concat_slices'] = np.array(test_dict['concat_slices'])[indices_movies_test]
movies_dict_test['concat_patchs'] = np.array(test_dict['concat_patchs'])[indices_movies_test]
    
print(movies_feat.shape)
print(celeb_feat.shape)
print(movies_dict_test['concat_features'].shape)

out_dir = f'voxmovies_normalized_diagonal_validation_{method}'#"voxmovies_test_no_normalization"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

start = time.time()

seed = 42
n_tasks = 10000
batch_size = 10000

args={}
args['iter']=20

normalize = True
use_mean = True

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

alpha = 1
n_q = 3
k_shot = 3
n_ways_eff = 1

#thetas = [100,90,80,70,60,50,40,30,2010,1,0.1,0,0.5,-0.5,-1,-5,-10,-100,9999]#[1000000,100000000,1000,100,10000,1,10,0.5,0,-1,-100,-1000.-1000000,-1500000]
#thetas = [1000000000,100000000,10000000,1000000,100000,10000,1000,100,10,0,-10,-100,-1000,-10000,-100000,-1000000,-10000000,-100000000,-1000000000]#[9999,-1000000,-1500000,-1700000,-1300000,-1800000,-1900000,-2000000]
thetas = [1000]#[i for i in range(-200,-100,1)]#[i for i in range(800,900,1)]
thetas.append("no_adapt")

final_json = {}
final_json['simpleshot'] = {}
final_json['simpleshot_5'] = {}
final_json['paddle'] = {}
final_json['2stage_paddle'] = {}
out_file = os.path.join(out_dir,f'k_{k_shot}_neff_{n_ways_eff}_nq_{n_q}.json')
    
#for theta in thetas:
#    print(f"Theta:{theta}")
acc = {}
acc["simpleshot"] = []
acc["paddle"] = []
start = time.time()

print(movies_feat.shape)
print(celeb_feat.shape)

if normalize == True:
    #movies_feat,celeb_feat = CL2N_embeddings(np.expand_dims(movies_feat,0),np.expand_dims(celeb_feat,0),normalize,use_mean=True)   
    movies_feat = embedding_normalize(movies_feat,use_mean=use_mean)
    celeb_feat = embedding_normalize(celeb_feat,use_mean=use_mean)
    movies_feat = np.squeeze(movies_feat)
    celeb_feat = np.squeeze(celeb_feat)
       
if method == "diag":
    #A_matrix = class_compute_diagonal_A(movies_feat,movies_labels,celeb_feat,celeb_labels, 999)
    sum_up,sum_down = class_compute_sums_A(movies_feat,movies_labels,celeb_feat,celeb_labels)
    
elif method == "norm":
    #A_matrix = class_compute_transform_A(movies_feat,movies_labels,celeb_feat,celeb_labels, theta)
    sum_up,sum_down = class_compute_transform_A(movies_feat,movies_labels,celeb_feat,celeb_labels)


movies_dict_adapted = {}
movies_dict_adapted['concat_features'] = movies_dict_test['concat_features']
movies_dict_adapted['concat_labels'] = movies_dict_test['concat_labels']
movies_dict_adapted['concat_slices'] = movies_dict_test['concat_slices']
movies_dict_adapted['concat_patchs'] = movies_dict_test['concat_patchs']

dur = time.time()-start
print(f"Time taken to compute A:{dur} seconds")

uniq_classes = sorted(list(set(movies_dict_adapted['concat_labels'])))

print(len(set(movies_dict_adapted['concat_labels'])))
task_generator = Tasks_Generator(uniq_classes=uniq_classes,
                                n_tasks=n_tasks,
                                n_ways=len(set(movies_dict_adapted['concat_labels'])),
                                n_ways_eff=n_ways_eff,
                                n_query=n_q,
                                k_shot=k_shot,
                                seed=seed)

test_embs, test_labels, test_audios = task_generator.sampler(movies_dict_adapted,mode='query')
enroll_embs, enroll_labels, enroll_audios = task_generator.sampler(celeb_dict_test,mode='support')

if normalize == True:
    #enroll_embs, initial_test_embs = CL2N_embeddings(enroll_embs,test_embs,normalize,use_mean=True)    
    initial_test_embs = embedding_normalize(test_embs,use_mean=use_mean)
    initial_enroll_embs = embedding_normalize(enroll_embs,use_mean=use_mean)
else:
    initial_test_embs = np.copy(test_embs)
    initial_test_embs = np.copy(enroll_embs)

for theta in thetas:
    acc = {}
    acc["simpleshot"] = []
    acc["simpleshot_5"] = []
    acc["paddle"] = []
    
    test_embs= np.copy(initial_test_embs)
    enroll_embs = np.copy(initial_enroll_embs)
    #A_matrix = np.zeros((192,192))
    if theta != "no_adapt":
    #    for j in range(test_embs.shape[-1]):       
    #        A_matrix[j,j] = sum_up[j][j]/(sum_down[j][j]+theta)
    #   print(A_matrix)
        #sum_down_x = sum_down + np.eye(192)*theta
        #matrix_inverse = np.linalg.inv(sum_down_x)
        #A_matrix = np.matmul(sum_up,matrix_inverse)
        #test_embs = np.matmul(test_embs, A_matrix.T)
        # Iterative A
        # Voxceleb closer to VoxMovies
        sampled_classes=sorted(list(set(celeb_labels)))
        sampled_classes_dict = {label:i for i,label in enumerate(sampled_classes)}
        print('--')
        mu = calculate_centroids(movies_feat,movies_labels)#(celeb_feat,celeb_labels)#
        processed_labels_q = []
        for label in celeb_labels:#movies_labels:#
            processed_labels_q.append(sampled_classes_dict[label])
        processed_labels_q = np.array(processed_labels_q)    
        mu_N = []
        for label in processed_labels_q:
            mu_N.append(mu[label])
        mu_N = np.array(mu_N)
        print('!!')

        #A, As, crit = iterative_A(celeb_feat,mu_N,alpha=theta)
        #enroll_embs = enroll_embs @ A.astype(np.float32).T
        #test_embs = test_embs @ A.astype(np.float32).T
        #enroll_embs = enroll_embs @ A.T.astype(np.float32)
        # Coral adaptation
        A = coral(movies_feat,celeb_feat,theta)
        test_embs = test_embs @ A.astype(np.float32).T
        #A = coral(celeb_feat,movies_feat,theta)
        #enroll_embs = enroll_embs @ A.astype(np.float32).T
              
    for start in tqdm(range(0,n_tasks,batch_size)):
        end = (start+batch_size) if (start+batch_size) <= n_tasks else n_tasks
        
        x_q,y_q,x_s,y_s = (test_embs[start:end],
                        test_labels[start:end],
                        enroll_embs[start:end],
                        enroll_labels[start:end])
        #print(x_q.shape)
        #print(y_q.shape)
        #print(x_s.shape)
        #print(y_s.shape)
        if n_ways_eff == 1:
            eval = Simpleshot(avg="mean",backend="L2",method="transductive_centroid")
            acc_list,acc_list_5,pred_labels_5 = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 
        else:
            eval = Simpleshot(avg="mean",backend="L2",method="inductive")
            acc_list = eval.eval(x_s, y_s, x_q, y_q, test_audios[start:end]) 

        acc["simpleshot"].extend(acc_list)
        acc["simpleshot_5"].extend(acc_list_5)

        if n_ways_eff == 1:
            args['maj_vote'] = True
        else:
            args['maj_vote'] = False

        args['alpha'] = alpha
        method_info = {'device':'cpu','args':args}#'log_file':log_file,'args':args}
        acc_list_paddle,preds_q = run_paddle_new(x_s, y_s, x_q, y_q,method_info,'paddle')
        acc["paddle"].extend(acc_list_paddle)

    count_paddle = 0
    count_simple = 0
    for i in range(len(acc['simpleshot'])):
        if acc['paddle'][i] < acc['simpleshot'][i]:
            count_simple += 1
        if acc['paddle'][i] > acc['simpleshot'][i]:
            count_paddle += 1

    
    new_x_s = []
    new_y_s = []
    new_x_q = []
    new_y_q = []
    stage2_acc_list = []
    for i in range(len(acc_list)):
        task_x_s = []
        task_y_s = []
        if True:#acc_list[i] < acc_list_5[i]:
            top_classes = pred_labels_5[i][0].tolist()
            #print(top_classes)
            #tsne_query_support(x_q[i],x_s[i],y_q[i],y_s[i],top_classes)
            
            classes_dict = {str(label):i for i,label in enumerate(top_classes)}

            #print(f"Task: {i}")
            
            for label in sorted(top_classes):
                top_indices = np.where(y_s[i] == label)
               
                task_x_s.extend(x_s[i][top_indices[0]])
                task_y_s.extend(np.array([classes_dict[str(label)] for label in y_s[i][top_indices[0]]]))

                #cos_sim = np.dot(x_s[i][top_indices[0]][2],x_s[i][top_indices[0]][1])/(norm(x_s[i][top_indices[0]][0])*norm(x_s[i][top_indices[0]][1]))
                #print(cos_sim)
                 
            task_x_s = np.array(task_x_s)
            task_y_s = np.array(task_y_s)

            
            new_x_s.append(task_x_s)
            new_y_s.append(task_y_s)
            new_x_q.append(x_q[i])
            new_y_q.append(np.array(y_q[i]))

        else:
            stage2_acc_list.append(acc_list[i])
    
    new_x_s = np.array(new_x_s)
    new_y_s = np.array(new_y_s)
    new_x_q = np.array(new_x_q)
    new_y_q = np.array(new_y_q)

    if n_ways_eff == 1:
        args['maj_vote'] = True
    else:
        args['maj_vote'] = False
    
    args['alpha'] = alpha
    method_info = {'device':'cpu','args':args}
    acc_list, preds_q = run_paddle_new(new_x_s, new_y_s, new_x_q, new_y_q,method_info,'paddle')
    
    preds_q = preds_q[0].tolist()

    original_preds_q = []
    for task in range(len(preds_q)):
        task_preds = []
        for pred in preds_q[task]:
            original_top_5 = pred_labels_5[task][0].tolist()

            task_preds.append(original_top_5[pred])
        original_preds_q.append(task_preds)
    
    #print(original_preds_q)
    #print(y_q)
    acc_list_stage2 = compute_acc(torch.tensor(original_preds_q),torch.tensor(new_y_q))
 
    stage2_acc_list.extend(acc_list_stage2)

    final_json['simpleshot'][str(theta)] = 100*sum(acc["simpleshot"])/len(acc["simpleshot"])
    final_json['simpleshot_5'][str(theta)] = 100*sum(acc["simpleshot_5"])/len(acc["simpleshot_5"])
    final_json['paddle'][str(theta)] = 100*sum(acc["paddle"])/len(acc["paddle"])
    final_json['2stage_paddle'][str(theta)] =100*sum(stage2_acc_list)/len(stage2_acc_list)

    #print(len(acc['simpleshot']))
    #print(len(acc['simpleshot_5']))
    #print(len(acc['paddle']))
    #print(len(stage2_acc_list))

with open(out_file,'w') as f:
    json.dump(final_json,f)
