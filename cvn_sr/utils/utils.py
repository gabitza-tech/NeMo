import os
import numpy as np
import pickle
from collections import Counter

def majority_element(input_list):
    counts = Counter(input_list)
    majority_item = counts.most_common(1)[0][0]
    return majority_item

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def sampler_query(test_dict,sampled_classes):
    # We find which indices in the lists are part of the sampled classes
    test_indices = [index for index, element in enumerate(test_dict['concat_labels']) if element in sampled_classes]
            
    """
    We first construct the test/query, as it is easier and we always use all of it, one file at a time
    We will create a tensor of size [n_query,192], where n_query are all queries that are part of the sampled classes.
    """
    print("Creating test embeddings vector and the reference labels")
    test_embs = test_dict['concat_features'][test_indices] 
    test_labels = np.asarray([test_dict['concat_labels'][index] for index in test_indices]) 

    return test_embs, test_labels

def sampler_support(enroll_dict, sampled_classes,k_shot):
    enroll_indices = [index for index, element in enumerate(enroll_dict['concat_labels']) if element in sampled_classes]
    """
    We sample the embeddings from k_shot audios in each sampled class
    If audios are normal, it wil sample exactly k_shot audios per class
    If audios are split, it will sample a variable number of window_audios, but still coming from k_shot audios
    We don't oversample embs in a class to equalize the classes lengths at the moment, as we calculate the mean anyway
    """
    
    all_enroll_labels = np.array(enroll_dict['concat_labels'])
    value_masks = [all_enroll_labels == class_name for class_name in sorted(sampled_classes)]
    # Extract indices for each value
    value_indices = [np.where(mask)[0] for mask in value_masks]

    # We do this in order to not take more samples from a class than the existing maximum, or simply not duplicate samples in classes when extracting
    
    if k_shot > 10:
        k_shot_list = []
        for indices in value_indices:
            if k_shot > len(indices):
                k_shot_list.append(len(indices))
            else:
                k_shot_list.append(k_shot)

        # Shuffle the indices for each value
        enroll_indices = np.concatenate([np.random.choice(indices, size=k_shot_list[index], replace=False) for index,indices in enumerate(value_indices)])
    else:
        enroll_indices = np.concatenate([np.random.choice(indices, size=k_shot, replace=False) for indices in value_indices])
    
    enroll_embs = enroll_dict['concat_features'][enroll_indices]
    enroll_labels = all_enroll_labels[enroll_indices]

    return enroll_embs, enroll_labels


def compute_acc(pred_labels,test_labels,sampled_classes):
    total_preds = 0
    correct_preds = 0

    class_acc = {}
    for cls in sampled_classes:
        class_acc[cls] = {}
        class_acc[cls]['total_preds'] = 0
        class_acc[cls]['correct_preds'] = 0
        class_acc[cls]['preds'] = []

    # label in matched_labels is the position of a class in sampled_classes, from argmax
    # matched_labels and test_labels have the same size.
    for (j,label) in enumerate(pred_labels):
        total_preds += 1
        class_acc[test_labels[j]]['total_preds'] +=1
        class_acc[test_labels[j]]['preds'].append(label)

        if label == test_labels[j]:
            correct_preds += 1
            class_acc[test_labels[j]]['correct_preds'] +=1

    acc = 100*(correct_preds/total_preds)

    return acc