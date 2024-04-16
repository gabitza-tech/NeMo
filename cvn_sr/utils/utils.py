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

    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    test_labels = np.asarray([test_dict['concat_labels'][index] for index in test_indices])
    test_labels = np.asarray([label_dict[label] for label in test_labels])

    return test_embs, test_labels

def sampler_windows_query(test_dict,sampled_classes):
    #test_dict['concat_labels'] = np.repeat(test_dict['concat_labels'],10,0)
    #test_dict['concat_features'] = np.repeat(test_dict['concat_features'],10,0)
    #test_dict['concat_slices'] = np.repeat(test_dict['concat_slices'],10,0)
    #print(test_dict['concat_labels'].shape)
    #print(test_dict['concat_features'].shape)
    #print(test_dict['concat_slices'].shape)

    # We find which indices in the lists are part of the sampled classes
    test_label_indices = [index for index, element in enumerate(test_dict['concat_labels']) if element in sampled_classes]

    all_labels = np.asarray(test_dict['concat_labels'])[test_label_indices]
    all_slices = np.asarray(test_dict['concat_slices'])[test_label_indices]

    combined_array = np.column_stack((all_labels, all_slices))
    unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)
    grouped_indices = [np.where(inverse_indices == i)[0] for i in range(len(unique_pairs))]

    test_embs = np.asarray([test_dict['concat_features'][indices] for indices in grouped_indices])

    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    test_labels = np.asarray([all_labels[indices[0]] for indices in grouped_indices])
    test_labels = np.asarray([label_dict[label] for label in test_labels])

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
    
    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    enroll_labels = all_enroll_labels[enroll_indices]
    enroll_labels = np.asarray([label_dict[label] for label in enroll_labels])

    return enroll_embs, enroll_labels


def sampler_windows_support(enroll_dict, sampled_classes,k_shot):
    # We find which indices in the lists are part of the sampled classes

    #enroll_dict['concat_labels'] = np.repeat(enroll_dict['concat_labels'],10,0)
    #enroll_dict['concat_features'] = np.repeat(enroll_dict['concat_features'],10,0)
    #enroll_dict['concat_slices'] = np.repeat(enroll_dict['concat_slices'],10,0)
    #print(enroll_dict['concat_labels'].shape)
    #print(enroll_dict['concat_features'].shape)
    #print(enroll_dict['concat_slices'].shape)

    enroll_label_indices = [index for index, element in enumerate(enroll_dict['concat_labels']) if element in sampled_classes]

    all_labels = np.asarray(enroll_dict['concat_labels'])[enroll_label_indices]
    all_slices = np.asarray(enroll_dict['concat_slices'])[enroll_label_indices]

    combined_array = np.column_stack((all_labels, all_slices))
    unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)
    
    random_pairs = [(label, np.random.choice(unique_pairs[unique_pairs[:, 0] == label, 1], size=k_shot, replace=False)) for label in sorted(sampled_classes)]
    random_pairs_array = np.concatenate([[[label, id_] for id_ in ids] for label, ids in random_pairs])

    mask = np.isin(combined_array, random_pairs_array)
    enroll_indices = np.where(mask.all(axis=1))[0]

    enroll_embs = enroll_dict['concat_features'][enroll_indices]
    
    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    enroll_labels = np.asarray(enroll_dict['concat_labels'])[enroll_indices]
    enroll_labels = np.asarray([label_dict[label] for label in enroll_labels])

    return enroll_embs, enroll_labels

def embedding_normalize(embs, use_std=False, eps=1e-10):
    """
    Mean and l2 length normalize the input speaker embeddings

    Args:
        embs: embeddings of shape (Batch,emb_size)
    Returns:
        embs: normalized embeddings of shape (Batch,emb_size)
    """
    initial_shape = embs.copy().shape
    if len(initial_shape) == 3:
        embs =embs.reshape((-1, embs.shape[-1]))

    embs = embs - embs.mean(axis=0)
    if use_std:
        embs = embs / (embs.std(axis=0) + eps)
    embs_l2_norm = np.expand_dims(np.linalg.norm(embs, ord=2, axis=-1), axis=1)
    embs = embs / embs_l2_norm

    if len(initial_shape) == 3:
        embs = embs.reshape(initial_shape)

    return embs

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