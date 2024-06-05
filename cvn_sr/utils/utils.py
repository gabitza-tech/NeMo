import os
import numpy as np
import pickle
import torch

def majority_or_original(tensor):
    majority_labels = []
    for task in tensor:
        values, counts = task.unique(return_counts=True)
        max_count = counts.max().item()
        modes = values[counts == max_count]
        
        # If there's a tie (multiple modes), keep the original values for this task
        if len(modes) > 1:
            majority_labels.append(task)
        else:
            majority_labels.append(modes.repeat(len(task)))
    
    return torch.stack(majority_labels)

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
    test_labels = np.asarray(test_dict['concat_labels'])[test_indices]
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
    all_embs = np.asarray(test_dict['concat_features'])[test_label_indices]

    combined_array = np.column_stack((all_labels, all_slices))
    unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)
    grouped_indices = [np.where(inverse_indices == i)[0] for i in range(len(unique_pairs))]

    test_embs = np.asarray([all_embs[indices] for indices in grouped_indices])

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
    all_slices = np.asarray(enroll_dict['concat_slices'])

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
    all_patchs = np.asarray(enroll_dict['concat_patchs'])[enroll_label_indices]
    all_embs = np.asarray(enroll_dict['concat_features'])[enroll_label_indices]

    combined_array = np.column_stack((all_labels, all_slices))
    unique_pairs, inverse_indices = np.unique(combined_array, axis=0, return_inverse=True)
    
    random_pairs = [(label, np.random.choice(unique_pairs[unique_pairs[:, 0] == label, 1], size=k_shot, replace=False)) for label in sorted(sampled_classes)]
    random_pairs_array = np.concatenate([[[label, id_] for id_ in ids] for label, ids in random_pairs])

    enroll_indices = np.array(find_matching_positions(combined_array, random_pairs_array))

    enroll_embs = all_embs[enroll_indices]
    
    label_dict = {label:i for i,label in enumerate(sampled_classes)}
    enroll_labels = all_labels[enroll_indices]
    enroll_labels = np.asarray([label_dict[label] for label in enroll_labels])

    return enroll_embs, enroll_labels#, enroll_slices,enroll_patchs

def data_SQ_from_pkl(filepath):
    data_dict = load_pickle(filepath)
        
    test_embs = data_dict['test_embs']
    test_labels = data_dict['test_labels']
    test_audios = data_dict['test_audios']
    enroll_embs = data_dict['enroll_embs']
    enroll_labels = data_dict['enroll_labels']
    enroll_audios = data_dict['enroll_audios']

    return test_embs,test_labels,test_audios,enroll_embs,enroll_labels,enroll_audios

def find_matching_positions(list1, list2):
    set_list2 = set(map(tuple, list2))
    matching_positions = [i for i, vector in enumerate(list1) if tuple(vector) in set_list2]
    return matching_positions

def analyze_data(data):
    unique_labels, counts = np.unique(data, return_counts=True)

    # Calculate additional information
    num_unique_labels = len(unique_labels)
    min_appearances = np.min(counts)
    max_appearances = np.max(counts)
    average_appearances = np.mean(counts)

    # Print the results (optional)
    print(f"Number of unique labels: {num_unique_labels}")
    print(f"Minimum appearances of a label: {min_appearances}")
    print(f"Maximum appearances of a label: {max_appearances}")
    print(f"Average appearances of a label: {average_appearances}")

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

def CL2N_embeddings(enroll_embs,test_embs,normalize):
    if not normalize:
        return enroll_embs, test_embs
    
    all_embs = np.concatenate((enroll_embs,test_embs),axis=1)
    all_embs = embedding_normalize(all_embs)
    enroll_embs = all_embs[:,:enroll_embs.shape[1]]
    test_embs = all_embs[:,enroll_embs.shape[1]:]