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