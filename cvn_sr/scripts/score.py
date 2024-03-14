import sys
import os
from collections import Counter

def majority_element(input_list):
    counts = Counter(input_list)
    majority_item = counts.most_common(1)[0][0]
    return majority_item

def calculate_accuracy(file_path):
    correct_predictions = 0
    total_predictions = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            data = eval(line)  # Safely evaluate the string as a Python expression
            
            label = data.get('label')
            infer = data.get('infer')
            
            if label and infer:
                total_predictions += 1
                if label == infer:
                    correct_predictions += 1
    
    if total_predictions == 0:
        return 0
    else:
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy

def calculate_window_accuracy(file_path):
    correct_predictions = 0
    total_predictions = 0
    
    ids = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = eval(line)  # Safely evaluate the string as a Python expression
            
            file_id = data.get('file_id') #os.path.basename(data.get('audio_filepath')).split("_window")[0]
            print(file_id)
            if file_id not in ids.keys():
                ids[file_id] = {}
                ids[file_id]['label'] = data.get('label')
                ids[file_id]['infer'] = []

            infer = data.get('infer')
            
            ids[file_id]['infer'].append(infer)

    for file_id in ids.keys():
        pred = majority_element(ids[file_id]['infer'])
        label =  ids[file_id]['label']
        if label and infer:
            total_predictions +=1
            if label == infer:
                correct_predictions +=1
    
    if total_predictions == 0:
        return 0
    else:
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy

file_path = sys.argv[1]  # Replace with the path to your file
mode = sys.argv[2]

if mode == "normal":
    print("entered normal")
    accuracy = calculate_accuracy(file_path)
else:
    accuracy = calculate_window_accuracy(file_path)
print("Accuracy:", accuracy, "%")
