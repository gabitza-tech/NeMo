import os
import json
import argparse
import pickle
import random
import librosa
import sys

if len(sys.argv) == 4:
    manifest = sys.argv[1]
    out = sys.argv[2]
    no_shots = sys.argv[3]
else:
    manifest = sys.argv[1]
    out = sys.argv[2]
    no_shots = 5


class_dict = {}
out_lines = []
with open(manifest, "r") as f, open(out,"a") as f2:
    lines = f.readlines()
    for i,line in enumerate(lines):
        data = eval(line)

        label = data.get('label')

        if label not in class_dict.keys():
            class_dict[label] = [i]
        else:
            class_dict[label].append(i)
    
    for label in class_dict.keys():
        rows_class = random.sample(class_dict[label], no_shots)
        out_lines.extend(rows_class)
    
    for line in out_lines:
        f2.write(lines[line])
