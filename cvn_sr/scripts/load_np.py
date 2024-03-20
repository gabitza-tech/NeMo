import numpy as np
import sys

file = sys.argv[1]

class_splits = np.load(file, allow_pickle=True)

train, val, test = class_splits

print(len(test))
print(len(val))
print(len(train))