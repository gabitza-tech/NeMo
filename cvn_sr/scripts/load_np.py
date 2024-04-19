import numpy as np
import sys

file = sys.argv[1]

enroll_dict = np.load(file, allow_pickle=True)

feat = enroll_dict['concat_features']

labels = np.expand_dims(np.asarray(enroll_dict['concat_labels']),axis=1)

slices = np.expand_dims(np.asarray(enroll_dict['concat_slices']),axis=1)

patchs = np.expand_dims(np.asarray(enroll_dict['concat_patchs']),axis=1)

print(feat.shape)
print(labels.shape)
print(slices.shape)
print(patchs.shape)

#print(feat[10])
#for i in range(100):
#    print([labels[i],slices[i],patchs[i]])

seen = []
dups = 0 
for (i,label) in enumerate(labels):
    if label[0] == 'id10097':
        print(patchs[i][0])

