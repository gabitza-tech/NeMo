import json
import os
import pickle as pkl
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.stats import norm
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.utils import logging
import sys
import matplotlib.pyplot as plt

speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
speaker_model.eval()
speaker_model.to(device)

audio_dir = sys.argv[1]
out_histo = sys.argv[2]
embeddings_list = []

for (i,file) in enumerate(os.listdir(audio_dir)):
    print(file)
    audio = os.path.join(audio_dir, file)    
    with torch.no_grad():
        emb,logits = speaker_model.infer_file(audio)    
        embeddings_np = emb.cpu().numpy().flatten()
    embeddings_list.append(embeddings_np)
    print(embeddings_np)

emb_np = sum(embeddings_list)/len(embeddings_list)
mean = np.mean(emb_np)
std_dev = np.std(emb_np)# Plot the density distribution

plt.figure(figsize=(10, 6))
plt.hist(emb_np, bins=30, density=True, alpha=0.6, color='b')# Plot the PDF (Probability Density Function) using a normal distribution

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std_dev)

plt.plot(x, p, 'k', linewidth=2)# Annotate mean and standard deviation
plt.text(mean, 0.05, f'Mean: {mean:.2f}', fontsize=12, ha='center')
plt.text(mean + 2 * std_dev, 0.05, f'Standard Deviation: {std_dev:.2f}', fontsize=12, ha='center')
plt.xlabel('Embedding Feature Value')
plt.ylabel('Density')
plt.title('Density Distribution of Embedding Features with Mean and Standard Deviation')
plt.grid(True)
plt.savefig(f'{out_histo}.png')
plt.show()
