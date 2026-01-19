import os
from collections import Counter
import json
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
import nltk
nltk.download('punkt')

def create_vocab(captions_path, threshold=5):
    with open(captions_path, 'r') as f:
        captions = f.read().split('\n')
    counter = Counter()
    for sent in captions:
        tokens = nltk.tokenize.word_tokenize(sent.lower())
        counter.update(tokens)
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    for i, word in enumerate(words):
        vocab[word] = i + 4
    return vocab

# Assume Flickr8k.anns.txt and Flickr8k.token.txt downloaded
captions_path = 'data/Flickr8k.token.txt'
word_map = create_vocab(captions_path)
with open('data/wordmap.json', 'w') as f:
    json.dump(word_map, f)

# Extract features (use precomputed or run encoder on images)
h = h5py.File('data/caption_train.h5', 'w')
# Loop over train images, save encoder.features(Image.open(img))
# Similar for val/test, captions as lists
