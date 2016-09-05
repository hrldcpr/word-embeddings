import heapq
import itertools

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

vectors = {}
with open('glove.6B/glove.6B.50d.txt') as f:
    for line in tqdm(f, total=400000):
        word, *vector = line.split()
        vectors[word] = np.array([float(x) for x in vector])
        assert len(vectors[word]) == 50

def get_nearest(v, N=10):
    nearest = []
    for word, u in tqdm(vectors.items()):
        nearness = -cosine(v, u)
        if len(nearest) < N:
            heapq.heappush(nearest, (nearness, word))
        else:
            heapq.heappushpop(nearest, (nearness, word))
    return sorted(nearest, reverse=True)

def print_nearest(phrase, v=None):
    if v is None: v = sum(vectors[w] for w in phrase.split())
    print(phrase, np.linalg.norm(v))
    for nearness, word in get_nearest(v):
        print(word, nearness)

print_nearest('bank')
print_nearest('river bank')
print_nearest('queen - woman + man', vectors['queen'] - vectors['woman'] + vectors['man'])
print_nearest('kitten - cat + dog', vectors['kitten'] - vectors['cat'] + vectors['dog'])
