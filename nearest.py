import heapq
import itertools

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm


def load_word2vec(normalize=False):
    vectors = {}
    with open('GoogleNews-vectors-negative300.bin', 'rb') as f:
        N, K = map(int, f.readline().split())
        print(N, K)
        for _ in tqdm(range(N)):
            word = []
            while True:
                c = f.read(1)
                if c == b' ': break
                word.append(c)
            v = np.fromstring(f.read(K * 4), dtype=np.float32)

            if normalize: v /= np.linalg.norm(v)
            vectors[b''.join(word).decode()] = v
    return vectors

def load_glo_ve(K=300, normalize=False):
    N = 400000
    vectors = {}
    with open('glove.6B/glove.6B.{}d.txt'.format(K)) as f:
        for line in tqdm(f, total=N):
            word, vector = line.split(maxsplit=1)
            v = np.fromstring(vector, sep=' ')
            assert len(v) == K

            if normalize: v /= np.linalg.norm(v)
            vectors[word] = v
    return vectors

def get_nearest(vectors, v, N=10):
    nearest = []
    for word, u in tqdm(vectors.items()):
        nearness = -cosine(v, u)
        if len(nearest) < N:
            heapq.heappush(nearest, (nearness, word))
        else:
            heapq.heappushpop(nearest, (nearness, word))
    return sorted(nearest, reverse=True)

def print_nearest(vectors, phrase, v=None):
    if v is None: v = sum(vectors[w] for w in phrase.split())
    print(phrase, np.linalg.norm(v))
    for nearness, word in get_nearest(vectors, v):
        print(word, nearness)

def main():
    vs = load_glo_ve()
    print_nearest(vs, 'bank')
    print_nearest(vs, 'river bank')
    print_nearest(vs, 'queen - woman + man', vs['queen'] - vs['woman'] + vs['man'])
    print_nearest(vs, 'kitten - cat + dog', vs['kitten'] - vs['cat'] + vs['dog'])

if __name__ == '__main__':
    main()
