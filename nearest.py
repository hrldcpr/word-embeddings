import heapq

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

def print_nearest(vectors, phrase):
    K = len(next(iter(vectors.values())))
    v = np.zeros(K)
    for word in phrase.split():
        if word.startswith('-'): v -= vectors[word[1:]]
        else: v += vectors[word]

    print(phrase, np.linalg.norm(v))
    for nearness, word in get_nearest(vectors, v):
        print(word, nearness)

def main():
    vs = load_glo_ve(50)
    print_nearest(vs, 'king')
    print_nearest(vs, 'woman')
    print_nearest(vs, 'king -man woman')

if __name__ == '__main__':
    main()
