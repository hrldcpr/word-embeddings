import heapq

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm


def pcosine(u, v):
    return (cosine(u, v) + 1) / 2

# metrics from Levy-Goldberg 2014:

def three_cos_add(a1, a2, b1, b2):  # 3CosAdd
    return cosine(b2 , a2 - a1 + b1)

def pair_direction(a1, a2, b1, b2):  # PairDirection
    return cosine(b2 - b1, a2 - a1)

def three_cos_mul(a1, a2, b1, b2, epsilon=0.001):  # 3CosMul
    return pcosine(b2, b1) * pcosine(b2, a2) / (pcosine(b2, a1) + epsilon)


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

def get_nearest(vectors, distance, N=10):
    nearest = []
    for word, u in tqdm(vectors.items()):
        nearness = -distance(u)
        if len(nearest) < N:
            heapq.heappush(nearest, (nearness, word))
        else:
            heapq.heappushpop(nearest, (nearness, word))
    return sorted(nearest, reverse=True)

def print_nearest(vectors, word):
    v = vectors[word]
    print(word, np.linalg.norm(v))
    for nearness, word in get_nearest(vectors, lambda u: cosine(u, v)):
        print(word, nearness)

def get_analogues(vectors, a1, a2, b1, distance):
    """find b2 such that a1 is to a2 as b1 is to b2"""
    return get_nearest(vectors, lambda b2: distance(a1, a2, b1, b2))

def print_analogues(vectors, a1, a2, b1, distance=three_cos_add):
    print(a1, a2, b1)
    for nearness, word in get_analogues(vectors, vectors[a1], vectors[a2], vectors[b1],
                                        distance):
        print(word, nearness)


def main():
    vectors = load_glo_ve(50, normalize=True)
    print_nearest(vectors, 'king')
    print_nearest(vectors, 'woman')
    print('3CosAdd')
    print_analogues(vectors, 'man', 'king', 'woman', distance=three_cos_add)
    print('PairDirection')
    print_analogues(vectors, 'man', 'king', 'woman', distance=pair_direction)
    print('3CosMul')
    print_analogues(vectors, 'man', 'king', 'woman', distance=three_cos_mul)

if __name__ == '__main__':
    main()
