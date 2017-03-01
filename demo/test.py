import numpy as np
from scipy.spatial.distance import cdist as pairwise_distances

if __name__ == '__main__':
    # round 1: two classes intertwined with each other
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 2)
    D = pairwise_distances(X, Y)

    print('Intertwined classes:')
    print(X)
    print(Y)

    n, m = D.shape
    D.shape = (1, m*n)
    separation = np.sum(D)
    print('Separation: {}'.format(separation))
    print('Average Separation: {}'.format(separation / (m*n)))

    print('-'*80)

    # round 2: two classes separated
    Y = Y + 8
    D = pairwise_distances(X, Y)

    print('Separated classes:')
    print(X)
    print(Y)

    n, m = D.shape
    D.shape = (1, m*n)
    separation = np.sum(D)
    print('Separation: {}'.format(separation))
    print('Average Separation: {}'.format(separation / (m*n)))

