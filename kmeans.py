import pprint

import numpy
from scipy.spatial import distance
import collections

def it(vectors, k, distance, initial_centroid_method):
    # num_vectors = len(vectors)
    # vector_indices = numpy.arange(num_vectors)

    # 1. initialize centroids
    centroids = globals()[initial_centroid_method](vectors=vectors,
                                                   k=k)
    print('Initial centroids:')
    print(pprint.pformat(list(enumerate(centroids))))

    centroid_indices = numpy.arange(k)

    # Create a pairing of each vector to each centroid so that distance
    # calculations can be (naively) faster.
    # vector_to_centroid_pairs = numpy.transpose([
    #     numpy.tile(vector_indices, k),
    #     numpy.repeat(centroid_indices, num_vectors)
    # ])

    # 2. Loop until centroids don't change anymore
    old_cluster_indices = numpy.zeros(len(vectors))-1
    iteration = 1
    while True:
        #   2.a Associate vectors with their closest centroid
        new_cluster_indices = numpy.array([
            min(centroid_indices, key=lambda idx: distance(v, centroids[idx]))
            for v in vectors
        ])
        if (new_cluster_indices == old_cluster_indices).all():
            break

        old_cluster_indices = new_cluster_indices

        #   2.b Recompute centroid
        clusters = [[] for _ in range(k)]
        [clusters[cluster_idx].append(vector_idx)
         for (vector_idx,), cluster_idx
         in numpy.ndenumerate(new_cluster_indices)]
        centroids = numpy.array([
            numpy.mean(vectors[indices], axis=0)
            for indices in clusters
        ])

        print('.'*80)
        print('Iteration #{}'.format(iteration))
        print('Clusters:')
        for cluster_idx, cluster in enumerate(clusters):
            print('{0}: {1}'.format(cluster_idx, cluster))

        print('Centroids:')
        print(pprint.pformat(list(enumerate(centroids))))

        iteration += 1


def random_centroids(vectors, k):
    return vectors[numpy.random.choice(numpy.arange(len(vectors)), size=k)]


if __name__ == '__main__':
    vectors = numpy.random.rand(100, 5)
    k = 7
    distance = distance.euclidean
    centroid_method = 'random_centroids'

    it(vectors, k, distance, centroid_method)
