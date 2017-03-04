import pprint

import numpy
from scipy.spatial import distance

def it(vectors, k, distance, initial_centroid_method):
    # 1. initialize centroids
    centroids = globals()[initial_centroid_method+'_centroids']\
                         (vectors=vectors, k=k)
    print('Initial centroids:')
    print(pprint.pformat(list(enumerate(centroids))))

    centroid_indices = numpy.arange(k)


    # 2. Loop until centroids don't change anymore
    old_cluster_indices = numpy.zeros(len(vectors))-1
    iteration = 1
    clusters = None
    while True:
        #   2.a Associate vectors with their closest centroid
        for i, v in enumerate(vectors):
            distances = [
                distance(v, centroids[idx])
                for idx in centroid_indices
            ]

            closest_centroid_idx = min(centroid_indices,
                                       key=lambda idx: distances[idx])
            print('Article #{}'.format(i))
            print('Distances:\n{}'.format(
                pprint.pformat(list(enumerate(distances)))))
            print('Closest: {}'.format(closest_centroid_idx))
            print('.'*40)


        new_cluster_indices = numpy.array([
            min(centroid_indices, key=lambda idx: distance(v, centroids[idx]))
            for v in vectors
        ])
        clusters = [[] for _ in range(k)]
        [clusters[cluster_idx].append(vector_idx)
         for (vector_idx,), cluster_idx
         in numpy.ndenumerate(new_cluster_indices)]

        print('.' * 80)
        print('Iteration #{}'.format(iteration))
        print('Clusters:')
        for cluster_idx, cluster in enumerate(clusters):
            print('{0}: {1}'.format(cluster_idx, cluster))

        if (new_cluster_indices == old_cluster_indices).all():
            return clusters

        old_cluster_indices = new_cluster_indices

        #   2.b Recompute centroid
        centroids = numpy.array([
            numpy.mean(vectors[indices], axis=0)
            for indices in clusters
        ])

        print('Centroids:')
        print(pprint.pformat(list(enumerate(centroids))))

        iteration += 1


def random_centroids(vectors, k):
    return vectors[numpy.random.choice(numpy.arange(vectors.shape[0]), size=k)]


if __name__ == '__main__':
    vectors = numpy.random.rand(100, 5)
    k = 7
    distance = distance.euclidean
    centroid_method = 'random'

    it(vectors, k, distance, centroid_method)
