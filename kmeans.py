import pprint

import numpy
from matplotlib import pyplot
from scipy.spatial import distance
import utils

@utils.pickled('k', 'distance', 'initial_centroid_method')
def it(vectors, k, distance, initial_centroid_method, verbose=False):
    # 1. initialize centroids
    centroids = globals()[initial_centroid_method+'_centroids']\
                         (vectors=vectors, k=k)

    if verbose:
        print('Initial centroids:')
        print(pprint.pformat(list(enumerate(centroids))))

    centroid_indices = numpy.arange(k)


    # 2. Loop until centroids don't change anymore
    old_cluster_indices = numpy.zeros(len(vectors))-1
    iteration = 1
    clustering = None
    while True:
        #   2.a Associate vectors with their closest centroid
        if verbose:
            for i, v in enumerate(vectors):
                distances = [
                    distance(v, centroids[idx])
                    for idx in centroid_indices
                ]

                closest_centroid_idx = min(centroid_indices,
                                           key=lambda idx: distances[idx])
                print('Article #{0} -- {1}'.format(i, v))
                print('Distances:\n{}'.format(
                    pprint.pformat(list(enumerate(distances)))))
                print('Closest: {}'.format(closest_centroid_idx))
                print('.'*40)


        new_cluster_indices = numpy.array([
            min(centroid_indices, key=lambda idx: distance(v, centroids[idx]))
            for v in vectors
        ])
        clustering = [[] for _ in range(k)]
        [clustering[cluster_idx].append(vector_idx)
         for (vector_idx,), cluster_idx
         in numpy.ndenumerate(new_cluster_indices)]

        if verbose:
            print('.' * 80)
            print('Iteration #{}'.format(iteration))
            print('Clusters:')
            for cluster_idx, cluster in enumerate(clustering):
                print('{0}: {1}'.format(cluster_idx, cluster))

        if (new_cluster_indices == old_cluster_indices).all():
            return clustering, centroids

        old_cluster_indices = new_cluster_indices

        #   2.b Recompute centroid
        centroids = numpy.array([
            numpy.mean(vectors[indices], axis=0)
            for indices in clustering
        ])

        if verbose:
            print('Centroids:')
            print(pprint.pformat(list(enumerate(centroids))))

        iteration += 1


def random_centroids(vectors, k):
    return vectors[numpy.random.choice(numpy.arange(vectors.shape[0]),
                                       size=k,
                                       replace=False)]


if __name__ == '__main__':
    vectors = numpy.random.rand(100, 2)
    k = 7
    distance = distance.euclidean
    centroid_method = 'random'

    clustering, centroids = it(vectors, k, distance, centroid_method,
                               verbose=True)

    markers = '.ov^*+x'
    colors = 'rgbcmyk'
    for cluster_idx, cluster in enumerate(clustering):
        pyplot.scatter(vectors[cluster][:,0],
                       vectors[cluster][:,1],
                       marker='.',
                       #marker=markers[cluster_idx],
                       c=colors[cluster_idx])
        pyplot.scatter(centroids[cluster_idx][0],
                       centroids[cluster_idx][1],
                       marker=markers[cluster_idx],
                       c=colors[cluster_idx])

    pyplot.show()
