import logging

import numpy
from matplotlib import pyplot
from scipy.spatial import distance

import kmeans

logger = logging.getLogger('cnn.' + __name__)

import utils

def plot(sorted_matrix, distance_metric):
    # Create an array of distance values between each pair
    similarities = utils.similarity_matrix(
        matrix=sorted_matrix,
        distance_metric=distance_metric,
    )
    logger.debug('\n{}'.format(similarities))

    x_vals = []
    y_vals = []
    colors = []
    for (x, y), color in numpy.ndenumerate(similarities):
        x_vals.append(x)
        y_vals.append(y)
        colors.append(color)

    logger.debug('X: {0} -- {1}'.format(len(x_vals), x_vals))
    logger.debug('Y: {0} -- {1}'.format(len(y_vals), y_vals))
    logger.debug('C: {0} -- {1}'.format(len(colors), colors))

    pyplot.pcolor(similarities)
    #pyplot.xticks(numpy.arange(sorted_matrix.shape[0]),
    # article_cluster_indices)
    pyplot.colorbar()

    pyplot.show()


if __name__ == '__main__':
    # Create 3 clustered regions
    at_origin = numpy.random.rand(30, 2)
    positive_cluster = numpy.random.rand(30, 2)+3
    negative_cluster = numpy.random.rand(30, 2)-5

    random_vectors = numpy.concatenate([at_origin,
                                        positive_cluster,
                                        negative_cluster],
                                       axis=0)

    print('Random clusters concatd together')
    print(random_vectors.shape)

    print('Clustering...')
    clustering, centroids = kmeans.it(vectors=random_vectors,
                                      k=3,
                                      distance=distance.euclidean,
                                      initial_centroid_method='random')

    sorted_vectors = None
    for cluster_indices in clustering:
        cluster_vectors = random_vectors[cluster_indices, :]
        # pyplot.scatter(cluster_vectors[:, 0],
        #                cluster_vectors[:, 1])

        if sorted_vectors is None:
            sorted_vectors = cluster_vectors
        else:
            sorted_vectors = numpy.concatenate([sorted_vectors,
                                                cluster_vectors],
                                               axis=0)

    plot(sorted_matrix=sorted_vectors,
         distance_metric=distance.euclidean)

