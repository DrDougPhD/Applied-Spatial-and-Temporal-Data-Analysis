import logging

import itertools
import numpy
from matplotlib import pyplot
from scipy.spatial import distance

import kmeans

logger = logging.getLogger('cnn.' + __name__)

import utils

def plot(sorted_matrix, distance_metric, cart_prod_indices):
    # Create an array of distance values between each pair
    similarities = utils.similarity_matrix(
        matrix=sorted_matrix,
        distance_metric=distance_metric,
        cart_product_indices=cart_prod_indices
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
    logger = utils.setup_logger('cnn'+__name__)
    at_origin = numpy.random.rand(30, 2)
    positive_cluster = numpy.random.rand(30, 2)+0
    negative_cluster = numpy.random.rand(30, 2)-0
    class_indices = numpy.concatenate([
        [0 for _ in range(30)],
        [1 for _ in range(30)],
        [2 for _ in range(30)]])
    logger.debug('Class indices: {}'.format(class_indices))

    random_vectors = numpy.concatenate([at_origin,
                                        positive_cluster,
                                        negative_cluster],
                                       axis=0)

    print('Random clusters concatd together')
    print(random_vectors.shape)

    print('Clustering...')
    distance_func = distance.euclidean
    clustering, centroids = kmeans.kmeans(vectors=random_vectors,
                                          k=3,
                                          distance=distance_func,
                                          initial_centroid_method='random')

    sorted_vectors = None
    for cluster_indices in clustering:
        cluster_vectors = random_vectors[cluster_indices, :]
        pyplot.scatter(cluster_vectors[:, 0],
                       cluster_vectors[:, 1])

        if sorted_vectors is None:
            sorted_vectors = cluster_vectors
        else:
            sorted_vectors = numpy.concatenate([sorted_vectors,
                                                cluster_vectors],
                                               axis=0)
    pyplot.show()

    indices = numpy.arange(random_vectors.shape[0])
    cart_product_indices = list(itertools.product(indices,
                                             repeat=2))
    logger.debug('Pairwise indices: {}'.format(cart_product_indices))

    plot(sorted_matrix=sorted_vectors,
         distance_metric=distance_func,
         cart_prod_indices=cart_product_indices)

    article_cluster_indices = []
    class_indices_in_sorted = []
    for cluster_index, cluster in enumerate(clustering):
        # flatten cluster
        article_cluster_indices.extend([cluster_index
                                        for _ in range(len(cluster))])
        class_indices_in_sorted.extend([
            class_indices[i]
            for i in cluster
        ])

    logger.debug('Cluster-based indices: {}'.format(article_cluster_indices))
    logger.debug('Class-based indices: {}'.format(class_indices_in_sorted))


    ## Ideal Cluster to Ideal Class Similarity Matrix correlation
    correlation = utils.ideal_correlation(
        cluster_indices=article_cluster_indices,
        class_indices=class_indices_in_sorted,
        n=random_vectors.shape[0])
    logger.info('Ideal Correlation: {}'.format(correlation))

    ## SSE
    sse = utils.calculate_sse(centroids, clustering, random_vectors)
    logger.debug('SSE: {}'.format(sse))

    ## Silhouette coefficient
    silhouette = utils.silhouette_coeff(clustering, random_vectors, distance_func)
    logger.debug('Silhouette Coefficient: {}'.format(silhouette))
