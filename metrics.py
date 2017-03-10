import itertools
import numpy
import logging

from scipy.stats import pearsonr

from lib.lineheaderpadded import hr

logger = logging.getLogger('cnn.' + __name__)


def ideal_matrix(n, labels):
    ideal_mtx = numpy.zeros((n, n))
    for i, j in numpy.ndindex((n, n)):
        if labels[i] == labels[j]:
            ideal_mtx[i, j] = 1
    return ideal_mtx


def ideal_correlation(cluster_indices, class_indices, n):
    ideal_cluster_matrix = ideal_matrix(n, cluster_indices)
    ideal_class_matrix = ideal_matrix(n, class_indices)
    ideal_class_matrix.shape = ideal_cluster_matrix.shape = (n*n,)

    correlation, pval = pearsonr(ideal_cluster_matrix, ideal_class_matrix)

    # logger.debug('Ideal cluster matrix')
    # logger.debug(ideal_cluster_matrix)
    # logger.debug('Ideal class matrix')
    # logger.debug(ideal_class_matrix)
    # logger.info('Correlation between Ideal Cluster Similarity Matrix and '
    #             'Ideal Class Similarity Matrix: {}'.format(correlation))

    # correlation between ideal cluster matrix and class matrix
    return correlation


def calculate_sse(centroids, clustering, matrix, distance):
    logger.debug('Distances calculated through {}'.format(distance.__name__))
    squared_distances = []
    for cluster_centroid, cluster_indices in zip(centroids, clustering):
        try:
            cluster = matrix[cluster_indices].toarray()
        except:
            cluster = matrix[cluster_indices]

        sqrd_distances_to_centroid = numpy.apply_along_axis(
            lambda v: distance(v, cluster_centroid) ** 2,
            arr=cluster,
            axis=1,
        )

        # logger.debug('Cluster centroid (shape: {0}): {1}'.format(
        #     cluster_centroid.shape,
        #     cluster_centroid
        # ))
        # logger.debug('Cluster (shape: {0}): {1}'.format(cluster.shape,
        #                                                 cluster))
        # logger.debug('Distances to centroid: {}'.format(
        #     sqrd_distances_to_centroid))

        squared_distances.append(sqrd_distances_to_centroid)
    squared_distances_flattened = numpy.concatenate(squared_distances)
    sse = numpy.sum(squared_distances_flattened)

    # logger.debug('Flattened squared distances: {}'.format(
    #     squared_distances_flattened))
    # logger.debug('SSE: {}'.format(sse))

    return sse


def silhouette_coeff(clustering, matrix, distance):
    logger.debug('{} clusters to consider'.format(len(clustering)))
    logger.debug('Distances calculated through {}'.format(distance.__name__))

    silhouette_coeffs = []
    for i, cluster_indices in enumerate(list(clustering)):
        # logger.debug(hr('Cluster {0}'.format(i), '~'))
        # logger.debug('\n{}'.format(matrix[cluster_indices]))
        # logger.debug(hr('Intracluster distances:', '-'))

        cluster = matrix[cluster_indices]

        # pairwise_indices = list(itertools.combinations(
        #     cluster_indices, 2))
        # pairwise_indices = numpy.transpose([
        #     numpy.tile(cluster_indices, len(cluster_indices)),
        #     numpy.repeat(cluster_indices, len(cluster_indices))
        # ])
        pairwise_indices = []
        for j in list(cluster_indices):
            for k in list(cluster_indices):
                if j != k:
                    pairwise_indices.append((j,k))

        # logger.debug('Pairwise indices: {}'.format(pairwise_indices))

        # distances = []
        # for i,j in pairwise_indices:
        #     u = corpus.matrix[i].toarray()
        #     v = corpus.matrix[j].toarray()
        #
        #     logger.debug('u ({0.category}) := {0.title}'.format(
        #         corpus.corpus[i]))
        #     logger.debug('v ({0.category}) := {0.title}'.format(
        #         corpus.corpus[j]))
        #
        #     d = distance_func(u, v)
        #     logger.debug('distance := {}'.format(d))
        #     logger.debug('.'*50)
        #     distances.append(d)

        # logger.debug(hr('Better distance calculation', '.'))
        distances = numpy.apply_along_axis(
            lambda indices: distance(matrix[indices[0]],
                                     matrix[indices[1]]),
            axis=1,
            arr=pairwise_indices
        )
        # for indices, d in zip(pairwise_indices, distances):
        #     logger.debug('{2:.5f} -- ({0}, {1})'.format(matrix[indices[0]],
        #                                                 matrix[indices[1]],
        #                                                 d))
        # logger.debug('Distances between vectors in cluster:')
        # logger.debug(distances)
        average_distance = numpy.mean(distances)
        # logger.debug('Average distance: {}'.format(average_distance))

        # calculate average distance between this cluster and its closest
        # neighboring cluster
        closest_neighbor_distance = float('inf')
        for j, neighbor_indices in enumerate(list(clustering)):
            # logger.debug(hr('Comparing against C{}'.format(j), '-'))
            if i == j:
                # logger.debug('No self checks needed')
                continue

            # logger.debug(hr('Distance between C{0} and C{1}'.format(i, j), '.'))
            pairwise_indices = list(itertools.product(cluster_indices,
                                                      neighbor_indices))
            neighbor_distances = numpy.apply_along_axis(
                lambda indices: distance(
                    matrix[indices[0]],
                    matrix[indices[1]]),
                axis=1,
                arr=pairwise_indices
            )
            # for (x,y), d in zip(pairwise_indices, neighbor_distances):
            #     logger.debug('{0:.5f} -- ({1}, {2})'.format(d, matrix[x], matrix[y]))
            avg_distance_to_neighbors = numpy.mean(neighbor_distances)
            # logger.debug(
            #     'Average distance: {}'.format(avg_distance_to_neighbors))
            closest_neighbor_distance = min(closest_neighbor_distance,
                                            avg_distance_to_neighbors)

        # logger.debug('Closest neighboring cluster (avg distance): {}'.format(
        #     closest_neighbor_distance
        # ))

        silhouette = (closest_neighbor_distance - average_distance) \
                   / max(closest_neighbor_distance, average_distance)

        silhouette_coeffs.append(silhouette)

        logger.debug('Silhouette Coefficient for Cluster {0}: {1}'.format(
            i, silhouette
        ))

    return numpy.mean(silhouette_coeffs)

