import os
import logging
import pprint
import shutil

import pickle

import itertools
import numpy
from scipy.stats import pearsonr

import config
from lib.lineheaderpadded import hr

logger = logging.getLogger('cnn.' + __name__)

def setup_logger(name):
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', name + '.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    ch.setFormatter(logging.Formatter(
        "%(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ] %("
        "message)s"
    ))
    # add the handlers to the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_category(filename):
    c = filename.split('-')[0]
    return c.split('_')[1]


def archive_files(to, filelist):
    os.makedirs(to, exist_ok=True)
    logger.debug('Copying files to {}'.format(to))

    [shutil.copy(f.path, to) for f in filelist]
    logger.debug('{} files copied'.format(len(filelist)))

## decorators

class pickled(object):
    def __init__(self, *args):
        self.keywords = args

    def __call__(self, func):
        def func_wrapper(*args, **kwargs):
            if 'disable_pickle' in kwargs and kwargs['disable_pickle']:
                return func(*args, **kwargs)

            kwstring_values = {
                k: s.__name__ if hasattr(s, '__name__') else s
                for k, s in kwargs.items()
            }
            pickle_key = func.__name__

            pickle_filename = 'pickle.{fn}.{values}.bin'.format(
                fn=pickle_key,
                values=''.join([
                    '({k},{v})'.format(k=k, v=kwstring_values[k])
                    for k in self.keywords]))
            pickle_path = os.path.join(config.PICKLE_STORAGE, pickle_filename)

            if config.PICKLING_ENABLED and not config.UPDATE_PICKLES:
                try:
                    with open(pickle_path, 'rb') as pkl:
                        result = pickle.load(pkl)
                    logger.debug('Pickle loaded from {}'.format(pickle_path))
                    return result

                except:
                    logger.warning('No pickle for {0} at "{1}".'
                                   ' It will be created after execution.'.format(
                        pickle_key, pickle_path))

            result = func(*args, **kwargs)

            if config.UPDATE_PICKLES:
                logger.debug('Pickling result to {}'.format(pickle_path))
                with open(pickle_path, 'wb') as pkl:
                    pickle.dump(result, pkl)

            return result

        return func_wrapper


@pickled('distance_metric')
def similarity_matrix(matrix, distance_metric, cart_product_indices):
    distances = distance_matrix(matrix=matrix,
                                distance_func=distance_metric,
                                n=len(matrix),
                                cart_product_indices=cart_product_indices)
    # e.g. euclidean_similarities, cosine_similarities, jaccard_similarities
    similarities = globals()[distance_metric.__name__ + '_similarities']\
                            (distances.get_matrix())
    return numpy.array(similarities)


@pickled('n','distance_func')
def distance_matrix(matrix, distance_func, n, cart_product_indices, **kwargs):
    return MemoizedDistances(matrix=matrix,
                             distance_func=distance_func,
                             cart_product_indices=cart_product_indices)


class MemoizedDistances(object):
    def __init__(self, matrix, cart_product_indices, distance_func):
        try:
            n = len(matrix)
        except:
            n = matrix.shape[0]

        self.distance_fn = distance_func

        indices = numpy.arange(n)
        self.memoized = {i: {} for i in indices}

        self.distance_matrix = numpy.array([
            self._get_distance(matrix[i], matrix[j],
                               i, j)
            for i, j in cart_product_indices
        ])
        self.distance_matrix.shape = (n, n)

    def _get_distance(self, u, v, u_idx, v_idx):
        # logger.debug('Calculating distance for vectors '
        #              'of shape {0} and {1}...'.format(
        #     u.shape, v.shape
        # ))
        # logger.debug(u)
        # logger.debug(v)
        if u_idx == v_idx:
            return 0

        min_idx = min(u_idx, v_idx)
        max_idx = max(u_idx, v_idx)
        memoized_for_u = self.memoized[min_idx]

        if max_idx not in memoized_for_u:
            memoized_for_u[max_idx] = self.distance_fn(u, v)

        return memoized_for_u[max_idx]

    def get_matrix(self):
        return self.distance_matrix

    def __str__(self):
        return pprint.pformat(self.distance_matrix)


def euclidean_similarities(distances):
    min_distance = numpy.amin(distances)
    max_distance = numpy.amax(distances)
    return -1* ((distances-min_distance)/(max_distance-min_distance)) + 1


def cosine_similarities(distances):
    return (-1*distances) + 1


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


def calculate_sse(centroids, clustering, matrix, distance_func):
    squared_distances = []
    for cluster_centroid, cluster_indices in zip(centroids, clustering):
        cluster = matrix[cluster_indices].toarray()
        sqrd_distances_to_centroid = numpy.apply_along_axis(
            lambda v: distance_func(v, cluster_centroid) ** 2,
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

def silhouette_coeff(clustering, corpus, distance_func):
    for i, cluster_indices in enumerate(list(clustering)):
        cluster = corpus.matrix[cluster_indices]
        pairwise_indices = list(itertools.combinations(
            cluster_indices, 2))
        logger.debug('Pairwise indices: {}'.format(pairwise_indices))

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
            lambda indices: distance_func(corpus.matrix[indices[0]].toarray(),
                                          corpus.matrix[indices[1]].toarray()),
            axis=1,
            arr=pairwise_indices
        )
        logger.debug('Distances between vectors in cluster:')
        logger.debug(distances)
        average_distance = numpy.mean(distances)
        logger.debug('Average distance: {}'.format(average_distance))

        # calculate average distance between this cluster and its closest
        # neighboring cluster
        closest_neighbor_distance = float('inf')
        for j, neighbor_indices in enumerate(list(clustering)):
            if i == j:
                continue

            logger.debug(hr('Distance to Cluster {}'.format(j)))
            pairwise_indices = list(itertools.product(cluster_indices,
                                                      neighbor_indices))
            neighbor_distances = numpy.apply_along_axis(
                lambda indices: distance_func(
                    corpus.matrix[indices[0]].toarray(),
                    corpus.matrix[indices[1]].toarray()),
                axis=1,
                arr=pairwise_indices
            )
            avg_distance_to_neighbors = numpy.mean(neighbor_distances)
            logger.debug(
                'Average distance: {}'.format(avg_distance_to_neighbors))
            closest_neighbor_distance = min(closest_neighbor_distance,
                                            avg_distance_to_neighbors)

        logger.debug('Closest neighboring cluster (avg distance): {}'.format(
            closest_neighbor_distance
        ))

        silhouette = (closest_neighbor_distance - average_distance) \
                   / max(closest_neighbor_distance, average_distance)

        return silhouette