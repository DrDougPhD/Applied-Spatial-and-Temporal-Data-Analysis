import os
import logging
import pprint
import shutil

import pickle

import itertools
import numpy

import config

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
def similarity_matrix(matrix, distance_metric):
    distances = distance_matrix(matrix=matrix,
                                distance_func=distance_metric,
                                n=len(matrix))
    # e.g. euclidean_similarities, cosine_similarities, jaccard_similarities
    similarities = globals()[distance_metric.__name__ + '_similarities']\
                            (distances.get_matrix())
    return numpy.array(similarities)


@pickled('n','distance_func')
def distance_matrix(matrix, distance_func, n):
    return MemoizedDistances(matrix=matrix,
                             distance_func=distance_func)


class MemoizedDistances(object):
    def __init__(self, matrix, distance_func):
        n = len(matrix)
        self.distance_fn = distance_func

        indices = numpy.arange(n)
        self.memoized = {i: {} for i in indices}

        cart_product_indices = itertools.product(indices,
                                                 repeat=2)
        self.distance_matrix = numpy.array([
            self._get_distance(matrix[i], matrix[j],
                               i, j)
            for i, j in cart_product_indices
        ])
        self.distance_matrix.shape = (n, n)

    def _get_distance(self, u, v, u_idx, v_idx):
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


# class ArchiveReturnedFiles(object):
#     def __init__(self, archive_directory):
#         os.makedirs(archive_directory, exist_ok=True)
#         self.directory = archive_directory
#
#     def __call__(self, func):
#         def func_wrapper(*args, **kwargs):
#             files = func(*args, **kwargs)
#
#             logger.debug('Copying files to {}'.format(self.directory))
#
#             [shutil.copy(f.path, self.directory) for f in files]
#             logger.debug('{} files copied'.format(len(files)))
#
#             return files
#
#         return func_wrapper
