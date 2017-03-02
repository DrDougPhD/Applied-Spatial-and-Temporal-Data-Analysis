import os
import logging
import shutil

import pickle

import config

logger = logging.getLogger('cnn.' + __name__)


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
            pickle_filename = 'pickle.{fn}.{values}.bin'.format(
                fn=func.__name__,
                values='_'.join([
                    '{k},{v}'.format(k=k, v=kwargs[k])
                    for k in self.keywords]))
            pickle_path = os.path.join(config.PICKLE_STORAGE, pickle_filename)

            if config.PICKLING_ENABLED:
                try:
                    with open(pickle_path, 'rb') as pkl:
                        result = pickle.load(pkl)
                    logger.debug('Pickle loaded from {}'.format(pickle_path))
                    return result

                except:
                    logger.warning('No pickle for {0} at "{1}".'
                                   ' It will be created after execution.'.format(
                        func.__name__, pickle_path))

            result = func(*args, **kwargs)

            if config.PICKLING_ENABLED:
                logger.debug('Pickling result to {}'.format(pickle_path))
                with open(pickle_path, 'wb') as pkl:
                    pickle.dump(result, pkl)

            return result

        return func_wrapper

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
