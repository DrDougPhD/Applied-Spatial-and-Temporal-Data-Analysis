import os
import logging
import shutil
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
