#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SYNOPSIS

	python SCRIPT.py [-h,--help] [-v,--verbose]


DESCRIPTION

	Concisely describe the purpose this script serves.


ARGUMENTS

	-h, --help		show this help message and exit
	-v, --verbose		verbose output


AUTHOR

	Doug McGeehan


LICENSE

	Copyright 2017 Doug McGeehan - GNU GPLv3

"""

__appname__ = "cnn"
__author__ = "Doug McGeehan"
__version__ = "0.0pre0"
__license__ = "GNU GPLv3"
__dev__ = True # used for debug messages in logs


import argparse
from datetime import datetime
import sys
import os
import logging

logger = logging.getLogger(__appname__)


IMPLEMENTED_ARCHIVE_EXTENSIONS = ['zip', 'tgz']
DATASETS_DOWNLOAD = {
    'CNN.Qian‎.zip':
        'https://docs.google.com/uc?id=0B6PIPLNXk0o1MFp0RzlwVlE0MnM&export=download',
    'CNN.DMQA.tgz':
        'https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ',
}
DATASET_DOWNLOADER_SCRIPT = 'gdown.pl'
DATASET_DOWNLOADER_URL = 'https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl'


def main(args):
    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), dataset_dir)

    archive_files = get_datasets(indir=dataset_dir)
    if not archive_files:
        raise ('Error loading datasets. Please download from the following urls:\n'
               '\thttp://cs.nyu.edu/~kcho/DMQA/\n'
               '\thttps://sites.google.com/site/qianmingjie/home/datasets/cnn'
               '-and-fox-news')




def get_datasets(indir):
    files = [f for f in os.listdir(indir) \
             if os.path.isfile(os.path.join(indir, f)) and \
             is_archive(f)]

    if not files:
        logger.debug('No archive files found within {}'.format(indir))
        files.extend(download_datasets(to=indir))

    logger.debug('Archive files: {}'.format(files))
    return set(files)


def is_archive(filename):
    extension = filename.split('.')[-1]
    if extension.lower() in IMPLEMENTED_ARCHIVE_EXTENSIONS:
        return True
    else:
        return False


def download_datasets(to):
    logger.debug('Download datasets from Internet')
    logger.debug('\n'.join(DATASETS_DOWNLOAD.values()))

    if DATASET_DOWNLOADER_SCRIPT not in os.listdir(to):
        logger.debug("Dataset downloader '{}' not found. Retrieving "
                     "from {}".format(
            DATASET_DOWNLOADER_SCRIPT,
            DATASET_DOWNLOADER_URL))
        import urllib.request
        downloader_opener = urllib.request.URLopener()
        download_to = os.path.join(to, DATASET_DOWNLOADER_SCRIPT)
        downloader_opener.retrieve(DATASET_DOWNLOADER_URL, download_to)

    for filename in DATASETS_DOWNLOAD:
        url = DATASETS_DOWNLOAD[filename]



def setup_logger(args):
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', __appname__ + '.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    if args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    ch.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Description printed to command-line if -h is called."
    )
    # during development, I set default to False so I don't have to keep
    # calling this with -v
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=__dev__, help='verbose output')

    """
    def path(*args):
        logger.debug('Path arguments: {}'.format(args))
        abspath = os.path.dirname(os.path.abspath(__file__))
        if len(args) == 0: # assume it was called for cwd
            pass # leave as cwd

        else:
            initial_path = os.path.join(*args)
            if not os.path.isabs(initial_path):
                abspath = os.path.join(abspath, initial_path)

        if not os.path.isdir(abspath):
            os.makedirs(abspath)

        assert os.path.exists(abspath), "Path doesn't exist: {}".format(
            abspath
        )
        return abspath
    """

    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir',
                        help='directory to download / to load datasets',
                        default=os.path.join('data', 'downloads'))

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    try:
        start_time = datetime.now()

        args = get_arguments()
        setup_logger(args)
        logger.debug('Command-line arguments:')
        for arg in vars(args):
            value = getattr(args, arg)
            logger.debug('\t{argument_key}:\t{value}'.format(argument_key=arg,
                                                           value=value))

        logger.debug(start_time)

        main(args)

        finish_time = datetime.now()
        logger.debug(finish_time)
        logger.debug('Execution time: {time}'.format(
            time=(finish_time - start_time)
        ))
        logger.debug("#" * 20 + " END EXECUTION " + "#" * 20)

        sys.exit(0)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Something happened and I don't know what to do D:")
        sys.exit(1)
