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
import requests
logger = logging.getLogger(__appname__)

"""
def directory_in_cwd(directory, create=True):
    cwd = os.path.dirname(os.path.abspath(__file__))
    directory_name = os.path.dirname(directory)
    directory_abs_path = os.path.join(cwd, directory)
    os.makedirs(directory_abs_path, exist_ok=create)
    return directory_abs_path

CACHE_DIR = directory_in_cwd('cache', create=False)
"""

IMPLEMENTED_ARCHIVE_EXTENSIONS = ['zip', 'tgz']
DATASETS_DOWNLOAD = {
    'CNN.Qian‎.zip':
        'https://docs.google.com/uc?id=0B6PIPLNXk0o1MFp0RzlwVlE0MnM&export=download',
    'CNN.DMQA.tgz':
        'https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ',
}
DATASET_DOWNLOADER_SCRIPT = 'gdown.pl'
DATASET_DOWNLOADER_URL = 'https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl'
DATASET_DOWNLOADER_COMMAND = 'perl "{script_location}" "{url}" "{download_to}"'


def main(args):
    dataset_dir = get_dataset_dir(args.dataset_dir)

    archive_files = get_datasets(indir=dataset_dir)
    if not archive_files:
        raise ('Error loading datasets. Please download from the following urls:\n'
               '\thttp://cs.nyu.edu/~kcho/DMQA/\n'
               '\thttps://sites.google.com/site/qianmingjie/home/datasets/cnn'
               '-and-fox-news')

    for f in archive_files:
        pass


def get_dataset_dir(dataset_dir):
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), dataset_dir)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    return dataset_dir


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
    import subprocess
    logger.debug('Downloading datasets from Internet')
    logger.debug('\n'.join(DATASETS_DOWNLOAD.values()))

    script_location = os.path.join(to, DATASET_DOWNLOADER_SCRIPT)
    if not os.path.isfile(script_location):
        logger.debug("Dataset downloader '{}' not found. Retrieving "
                     "from {}".format(
            DATASET_DOWNLOADER_SCRIPT,
            DATASET_DOWNLOADER_URL))
        import urllib.request
        downloader_opener = urllib.request.URLopener()
        downloader_opener.retrieve(DATASET_DOWNLOADER_URL, script_location)

    for filename in DATASETS_DOWNLOAD:
        url = DATASETS_DOWNLOAD[filename]
        destination = os.path.join(to, filename)
        download(url=url, to=destination)
        """
        logger.debug('\t{}'.format(url))
        logger.debug('\t  |')
        logger.debug('\t  v')
        logger.debug('\t{}'.format(destination))
        logger.debug('\t'+'-'*60)

        subprocess.run(
            DATASET_DOWNLOADER_COMMAND.format(
                script_location=script_location, url=url,
                download_to=destination)
        )
        """
        yield destination

def download(url, to):
    logger.debug('Downloading {0} to {1}'.format(url, to))

    # sourced from: http://stackoverflow.com/a/15645088
    """
    with open(to, "wb") as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)

        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
    """

import hashlib
def retrieve(articles, cache_in):
    """
    for article in articles:
        yield retrieve_article(article, cache)
    """
    logger.debug('Retrieving articles, either from website or cache')
    text = ''
    url = ''
    retrieved_article_count = 0
    for article in articles:
        text = retrieve_article(article, cache_in)
        url = article.url
        retrieved_article_count += 1
        break
    logger.debug('Retrieved {count} article{plural}'.format(
            count=retrieved_article_count,
            plural='s' if retrieved_article_count > 1 else ''))
    return text, url


def retrieve_article(article, cached_in=None):
    logger.debug('Accessing {}'.format(article.url))
    md5 = hashlib.md5()
    md5.update(article.url.encode('utf-8'))
    filename = '{hash}.txt'.format(h.hexdigest())
    filepath = os.path.join(filename)

    txt = ''
    if os.path.isfile(filepath):
        with open(filepath) as f:
            txt = f.read()
        logger.debug('Cached at {}'.format(filepath))
        

    else:
        logger.debug('Cache not found. Downloading article')
        article.download()
        article.parse()
        article.nlp()

        with open(filepath, 'w') as f:
            txt = article.text
            f.write(txt)
        logger.debug('Caching to {}'.format(filepath))

    return txt



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

    parser.add_argument('-n', '--newspaper', dest='newspaper_url',
                        default='http://www.cnn.com/',
                        help='URL for target newspaper')
    def directory_in_cwd(directory, create=True):
        cwd = os.path.dirname(os.path.abspath(__file__))
        directory_name = os.path.dirname(directory)
        directory_abs_path = os.path.join(cwd, directory)
        os.makedirs(directory_abs_path, exist_ok=create)
        return directory_abs_path

    parser.add_argument('-a', '--archive-to', dest='cache_to',
                        default=directory_in_cwd('cache'), type=directory_in_cwd,
                        help='cache newspaper articles to directory')
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
