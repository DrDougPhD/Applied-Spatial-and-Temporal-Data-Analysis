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


import argparse
from datetime import datetime
import sys
import os
import logging
import newspaper

logger = logging.getLogger(__appname__)


def main(args):
    """
    cnn_paper = newspaper.build('http://cnn.com')
    archive(cnn_paper)
    for article in cnn_paper.articles:
        preprocess(article)
        articles.download()
        article.parse()
        article.nlp()
        print(article.text)
    print(cnn_paper.articles)
    """
    #unique_words = identify_unique_words(cnn_paper.articles)
    #preprocess(cnn_paper.articles[0])

import hashlib
def archive(newspaper):
    for article in newspaper.articles:
        md5 = hashlib.md5()
        md5.update(article.url)
        filename = h.hexdigest()
        filepath = 'blah'

def preprocess(article):
    article.download()
    article.parse()
    article.nlp()
    print(article.text)
    print('-'*80)
    print(article.url)
    #for word in article.text.split():
        

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
                        default=False, help='verbose output')
    def directory_in_cwd(directory):
        cwd = os.path.dirname(os.path.abspath(__file__))
        directory_name = os.path.dirname(directory)
        directory_abs_path = os.path.join(cwd, directory)
        os.makedirs(directory_abs_path, exist_ok=True)
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
