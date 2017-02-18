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
__dev__ = True  # used for debug messages in logs

import argparse
import logging
import os
import random
import sys
from datetime import datetime
import plots
import dataset
import preprocess
import processing
import postprocess
import website

try:
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

logger = logging.getLogger(__appname__)

RANDOM_SEED = 1  # 0 was throwing a weird error
random.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEFAULT_DATASET_DIR = os.path.join(DATA_DIR, 'downloads')
MATRIX_FILE_PATH = os.path.join(DATA_DIR, 'matrix.csv')
FEATURES_FILE_PATH = os.path.join(DATA_DIR, 'feature_counts.csv')
PICKLED_RESULTS = os.path.join(DATA_DIR, 'pickled_seed{0}_{1}.p'.format(
    RANDOM_SEED, '{num_items}'))

CREATED_FILES = []



def main(args):
    data = load(args)
    if args.website:
        website.run(data, args)


def load(args):
    n = args.num_to_select
    no_pickle = args.no_pickle
    distance_fns = args.distance_fns
    if no_pickle:
        data = None
    else:
        data = dataset.load.from_pickle(n, distance_fns, PICKLED_RESULTS)

    if data is None:
        data = process(n=n, method=args.method,
                       dataset_dir=args.dataset_dir,
                       distance_fns=distance_fns,
                       args=args)
    else:
        logger.info('Data loaded from pickle')
    return data


def process(n=10, dataset_dir=DEFAULT_DATASET_DIR, method='tf',
            distance_fns=None, randomize=True, args=None):
    # Load data
    selected_articles = dataset.get(n=n, from_=dataset_dir,
                                    randomize=randomize)

    # Preprocess
    logger.info(hr('Pairwise Similarities'))
    similarity_calculator = preprocess.execute(corpus=selected_articles,
                                               exclude_stopwords=args.no_stopwords,
                                               method=method)
    similarity_calculator.save_matrix_to(directory=DATA_DIR)
    similarity_calculator.save_aggregate_feature_counts(directory=DATA_DIR)

    # Process
    data = processing.go(calc=similarity_calculator,
                         funcs=distance_fns,
                         store_in=DATA_DIR)

    # Postprocessing
    postprocess.these(data=data, n=n,
                      file_relocation=args.relocate_files_to,
                      files=CREATED_FILES,
                      pickle_to=PICKLED_RESULTS)

    # Plotting
    if args.plot_results:
        plots.store_to(directory=DATA_DIR, data=data)

    return data


def setup_logger(args):
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', __appname__ + '.log')
    CREATED_FILES.append(log_file)
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
                        default=False,  # default=__dev__,
                        help='verbose output')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir',
                        help='directory to download / to load datasets',
                        default=DEFAULT_DATASET_DIR)
    parser.add_argument('-N', '--num-articles', dest='num_to_select',
                        type=int, default=5,
                        help='number of articles to select for analysis')
    parser.add_argument('-m', '--method', dest='method',
                        default='tf', choices=['tf', 'existence', 'tfidf'],
                        help='matrix representation of matrix' \
                             ' - i.e. tf, existence, tfidf')

    function_choices = [fn.__name__ for fn in processing.ACTIVATED_DISTANCE_FNS]
    parser.add_argument('-D', '--distances', dest='distance_fns', nargs='+',
                        choices=function_choices,
                        default=function_choices,
                        help='distance functions to use (select 1 or more)')

    def directory_in_cwd(directory, create=True):
        cwd = os.path.dirname(os.path.abspath(__file__))
        directory_name = os.path.dirname(directory)
        directory_abs_path = os.path.join(cwd, directory)
        os.makedirs(directory_abs_path, exist_ok=create)
        return directory_abs_path

    parser.add_argument('-a', '--archive-to', dest='cache_to',
                        default=directory_in_cwd('cache'),
                        type=directory_in_cwd,
                        help='cache newspaper articles to directory')
    parser.add_argument('-z', '--clean-up', dest='relocate_files_to',
                        type=directory_in_cwd,
                        default=directory_in_cwd('results'),
                        help='delete files after execution (default: False)')
    parser.add_argument('-c', '--no-website', dest='website',
                        action='store_false', default=True,
                        help='specify if the website should not be run')
    parser.add_argument('-f', '--force-recompute', dest='no_pickle',
                        action='store_true', default=False,
                        help='force recomputing corpus w/o using pickle')
    parser.add_argument('-s', '--no-stop-words', dest='no_stopwords',
                        action='store_true', default=False,
                        help='perform analysis without using stopwords (default: use stopwords)')
    parser.add_argument('-p', '--plot', dest='plot_results',
                        action='store_true', default=False,
                        help='create plots of the results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        start_time = datetime.now()

        args = get_arguments()
        setup_logger(args)
        logger.info('Command-line arguments:')
        for arg in vars(args):
            value = getattr(args, arg)
            logger.info('\t{argument_key}:\t{value}'.format(argument_key=arg,
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
