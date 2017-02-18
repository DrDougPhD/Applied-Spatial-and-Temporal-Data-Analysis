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

from scipy.spatial import distance
import numpy
import shutil

import pickle

from progressbar import ProgressBar

# Custom modules
import plots
import dataset
import preprocess


try:  # this is my own package, but it might not be present
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


# note: jaccard from scipy is not jaccard similarity, but rather computing
#  the jaccard dissimilarity! i.e. numerator is cTF+cFT, not cTT
def jaccard(u, v):
    equal = (v == u)
    are_zero = (u == 0)
    equal_nonzero = (are_zero == False) * equal
    both_zeros = equal * are_zero
    results_not_zeros = (both_zeros == False)
    return numpy.sum(equal_nonzero) / numpy.sum(results_not_zeros)

ACTIVATED_DISTANCE_FNS = [distance.euclidean, jaccard, distance.cosine]

CREATED_FILES = []


def process(n=10, dataset_dir=DEFAULT_DATASET_DIR, method='tf',
            distance_fns=None, randomize=True, args=None):
    selected_articles = dataset.get(n=n, from_=dataset_dir,
                                    randomize=randomize)
    # loading data ends
    ###########################################################################

    # compute pairwise similarities between selected articles
    logger.info(hr('Pairwise Similarities'))
    similarity_calculator = preprocess.execute(corpus=selected_articles,
                                               exclude_stopwords=args.no_stopwords,
                                               method=method)
    similarity_calculator.save_matrix_to(directory=DATA_DIR)
    similarity_calculator.save_aggregate_feature_counts(directory=DATA_DIR)
    # preprocessing ends
    ############################################################################

    # select the distance functions that will be used in this script
    if distance_fns is None:
        distance_fns = ACTIVATED_DISTANCE_FNS
    else:
        distance_fns = [fn for fn in ACTIVATED_DISTANCE_FNS
                        if fn.__name__ in distance_fns]

    data = {}
    for fn in distance_fns:
        logger.info(hr(fn.__name__, line_char='-'))
        similarities = similarity_calculator.pairwise_compare(
            by=fn, save_to=DATA_DIR)
        data[fn.__name__] = similarities

    if args and args.relocate_files_to:
        logger.debug(hr('Relocating Created Files'))
        logger.debug('Storing files in {}'.format(args.relocate_files_to))
        for f in CREATED_FILES:
            logger.debug(f)
            filename = os.path.basename(f)
            dst = os.path.join(args.relocate_files_to, filename)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            # shutil.copy(f, args.relocate_files_to)
            os.rename(f, os.path.join(args.relocate_files_to, filename))

    # pickle the data
    pickle.dump(data, open(PICKLED_RESULTS.format(num_items=n), 'wb'))
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
    parser.add_argument('-D', '--distances', dest='distance_fns', nargs='+',
                        choices=[fn.__name__ for fn in ACTIVATED_DISTANCE_FNS],
                        default=[fn.__name__ for fn in ACTIVATED_DISTANCE_FNS],
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


def website(data, args):
    from flask import Flask
    app = Flask(__name__, static_url_path='')

    from flask import render_template
    from flask import send_from_directory

    @app.route('/')
    def matrix_choices():
        return render_template('choices.html', num_articles=args.num_to_select)

    @app.route('/<matrix_type>/<int:n>',
               defaults={'matrix_type': 'tf', 'n': 10})
    def similarities(matrix_type, n):
        return render_template('similarities.html', similarities=data)

    @app.route('/get/<filename>')
    def load_article(filename):
        return send_from_directory('results/articles', filename)

    app.run()


def load(args):
    n = args.num_to_select
    no_pickle = args.no_pickle
    distance_fns = args.distance_fns
    if no_pickle:
        data = None
    else:
        data = from_pickle(n, distance_fns)

    if data is None:
        data = process(n=n, method=args.method,
                       dataset_dir=args.dataset_dir,
                       distance_fns=distance_fns,
                       args=args)
    else:
        logger.info('Data loaded from pickle')
    return data


def from_pickle(n, fns):
    pfile = PICKLED_RESULTS.format(num_items=n)
    if not os.path.isfile(pfile):
        return None
    pkl = pickle.load(open(pfile, 'rb'))
    data = {k: pkl[k] for k in fns}
    return data


def main(args):
    data = load(args)
    if args.plot_results:
        plots.store_to(directory=DATA_DIR, data=data)

    elif args.website:
        website(data, args)


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
