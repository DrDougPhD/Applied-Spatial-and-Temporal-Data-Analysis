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

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = ENGLISH_STOP_WORDS.union(
    'new says time just like told cnn according did make way really dont going know said'.split())
import itertools
from scipy.spatial import distance
import csv
import numpy
import shutil
from progressbar import ProgressBar
import math
import pickle

# Custom modules
import plots
import dataset


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


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
    """
    def equal_nonzero(tup):
        s = sorted(tup)
        return s[0] != 0 and s[0] == s[1]
    """

    # TODO: zip together using numpy
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
    # select the distance functions that will be used in this script
    if distance_fns is None:
        distance_fns = ACTIVATED_DISTANCE_FNS
    else:
        distance_fns = [fn for fn in ACTIVATED_DISTANCE_FNS
                        if fn.__name__ in distance_fns]

    selected_articles = dataset.get(n=n, from_=dataset_dir,
                                    randomize=randomize)
    # loading data ends
    ###########################################################################

    # compute pairwise similarities between selected articles
    logger.info(hr('Pairwise Similarities'))

    if args.no_stopwords:
        logger.info('No stopwords will be used')
        stopwords = frozenset([])
    else:
        logger.info('Using stopwords')
        stopwords = STOP_WORDS

    similarity_calculator = PairwiseSimilarity(selected_articles,
                                               method=method,
                                               stopwords=stopwords)
    similarity_calculator.save_matrix_to(directory=DATA_DIR)
    similarity_calculator.save_aggregate_feature_counts(directory=DATA_DIR)
    # preprocessing ends
    ############################################################################

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


# compare.py

class PairwiseSimilarity(object):
    def __init__(self, corpus, method, stopwords):
        self.corpus = corpus
        self.method = method

        # specify method in which corpus is repr'd as matrix:
        #  1. an existence matrix (0 if token is abscent, 1 if present)
        #  2. a term freq matrix (element equals token count in doc)
        #  3. Tf-Idf matrix
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=1,
                                              stop_words=stopwords)
        else:
            # matrix will be converted to binary matrix further down
            self.vectorizer = CountVectorizer(min_df=1,
                                              stop_words=stopwords)

        plain_text = [str(document) for document in self.corpus]
        self._matrix = self.vectorizer.fit_transform(plain_text)
        for i in range(len(corpus)):
            vector = self._matrix.getrow(i).toarray()[0]
            doc = corpus[i]
            if method == 'existence':
                # convert vector into a binary vector (only 0s and 1s)
                # vector = [ int(bool(e)) for e in vector ]
                vector = (vector != 0).astype(int).toarray()[0]
            doc.vector = vector

        self.features = self.vectorizer.get_feature_names()
        logger.info('{} unique tokens'.format(len(self.features)))

    def pairwise_compare(self, by, save_to=None):
        progress = None
        i = 0
        n = int(nCr(len(self.corpus), 2))
        if __name__ == '__main__':
            progress = ProgressBar(
                max_value=n)

        similarity_calculations = []
        for u, v in itertools.combinations(self.corpus, 2):

            if progress:
                progress.update(i)
                i += 1

            comparison = ComparedArticles(u, v, by, self.features)
            logger.debug(comparison)
            logger.debug('-' * 80)
            similarity_calculations.append(comparison)

        if progress:
            progress.finish()

        # sort similarities by their normalized scores
        similarity_calculations.sort(key=lambda c: c.normalized, reverse=True)

        if save_to:
            similarities_file = os.path.join(
                save_to,
                '{method}.{distance}.{n}.tsv'.format(distance=by.__name__,
                                                     method=self.method,
                                                     n=n))
            with open(similarities_file, 'w') as f:
                CREATED_FILES.append(similarities_file)

                # find the length of the feature which occurs most commonly in
                # both articles. for pretty printing
                highest_feat_max_len_obj = max(
                    similarity_calculations,
                    key=lambda x: len(x.highest_common_feat.name))
                highest_feat_max_length = len(
                    highest_feat_max_len_obj.highest_common_feat.name)

                # find the article title that is the shortest. make pretty
                art_w_short_title = max([c.article[0]
                                         for c in similarity_calculations],
                                        key=lambda r: len(r.title))
                short_title_len = len(art_w_short_title.title) + 4

                f.write('{score:^10}\t'
                        '{normalized:^10}\t'
                        '{highest_common_feature}\t'
                        '{highest_common_feature_score:^10}\t'
                        '{title}\t'
                        'Article #2\n'.format(
                    title='Article #1'.ljust(short_title_len),
                    score='score',
                    normalized='similarity',
                    highest_common_feature='mcf'.center(
                        highest_feat_max_length),
                    highest_common_feature_score='# occurs',
                ))

                for calc in similarity_calculations:
                    f.write('{score:10.5f}\t'
                            '{normalized:10.5f}\t'
                            '{highest_common_feature}\t'
                            '{highest_common_feature_score:10.5f}\t'
                            '{art1}\t'
                            '"{art2}"\n'.format(
                        art1='"{}"'.format(calc.article[0].title)
                            .ljust(short_title_len),
                        art2=calc.article[1].title,
                        score=calc.score,
                        normalized=calc.normalized,
                        highest_common_feature=calc.highest_common_feat
                            .name.ljust(
                            highest_feat_max_length),
                        highest_common_feature_score=calc.highest_common_feat
                            .score
                    ))

        return similarity_calculations

    def save_matrix_to(self, directory):
        logger.info('Saving TF matrix to file')
        matrix_file = os.path.join(directory, self.method + '_matrix.csv')
        CREATED_FILES.append(matrix_file)
        with open(matrix_file, 'w') as f:
            csvfile = csv.writer(f, delimiter='|')
            csvfile.writerow(self.features)
            csvfile.writerows(self._matrix.toarray())

    def save_aggregate_feature_counts(self, directory):
        features_file = os.path.join(directory, 'aggregate_feature_counts.csv')
        CREATED_FILES.append(features_file)
        with open(features_file, 'w') as counts_file:
            csvfile = csv.writer(counts_file)
            csvfile.writerow(['token', 'count'])

            summed_vector = sum(self._matrix).toarray()[0]
            csvfile.writerows(sorted(
                zip(self.features, summed_vector),
                key=lambda e: e[1],
                reverse=True,
            ))


# TODO: make class for distance functions and normalizing them
"""
class BaseDistanceFunctor(object):
    def __init__(self):
        pass

    def __call__(self):
        # call distance function

    def normalize(self):
        score = self()
        ...
"""


class ComparedArticles(object):
    class HighestCommonFeature(object):
        def __init__(self, articles, features, max_or_min=max):
            u, v = map(lambda x: x.vector, articles)
            # only sum up token occurrences for tokens that appear in both documents
            shared_appearances = (u + v) * (u != 0) * (v != 0)

            (i,), score = max_or_min(numpy.ndenumerate(shared_appearances),
                                     key=lambda e: e[1])
            self.name = features[i]
            self.score = score

    def __init__(self, art1, art2, fn, features):
        # sort articles by title
        if len(art1.title) < len(art2.title):
            self.article = [art1, art2]
        else:
            self.article = [art2, art1]
        self.score = fn(art1.vector, art2.vector)
        self.distance_fn = fn.__name__

        # normalize the score based on the distance function used
        if self.distance_fn == 'euclidean':
            # [ 0, +inf ) --(flipped)-> ( 0, +inf ) -> ( 0, 1 ] --(flip)-> [ 0, 1 )
            self.normalized = 1 / (self.score + 1)
        elif self.distance_fn == 'cosine':
            # [ -1, 1 ] -> [ 0, 2 ] -> [ 0, 1 ]
            self.normalized = 1 - self.score
        elif self.distance_fn == 'jaccard':
            self.normalized = self.score  # no need to normalize

        self.highest_common_feat = ComparedArticles.HighestCommonFeature(
            articles=self.article,
            features=features)

    def __str__(self):
        return '{0}\n' \
               'vs.\n' \
               '{1}\n' \
               '== SCORE ({2}): {3}'.format(
            repr(self.article[0]), repr(self.article[1]),
            self.distance_fn,
            self.score)


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
