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
from bs4 import BeautifulSoup
import glob
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
ENGLISH_STOP_WORDS = list(ENGLISH_STOP_WORDS).extend([
  'said',
])
import shutil
import subprocess
import hashlib
import itertools
from scipy.spatial import distance
import csv
import numpy
from progressbar import ProgressBar
import math
import pickle

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


try:    # this is my own package, but it might not be present
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char*30 + title + line_char*30


logger = logging.getLogger(__appname__)


RANDOM_SEED = 0
random.seed(RANDOM_SEED)


IMPLEMENTED_ARCHIVE_EXTENSIONS = ['zip', 'tgz']
EXTRACTOR_SCRIPT_SOURCE = 'http://askubuntu.com/a/338759'
EXTRACTOR_SCRIPT = 'extract.sh'

DATA_DIR = 'data'
DEFAULT_DATASET_DIR = os.path.join(DATA_DIR, 'downloads')
MATRIX_FILE_PATH = os.path.join(DATA_DIR, 'matrix.csv')
FEATURES_FILE_PATH = os.path.join(DATA_DIR, 'feature_counts.csv')
SELECTED_ARTICLE_ARCHIVE = os.path.join(DATA_DIR, 'articles')
PICKLED_RESULTS = os.path.join(DATA_DIR, 'pickled_seed{0}_{1}.p'.format(
    RANDOM_SEED, '{num_items}'))


# note: jaccard from scipy is not jaccard similarity, but rather computing
#  the jaccard dissimilarity! i.e. numerator is cTF+cFT, not cTT
def jaccard(u, v):
    def equal_nonzero(tup):
        s = sorted(tup)
        return s[0] != 0 and s[0] == s[1]

    z = list(zip(u, v))
    positive_results = map(equal_nonzero, z)
    non_zero_results = map(all, z)
    return sum(positive_results)/sum(non_zero_results)


ACTIVATED_DISTANCE_FNS = [ distance.euclidean, jaccard, distance.cosine ]

CREATED_FILES = []

def process(n=10, dataset_dir=DEFAULT_DATASET_DIR, method='tf',
            distance_fns=None, randomize=False, args=None):
    # select the distance functions that will be used in this script
    if distance_fns is None:
        distance_fns = ACTIVATED_DISTANCE_FNS
    else:
        distance_fns = [ fn for fn in ACTIVATED_DISTANCE_FNS
                            if fn.__name__ in distance_fns ]

    # TODO: refactor this into a class perhaps?
    dataset_dir = get_dataset_dir(dataset_dir)
    archive_files = get_datasets(indir=dataset_dir)
    extractor_script = os.path.join(dataset_dir, EXTRACTOR_SCRIPT)
    decompressed_dataset_directories = {}
    for f in archive_files:
        filename = os.path.basename(f)
        filename_prefix = '.'.join(filename.split('.')[:-1])
        extract_to = os.path.join(dataset_dir, filename_prefix)
        decompress(f, to=extract_to, dataset_dir=dataset_dir)
        decompressed_dataset_directories[filename] = extract_to

    logger.info(hr('Datasets to preprocess'))
    for dbname in decompressed_dataset_directories:
        dir = decompressed_dataset_directories[dbname]
        logger.info(dir)

    # randomly select articles
    logger.info(hr('Article Selection'))
    selector = ArticleSelector(decompressed_dataset_directories)
    selected_articles = selector.get(n, randomize=randomize,
                                     archive_to=SELECTED_ARTICLE_ARCHIVE)
    CREATED_FILES.append(SELECTED_ARTICLE_ARCHIVE)
    

    # compute pairwise similarities between selected articles
    logger.info(hr('Pairwise Similarities'))
    data = {}
    similarity_calculater = PairwiseSimilarity(selected_articles,
                                               method=method)
    similarity_calculater.save_matrix_to(matrix_file=MATRIX_FILE_PATH,
                                         features_file=FEATURES_FILE_PATH)
    CREATED_FILES.append(MATRIX_FILE_PATH)
    CREATED_FILES.append(FEATURES_FILE_PATH)

    for fn in distance_fns:
        logger.info(hr(fn.__name__, line_char='-'))
        similarities = similarity_calculater.pairwise_compare(
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
          #shutil.copy(f, args.relocate_files_to)
          os.rename(f, os.path.join(args.relocate_files_to, filename))


    # pickle the data
    pickle.dump(data, open(PICKLED_RESULTS.format(num_items=n), 'wb'))
    return data



class ArticleSelector(object):
    """
    Given a dataset of articles, select a subset for further processing.
    Obtain the article's title, category, and plain text.
    """

    class BaseDatasetAccessor(object):
        """
        Base class for accessing articles within a given directory.
        """
        def __init__(self, dir):
            assert os.path.isdir(dir), \
                   "Directory doesn't exist: {}".format(dir)
            self.stored_in = dir

        def retrieve(self):
            raise NotImplemented('.retrieve() method has not been implemented')


    class QianDataset(BaseDatasetAccessor):
        """
        Retrieve article files from the Qian CNN dataset located in a specified
        directory.
        """
        def retrieve(self):
            logger.debug('Retrieving articles from within {}'.format(
                self.stored_in))
            article_categories_in = os.path.join(self.stored_in, 'Raw')
            categories = os.listdir(article_categories_in)
            logger.debug('Category directories: {}'.format(categories))
            for category in categories:
                abspath = os.path.join(article_categories_in, category)
                for article_path in self._retrieve_from_category(abspath):
                    yield QianArticle(path=article_path)

        def _retrieve_from_category(self, category_directory):
            glob_path = os.path.join(category_directory, 'cnn_*.txt')
            for filename in glob.glob(glob_path):
                #logger.debug('\t -->  {}'.format(filename))
                yield os.path.join(category_directory, filename)

    article_accessor = {
        # access articles by the file's bytesize
        136208660: lambda dir: ArticleSelector.QianDataset(dir),
    }

    def __init__(self, datasets):
        file_sizes = [ (file, os.stat(datasets[file]+'.zip').st_size)
                       for file in datasets ]
        self.accessors = [ ArticleSelector.article_accessor[size](datasets[k])
                           for k, size in file_sizes ]


    def get(self, count, randomize=True, archive_to=None):
        # evenly distribute articles selected from each located dataset
        articles = []
        for selector in self.accessors:
            subset = selector.retrieve()
            articles.extend(subset)
            assert len(articles) > 0, 'No articles found for {}'.format(
                selector.__class__.__name__)

        # shuffle and truncate set to the specified size
        if randomize:
            logger.debug('Random selection of {} articles'.format(count))
            try:
              selected_articles = random.choices(articles, k=count)
            except AttributeError:  # Python 3.6 is not installed
              random.shuffle(articles)
              selected_articles = articles[:count]
        else:
            logger.debug('Non-random selection of {} articles'.format(count))
            selected_articles = articles[:count]

        if archive_to:
            logger.debug('Copying files to {}'.format(archive_to))
            os.makedirs(archive_to, exist_ok=True)
            [ shutil.copy(f.path, archive_to) for f in selected_articles ]
            logger.debug('{} files copied'.format(len(selected_articles)))

        return selected_articles


class PairwiseSimilarity(object):
    def __init__(self, corpus, method):
        self.corpus = corpus

        # specify method in which corpus is repr'd as matrix:
        #  1. an existence matrix (0 if token is abscent, 1 if present)
        #  2. a term freq matrix (element equals token count in doc)
        #  3. Tf-Idf matrix
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=1,
                                              stop_words=ENGLISH_STOP_WORDS)
        else:
            # matrix will be converted to binary matrix further down
            self.vectorizer = CountVectorizer(min_df=1,
                                              stop_words=ENGLISH_STOP_WORDS)

        plain_text = [ str(document) for document in self.corpus ]
        self._matrix = self.vectorizer.fit_transform(plain_text)
        for i in range(len(corpus)):
            vector = self._matrix.getrow(i).toarray()[0]
            doc = corpus[i]
            if method == 'existence':
                # convert vector into a binary vector (only 0s and 1s)
                vector = [ int(bool(e)) for e in vector ]
            doc.vector = vector

        self.features = self.vectorizer.get_feature_names()
        logger.info('{} unique tokens'.format(len(self.features)))

    def pairwise_compare(self, by, save_to=None):
        progress = None
        i = 0
        if __name__ == '__main__':
            progress = ProgressBar(
                max_value=nCr(len(self.corpus), 2))

        similarity_calculations = []
        for u,v in itertools.combinations(self.corpus, 2):

            if progress:
                progress.update(i)
                i += 1

            comparison = ComparedArticles(u, v, by, self.features)
            logger.debug(comparison)
            logger.debug('-'*80)
            similarity_calculations.append(comparison)

        if progress:
            progress.finish()

        if save_to:
            similarities_file = os.path.join(save_to, '{distance}.tsv'.format(
                  distance=by.__name__))
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
                art_w_short_title = max([ c.article[0]
                                          for c in similarity_calculations ],
                                        key=lambda r: len(r.title))
                short_title_len = len(art_w_short_title.title) + 4

                f.write('{score:^10}\t'\
                        '{normalized:^10}\t'\
                        '{highest_common_feature}\t'\
                        '{highest_common_feature_score:^10}\t'\
                        '{title}\t'\
                        'Article #2\n'.format(
                    title='Article #1'.ljust(short_title_len),
                    score='score',
                    normalized='similarity',
                    highest_common_feature='mcf'.center(highest_feat_max_length),
                    highest_common_feature_score='# occurs',
                ))


                for calc in similarity_calculations:
                    f.write('{score:10.5f}\t'\
                            '{normalized:10.5f}\t'\
                            '{highest_common_feature}\t'\
                            '{highest_common_feature_score:10.5f}\t'\
                            '{art1}\t'\
                            '"{art2}"\n'.format(
                        art1='"{}"'.format(calc.article[0].title)
                                   .ljust(short_title_len),
                        art2=calc.article[1].title,
                        score=calc.score,
                        normalized=calc.normalized,
                        highest_common_feature=calc.highest_common_feat\
                                                   .name.ljust(
                                                      highest_feat_max_length),
                        highest_common_feature_score=calc.highest_common_feat\
                                                         .score
                    ))

        # sort similarities by their normalized scores
        similarity_calculations.sort(key=lambda c: c.normalized, reverse=True)
        return similarity_calculations

    def save_matrix_to(self, matrix_file, features_file):
        logger.info('Saving TF matrix to file')
        with open(matrix_file, 'w') as f:
            csvfile = csv.writer(f, delimiter='|')
            csvfile.writerow(self.features)
            csvfile.writerows(self._matrix.toarray())

        with open(features_file, 'w') as counts_file:
            csvfile = csv.writer(counts_file)
            csvfile.writerow(['token', 'count'])

            summed_vector = sum(self._matrix).toarray()[0]
            csvfile.writerows(zip(self.features, summed_vector))


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
            summed_vector = sum([a.vector for a in articles])
            i, score = max_or_min(enumerate(summed_vector),
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
            self.normalized = 1 - (1 / (self.score + 1))
        elif self.distance_fn == 'cosine':
            # [ -1, 1 ] -> [ 0, 2 ] -> [ 0, 1 ]
            self.normalized = 1 - self.score
        elif self.distance_fn == 'jaccard':
            self.normalized = self.score # no need to normalize

        self.highest_common_feat = ComparedArticles.HighestCommonFeature(
              articles=self.article,
              features=features)


    def __str__(self):
        return '{0}\n'\
               'vs.\n'\
               '{1}\n'\
               '== SCORE ({2}): {3}'.format(
                    repr(self.article[0]), repr(self.article[1]),
                    self.distance_fn,
                    self.score)


class NewspaperArticle(object):
    def __init__(self, path):
        assert os.path.isfile(path), 'File not found: {}'.format(path)
        self.path = path
        self.filename = os.path.basename(path)
        self.title = None
        self.abstract = None
        self.category = None
        self.vector = None


    def __radd__(self, other):
        return other + self.vector


    def __str__(self):
        """
        Return the plaintext of the article as a string.
        :return: string The plaintext of the article.
        """
        # simply iterate over every word in the document, removing newlines
        # and bad characters, and return as one long string
        return ' '.join(self)

    def __repr__(self):
        return '"{0.title}"\n'\
               '\tcategory: {0.category}\n'\
               '\tvector:   {1}'.format(self, self.vector)

    def __iter__(self):
        """
        Iterate through each word in this article.
        :return: string Next word in the article.
        """
        logger.debug(hr(
            title='Parsing through {}'.format(os.path.basename(self.path)),
            line_char='-'
        ))
        self._setup_reader()
        for w in self._next_word():
            yield w


class QianArticle(NewspaperArticle):
    punctuation_remover = str.maketrans('', '', string.punctuation)

    def _setup_reader(self):
        soup = BeautifulSoup(open(self.path), 'html.parser')
        self.title = soup.doc.title.text
        self.abstract = soup.doc.abstract.text
        self.text = soup.doc.find('text').text
        category_dir = os.path.basename(os.path.dirname(self.path))
        self.category = category_dir.split('_')[-1]

    def _next_word(self):
        for line in self.text.split('\n'):
            if self._matches_useless_line(line):
                continue

            if '(CNN)' in line:
                # first line of the article
                prefix_removed = line.split('(CNN)')[-1]
                line = prefix_removed

            for word in line.split():
                # word lowering is done internally by scikit-learn
                # word = word.lower()
                word = word.translate(QianArticle.punctuation_remover)
                if not word or word.isspace():
                    continue

                yield word.lower()

    def _matches_useless_line(self, line):
        if line.startswith('Watch Anderson Cooper'):
            # it's not a diss, Mr. Cooper. But the line containing that
            # text does not pertain to the article's contents
            return True

        return False


def get_dataset_dir(dataset_dir):
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), dataset_dir)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    return dataset_dir


def get_datasets(indir):
    files = set()
    for dirpath, _, filenames in os.walk(indir):
        files.update([ os.path.join(dirpath, f)
                       for f in filenames
                       if is_archive(f) ])
    if not files:
        raise Exception(
            'Error loading datasets. Please download from this url:\n'
            'https://sites.google.com'
            '/site/qianmingjie/home/datasets/cnn-and-fox-news')

    logger.debug('Archive files: {}'.format(files))
    return files


def is_archive(filename):
    extension = filename.split('.')[-1]
    if extension.lower() in IMPLEMENTED_ARCHIVE_EXTENSIONS:
        return True
    else:
        return False


def decompress(file, to, dataset_dir):
    if os.path.exists(to):
        logger.debug('Already existing file/dir at {}.'.format(to))
        logger.debug('No extraction will be done for {}.'.format(
            os.path.basename(file)))
        return

    extractor = os.path.join(dataset_dir, EXTRACTOR_SCRIPT)
    if not os.path.isfile(extractor):
        raise Exception('No archive extractor script found at {path}.\n'
                        'Create it from this post: {url}'.format(
            path=extractor,
            url=EXTRACTOR_SCRIPT_SOURCE))

    # take snapshot of directory so that the extracted directory can be spotted
    current_files = set(os.listdir(os.getcwd()))

    logger.info('Extracting dataset. This might take a while.')
    subprocess.run(['bash', extractor, file])
    logger.debug('Extraction complete.')

    logger.debug('Moving to {}'.format(to))
    new_files = list(set(os.listdir(os.getcwd())) - current_files)
    relocate(new_files, to)

    logger.info('Extraction complete. Uncompressed files'
                ' are within {}'.format(to))
    for f in os.listdir(to):
        dirname = os.path.basename(to)
        path = os.path.join('...', dirname, f)
        logger.debug('\t{}'.format(path))


def relocate(new_files, to):
    logger.debug('New files after extraction: {}'.format(new_files))
    if len(new_files) > 1:
        # move all files
        os.makedirs(to, exist_ok=True)
        for f in new_files:
            shutil.move(os.path.join(os.getcwd(), f), to)
        logger.debug('{} files moved.'.format(len(new_files)))

    elif len(new_files) == 1:
        new_path = os.path.join(os.getcwd(), new_files[0])
        if os.path.isfile(new_path):
            logger.debug('Extracted only one file.')
            os.makedirs(to, exist_ok=True)
            shutil.move(new_path, to)

        else:
            logger.debug('Extracted a whole directory.')
            os.rename(new_path, to)


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
                        default=False, #default=__dev__,
                        help='verbose output')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir',
                        help='directory to download / to load datasets',
                        default=DEFAULT_DATASET_DIR)
    parser.add_argument('-N', '--num-articles', dest='num_to_select',
                        type=int, default=5,
                        help='number of articles to select for analysis')
    parser.add_argument('-m', '--method', dest='method',
                        default='tf', choices=['tf', 'existence', 'tfidf'],
                        help='matrix representation of matrix'\
                             ' - i.e. tf, existence, tfidf')
    parser.add_argument('-D', '--distances', dest='distance_fns', nargs='+',
                        choices=[fn.__name__ for fn in ACTIVATED_DISTANCE_FNS],
                        help='distance functions to use (select 1 or more)')

    def directory_in_cwd(directory, create=True):
        cwd = os.path.dirname(os.path.abspath(__file__))
        directory_name = os.path.dirname(directory)
        directory_abs_path = os.path.join(cwd, directory)
        os.makedirs(directory_abs_path, exist_ok=create)
        return directory_abs_path

    parser.add_argument('-a', '--archive-to', dest='cache_to',
                        default=directory_in_cwd('cache'), type=directory_in_cwd,
                        help='cache newspaper articles to directory')
    parser.add_argument('-z', '--clean-up', dest='relocate_files_to',
                        type=directory_in_cwd, 
                        default=directory_in_cwd('results'),
                        help='delete files after execution (default: False)')
    parser.add_argument('-c', '--no-website', dest='website',
                        action='store_false', default=True,
                        help='specify if the website should not be run')

    args = parser.parse_args()
    return args


def website(data):
    from flask import Flask
    app = Flask(__name__, static_url_path='')

    from flask import render_template
    from flask import send_from_directory

    @app.route('/')
    def matrix_choices():
        return render_template('choices.html')


    @app.route('/<matrix_type>/<int:n>', defaults={'matrix_type': 'tf', 'n': 10})
    def similarities(matrix_type, n):
        return render_template('similarities.html', similarities=data)


    @app.route('/get/<filename>')
    def load_article(filename):
        return send_from_directory('results/articles', filename)

    app.run()


def load(n):
    data = from_pickle(n)
    if data is None:
        data = process(n=args.num_to_select, method=args.method,
                       dataset_dir=args.dataset_dir,
                       distance_fns=args.distance_fns,
                       args=args)
    else:
        logger.info('Data loaded from pickle')
    return data


def from_pickle(n):
    pfile = PICKLED_RESULTS.format(num_items=n)
    if not os.path.isfile(pfile):
        return None
    pkl = pickle.load(open(pfile, 'rb'))
    return pkl


def main(args):
    data = load(args.num_to_select)
    if args.website:
        website(data)


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
