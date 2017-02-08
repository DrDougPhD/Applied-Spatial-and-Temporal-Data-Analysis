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

try:    # this is my own package, but it might not be present
    from lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char*30 + title + line_char*30

logger = logging.getLogger(__appname__)


IMPLEMENTED_ARCHIVE_EXTENSIONS = ['zip', 'tgz']
EXTRACTOR_SCRIPT_SOURCE = 'http://askubuntu.com/a/338759'
EXTRACTOR_SCRIPT = 'extract.sh'

from scipy.spatial import distance
DISTANCE_FUNCTIONS = [ distance.euclidean, distance.jaccard, distance.cosine ]


def process(args, n=None):
    if n is None:
        n = args.num_to_select

    dataset_dir = get_dataset_dir(args.dataset_dir)
    archive_files = get_datasets(indir=dataset_dir)
    extractor_script = os.path.join(dataset_dir, EXTRACTOR_SCRIPT)
    decompressed_dataset_directories = {}
    for f in archive_files:
        filename = os.path.basename(f)
        filename_prefix = '.'.join(filename.split('.')[:-1])
        extract_to = os.path.join(dataset_dir, filename_prefix)
        decompress(f, to=extract_to, dataset_dir=dataset_dir)
        decompressed_dataset_directories[filename] = extract_to

    logger.info('-'*80)
    logger.info('Datasets to preprocess:')
    for dbname in decompressed_dataset_directories:
        dir = decompressed_dataset_directories[dbname]
        logger.info(dir)

    # randomly select articles
    logger.debug(hr('Article Selection'))
    selector = ArticleSelector(decompressed_dataset_directories)
    selected_articles = selector.get(n, randomize=not __dev__)

    data = {}
    similarity_calculater = PairwiseSimilarity(selected_articles)
    for fn in DISTANCE_FUNCTIONS:
        similarities = similarity_calculater.pairwise_compare(by=fn)
        data[fn.__name__] = similarities

    """
    return data
    """


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


    def get(self, count, randomize=True):
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
            selected_articles = random.choices(articles, k=count)
        else:
            logger.debug('Non-random selection of {} articles'.format(count))
            selected_articles = articles[:count]

        return selected_articles


import itertools
class PairwiseSimilarity(object):
    def __init__(self, corpus):
        self.corpus = corpus

        self.vectorizer = CountVectorizer(min_df=1)
        plain_text = [ str(document) for document in self.corpus ]
        self._matrix = self.vectorizer.fit_transform(plain_text)

        for i in range(len(corpus)):
            vector = self._matrix.getrow(i)
            doc = corpus[i]
            doc.vector = vector.toarray()

        self.features = self.vectorizer.get_feature_names()
        logger.debug('Unique tokens: {}'.format(self.features))


    def pairwise_compare(self, by):
        similarity_calculations = []
        for u,v in itertools.combinations(self.corpus, 2):
            comparison = ComparedArticles(u, v, by)
            logger.debug(comparison)
            similarity_calculations.append(comparison)
        return similarity_calculations


class ComparedArticles(object):
    def __init__(self, art1, art2, fn):
        self.article = [art1, art2]
        self.score = fn(art1.vector, art2.vector)
        self.distance_fn = fn.__name__

    def __str__(self):
        return '"{0.title}"\t:{0.vector}\n'\
               '\t vs. "{1.title}"\t:{1.vector}\n'\
               '\t Score: {2}'.format(
                    self.article[0], self.article[1], self.score)


class NewspaperArticle(object):
    def __init__(self, path):
        assert os.path.isfile(path), 'File not found: {}'.format(path)
        self.path = path
        self.title = None
        self.abstract = None
        self.vector = None


    def __str__(self):
        """
        Return the plaintext of the article as a string.
        :return: string The plaintext of the article.
        """
        # simply iterate over every word in the document, removing newlines
        # and bad characters, and return as one long string
        return ' '.join(self)


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


import string
class QianArticle(NewspaperArticle):
    punctuation_remover = str.maketrans('', '', string.punctuation)

    def _setup_reader(self):
        soup = BeautifulSoup(open(self.path), 'html.parser')
        self.title = soup.doc.title.text
        self.abstract = soup.doc.abstract.text
        self.text = soup.doc.find('text').text

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


from sklearn.feature_extraction.text import CountVectorizer
class BagOfWords(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = CountVectorizer(min_df=1)
        logger.debug('{} items in the supplied corpus'.format(len(corpus)))


    def baggify(self):
        """
        Find all unique words throughout every document supplied to this object.
        """
        # convert corpus, which hasn't been loaded from file, to strings
        corpus = [ str(document) for document in self.corpus ]
        self._matrix = self.vectorizer.fit_transform(corpus)
        self.features = self.vectorizer.get_feature_names()
        logger.debug('Unique tokens: {}'.format(self.features))


    def matrix(self, save_to=None):
        if save_to is not None:
            pass

        return self._matrix


def main(args):
    process(args=args, n=10)
    """
    dataset_dir = get_dataset_dir(args.dataset_dir)
    archive_files = get_datasets(indir=dataset_dir)
    extractor_script = os.path.join(dataset_dir, EXTRACTOR_SCRIPT)
    decompressed_dataset_directories = {}
    for f in archive_files:
        filename = os.path.basename(f)
        filename_prefix = '.'.join(filename.split('.')[:-1])
        extract_to = os.path.join(dataset_dir, filename_prefix)
        decompress(f, to=extract_to, dataset_dir=dataset_dir)
        decompressed_dataset_directories[filename] = extract_to

    logger.info('-'*80)
    logger.info('Datasets to preprocess:')
    for dbname in decompressed_dataset_directories:
        dir = decompressed_dataset_directories[dbname]
        logger.info(dir)

    # randomly select articles
    logger.debug(hr('Article Selection'))
    selector = ArticleSelector(decompressed_dataset_directories)
    selected_articles = selector.get(args.num_to_select, randomize=not __dev__,
                                     archive_to='blah')
    assert len(selected_articles) == args.num_to_select,\
        'Expected {0} articles, but received {1} articles'.format(
            args.num_to_select, len(selected_articles))
    #logger.debug('Selected articles ({} articles):'.format(len(
    #    selected_articles)))
    #import pprint
    #logger.debug(pprint.pformat(selected_articles))
    # corpus = selector.get(100, random=not __dev__, archive_to='blah')

    # break down articles into a bag of words
    logger.debug(hr('Bag of Words'))
    bag_of_words = BagOfWords(corpus=selected_articles)
    bag_of_words.baggify()
    matrix = bag_of_words.matrix(save_to='bag.mm')

    logger.debug(matrix.toarray()[0])
    logger.debug(type(matrix.toarray()[0]))
    logger.debug(len(matrix.toarray()))
    #logger.debug(matrix[0])
    #logger.debug(type(matrix[0]))


    similarity_calculater = PairwiseSimilarity(matrix)
    for fn in DISTANCE_FUNCTIONS:
        similarities = similarity_calculater.pairwise_compare(by=fn)
        #similarities.write_to(file='f', sort_by=fn.__name__)
    
    similarity_calculater = PairwiseSimilarity(matrix)# , distance_fns=[
    #    euclidean_distance, cosine_similarity, jaccard_similarity
    #]
    similarity_calculater.run()
    similarity_calculater.write_to(file='similarities.tsv')

        similarity_calculater.pairwise_combine_and_measure(by=fn)
        similarity_calculater.write_to(file='f', sort_by=fn.__name__)
    """


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


import shutil
import subprocess
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
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir',
                        help='directory to download / to load datasets',
                        default=os.path.join('data', 'downloads'))
    parser.add_argument('-n', '--newspaper', dest='newspaper_url',
                        default='http://www.cnn.com/',
                        help='URL for target newspaper')
    parser.add_argument('-N', '--num-articles', dest='num_to_select',
                        type=int, default=100,
                        help='number of articles to select for analysis')

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
