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
from lxml import objectify
from lxml import etree
import glob
import random
from gensim import corpora

try:    # this is my own package, but it might not be present
    from lineheaderpadded import hr
except:
    hr = lambda x: '='*30 + x + '='*30

logger = logging.getLogger(__appname__)


IMPLEMENTED_ARCHIVE_EXTENSIONS = ['zip', 'tgz']
EXTRACTOR_SCRIPT_SOURCE = 'http://askubuntu.com/a/338759'
EXTRACTOR_SCRIPT = 'extract.sh'


def dmqa_preprocess(stored_within):
    logger.debug('Preprocessing DMQA CNN articles')
    text_files_within = os.path.join(stored_within, 'stories')
    logger.debug('Plaintext articles are stored in {}'.format(text_files_within))
    for f in os.listdir(text_files_within):
        yield os.path.join(text_files_within, f)


def qian_preprocess(stored_within):
    for i in range(3):
        yield ''


preprocessor = {
    'CNN.DMQA.tgz': dmqa_preprocess,
    'CNN.Qian.zip': qian_preprocess
}


class NewspaperArticle(object):
    def __init__(self, path):
        assert os.path.isfile(path), 'File not found: {}'.format(path)
        self.path = path


    def __str__(self):
        """
        Return the plaintext of the article as a string.
        :return: string The plaintext of the article.
        """
        return ''


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
    ignore_ampersands = etree.XMLParser(recover=True)

    def _setup_reader(self):
        article_xml = objectify.parse(self.path)
        #                              parser=QianArticle.ignore_ampersands)
        root = article_xml.getroot()
        self.title = root.TITLE.text
        self.abstract = root.ABSTRACT.text
        self.text = root.TEXT.text
        #self.text_lines = root.TEXT.split('\n')
        #self.line = next(self.text_lines)

    def _next_word(self):
        for line in self.text.split('\n'):
            for word in line.split():
                logger.debug('\t{}'.format(repr(word)))
                yield word



class ArticleSelector(object):
    """
    Given a dataset of articles, select a subset for further processing.
    Obtain the article's title, category, and plain text.
    """

    class BaseArticleAccessor(object):
        """
        Base class for accessing articles within a given directory.
        """
        def __init__(self, dir):
            assert os.path.isdir(dir), "ArticleAccessor: Directory doesn't " \
                                       "exist: {}".format(dir)
            self.stored_in = dir

        def retrieve(self):
            raise NotImplemented('.retrieve() method has not been implemented')


    class QianArticles(BaseArticleAccessor):
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
        'CNN.Qian.zip': lambda dir: ArticleSelector.QianArticles(dir),
    }

    def __init__(self, datasets):
        self.accessors = [ ArticleSelector.article_accessor[k](datasets[k])
                           for k in datasets ]


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
            return random.choices(articles, k=count)
        else:
            logger.debug('Non-random selection of {} articles'.format(count))
            return articles[:count]


class BagOfWords(object):
    def __init__(self, corpus):
        self.corpus = corpus
        logger.debug('{} items in the supplied corpus'.format(len(corpus)))


    def baggify(self):
        """
        Find all unique words throughout every document supplied to this object.
        :return: set A set of words existing in at least one document.
        """
        corpus_as_2D_list_of_words = self._breakup_into_2D_list_of_words()
        self.dictionary = corpora.Dictionary(corpus_as_2D_list_of_words)
        pass

    def matrix(self, save_to=None):
        pass


    def _breakup_into_2D_list_of_words(self):
        """
        Iterate over every article supplied to this object and break it up
        into a 2D list of words.
        :return: a 2D list of words, e.g. [ ['hello', 'rick'], ['hi'], ... ]
        """
        words_matrix = []
        for doc in self.corpus:
            words_matrix.append(doc)
        return words_matrix


class PairwiseSimilarity(object):
    def __init__(self, matrix):
        pass

    def pairwise_combine_and_measure(self, by):
        pass


    def write_to(self, file, sort_by):
        pass


def euclidean_distance(u, v):
    return 0


def cosine_similarity(u, v):
    return 0


def jaccard_similarity(u, v):
    return 0

def main(args):
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
    selected_articles = selector.get(100, randomize=not __dev__, archive_to='blah')
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
    words = bag_of_words.baggify()
    """
    matrix = bag_of_words.matrix(save_to='bag.mm')

    distance_fns = [euclidean_distance, cosine_similarity, jaccard_similarity]
    similarity_calculater = PairwiseSimilarity(matrix)
    for fn in distance_fns:
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
    files = set([
                os.path.join(indir, f)\
                for f in os.listdir(indir) \
                    if os.path.isfile(os.path.join(indir, f))\
                    and is_archive(f)])
    if not files:
        raise Exception(
            'Error loading datasets. Please download from the following  urls:\n'
            '\thttp://cs.nyu.edu/~kcho/DMQA/\n'
            '\thttps://sites.google.com/site/qianmingjie/home/datasets/cnn-and-fox-news')

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
