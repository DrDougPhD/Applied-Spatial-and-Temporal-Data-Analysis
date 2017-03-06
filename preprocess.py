import numpy
from progressbar import ProgressBar
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import logging
import os
import csv

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import config

STOP_WORDS = ENGLISH_STOP_WORDS.union(
    'new says time just like told cnn according did make way really'
    ' dont going know said to before according her out might ago'
    ' subject had'.split())
logger = logging.getLogger('cnn.'+__name__)
import utils


@utils.pickled('method')
def preprocess(corpus, exclude_stopwords, method, mrmr):
    if not exclude_stopwords:
        logger.info('No stopwords will be used')
        stopwords = frozenset([])
    else:
        logger.info('Using stopwords')
        stopwords = STOP_WORDS

    logger.debug('Vectorizing corpus')
    vectorizer = CorpusVectorizer(corpus=corpus,
                                  method=method,
                                  stopwords=stopwords,
                                  mrmr=mrmr)
    vectorizer.to_csv()
    return vectorizer


class CorpusVectorizer(object):
    def __init__(self, corpus, method, stopwords, mrmr):
        self.corpus = corpus
        self.method = method
        self.count = len(corpus)

        logger.debug('Reading articles into memory')
        plain_text = []
        self.class_names = set()
        progress = ProgressBar(
            max_value=len(corpus))
        for i, document in enumerate(corpus):
            plain_text.append(str(document))
            self.class_names.add(document.category)
            progress.update(i)
        progress.finish()

        self.plain_text = plain_text

        # isolate terms that will be preserved
        irrelevant_features = self._load_mrmr(mrmr, plain_text)
        terms_to_remove = stopwords.union(irrelevant_features)

        # determine the unique categories represented in the dataset
        class_to_index = self.class_to_index = {
            k: i for i, k in enumerate(self.class_names)}
        self.classes = numpy.array([class_to_index[document.category]
                                    for document in corpus])

        logger.debug('Transforming articles into vector space')
        # specify method in which corpus is repr'd as matrix
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=1,
                                              stop_words=terms_to_remove)
        else:
            # matrix will be converted to binary matrix further down
            self.vectorizer = CountVectorizer(min_df=1,
                                              stop_words=terms_to_remove)

        self.matrix = self.vectorizer.fit_transform(plain_text)
        self.features = self.vectorizer.get_feature_names()
        logger.debug('{} unique tokens'.format(len(self.features)))

        # Assign vector representation to each article
        for i in range(len(corpus)):
            vector = self.matrix.getrow(i).toarray()[0]

            # if method == 'existence':
            #     # convert vector into a binary vector (only 0s and 1s)
            #     # vector = [ int(bool(e)) for e in vector ]
            #     vector = (vector != 0).astype(int).toarray()[0]
            corpus[i].vector = vector

    def __iter__(self):
        for article in self.corpus:
            yield article

    def _load_mrmr(self, mrmr, corpus):
        if mrmr is None:
            return []

        logger.debug('Loading mRMR features from "{}"'.format(mrmr))
        # load good features from mrmr file
        mrmr_good_features = set()
        try:
            with open(mrmr) as f:
                for line in f:
                    columns = line.split()
                    mrmr_good_features.add(columns[2])
            logger.debug('mRMR relevant features: {}'.format(mrmr_good_features))

        except:
            logger.exception('No mRMR file at {}. Please create it.'.format(mrmr))
            raise

        # do a quick pass over the corpus data to find all unique features
        # contained within
        unique_features = set()
        for i, article in enumerate(corpus):
            unique_terms_in_article = set(article.split())
            unique_features.update(unique_terms_in_article)
            # logger.debug('Article #{0} has {1} unique features, bringing '
            #              'total unique features up to {2}'.format(
            #                 i,
            #                 len(unique_terms_in_article),
            #                 len(unique_features)))

        # remove the good features, resulting in the irrelevant features
        # being left over
        irrelevant_features = unique_features - mrmr_good_features
        logger.debug('{0} irrelevant features found out of {1} total'.format(
            len(irrelevant_features),
            len(unique_features)
        ))

        return irrelevant_features

    def to_csv(self):
        unpreprocessed_vectorizer = CountVectorizer(min_df=1,
                                                    stop_words='english')
        matrix = list(unpreprocessed_vectorizer.fit_transform(self.plain_text)\
                                               .toarray())

        output_filepath = os.path.join(config.RESULTS_DIR,
                                       'corpus.{0}.{1}.csv'.format(
                                           self.method,
                                           self.count,
                                       ))

        logger.debug('Saving the matrix representation '
                     'of the corpus to {}'.format(
            output_filepath
        ))

        with open(output_filepath, 'w') as f:
            output_csv = csv.writer(f)
            header = ['category',
                      *unpreprocessed_vectorizer.get_feature_names()]
            output_csv.writerow(header)
            for article, vector in zip(self.corpus, matrix):
                row = [self.class_to_index[article.category],
                       *vector]
                output_csv.writerow(row)
