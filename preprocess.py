import numpy
from progressbar import ProgressBar
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import logging

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
    logger.debug(vectorizer)
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

        # isolate terms that will be preserved
        irrelevant_features = self._load_mrmr(mrmr, plain_text)
        terms_to_remove = stopwords.union(irrelevant_features)

        # determine the unique categories represented in the dataset
        class_to_index = {k: i for i, k in enumerate(self.class_names)}
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


# progress = ProgressBar(
#     max_value=n)
# progress.update(i)
# progress.finish()

    def _load_mrmr(self, mrmr, corpus):
        if mrmr is None:
            return []

        logger.debug('Loading mRMR features from "{}"'.format(mrmr))
        # load good features from mrmr file
        mrmr_good_features = set()
        with open(mrmr) as f:
            for line in f:
                columns = line.split()
                mrmr_good_features.add(columns[1])
        logger.debug('mRMR relevant features: {}'.format(mrmr_good_features))

        # do a quick pass over the corpus data to find all unique features
        # contained within
        unique_features = set()
        for article in corpus:
            unique_features.union(article)

        # remove the good features, resulting in the irrelevant features
        # being left over
        irrelevant_features = unique_features - mrmr_good_features

        return irrelevant_features