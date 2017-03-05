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
def preprocess(corpus, exclude_stopwords, method):
    if not exclude_stopwords:
        logger.info('No stopwords will be used')
        stopwords = frozenset([])
    else:
        logger.info('Using stopwords')
        stopwords = STOP_WORDS

    logger.debug('Vectorizing corpus')
    vectorizer = CorpusVectorizer(corpus=corpus,
                                  method=method,
                                  stopwords=stopwords)
    logger.debug(vectorizer)
    return vectorizer


class CorpusVectorizer(object):
    def __init__(self, corpus, method, stopwords):
        self.corpus = corpus
        self.method = method
        self.count = len(corpus)

        # specify method in which corpus is repr'd as matrix
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=1,
                                              stop_words=stopwords)
        else:
            # matrix will be converted to binary matrix further down
            self.vectorizer = CountVectorizer(min_df=1,
                                              stop_words=stopwords)

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

        # determine the unique categories represented in the dataset
        class_to_index = {k: i for i, k in enumerate(self.class_names)}
        self.classes = numpy.array([class_to_index[document.category]
                                    for document in corpus])

        logger.debug('Transforming articles into vector space')
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