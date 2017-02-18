from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import logging

from processing import PairwiseSimilarity

STOP_WORDS = ENGLISH_STOP_WORDS.union(
    'new says time just like told cnn according did make way really'
    ' dont going know said'.split())
logger = logging.getLogger('cnn.'+__name__)


def execute(corpus, exclude_stopwords, method):
    if exclude_stopwords:
        logger.info('No stopwords will be used')
        stopwords = frozenset([])
    else:
        logger.info('Using stopwords')
        stopwords = STOP_WORDS

    similarity_calculator = PairwiseSimilarity(corpus=corpus,
                                               method=method,
                                               stopwords=stopwords)

    return similarity_calculator
