import os

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import logging

from processing import PairwiseSimilarity

STOP_WORDS = ENGLISH_STOP_WORDS.union(
    'new says time just like told cnn according did make way really'
    ' dont going know said to before according her out might ago'
    ' subject had'.split())
logger = logging.getLogger('cnn.'+__name__)


def execute(corpus, exclude_stopwords, method, save_csv_to):
    if not exclude_stopwords:
        logger.info('No stopwords will be used')
        stopwords = frozenset([])
    else:
        logger.info('Using stopwords')
        stopwords = STOP_WORDS

    similarity_calculator = PairwiseSimilarity(corpus=corpus,
                                               method=method,
                                               stopwords=stopwords)
    # maximum_relevant_feature_indices = _load_mrmr(
    #     mrmr, similarity_calculator.features)
    # similarity_calculator.filter_features(maximum_relevant_feature_indices)

    to_csv(similarity_calculator, save_csv_to)
    return similarity_calculator

def to_csv(corpus, csv_directory):
    import csv
    output_filepath = os.path.join(csv_directory,
                                   'corpus.{0}.{1}.csv'.format(
                                       corpus.method,
                                       len(corpus.classes)
                                   ))
    with open(output_filepath, 'w') as f:
        output_csv = csv.writer(f)
        header = ['category', *corpus.features] #*range(len(corpus.features))]
        output_csv.writerow(header)
        for article in corpus.corpus:
            row = [corpus.class_to_index[article.category], *article.vector]
            output_csv.writerow(row)


def _load_mrmr(mrmr_file, features):
    mrmr_indices = []
    mrmr_names = []
    with open(mrmr_file) as f:
        for line in f:
            columns = line.split()
            mrmr_names.append(columns[1])
            mrmr_indices.append(features.index(columns[1]))


    # logger.debug('Selected features: {}'.format(mrmr_names))
    # logger.debug('Features by index: {}'.format(
    #    numpy.array(features, dtype=numpy.str_)[mrmr_indices]))
    for mrmr_feature in mrmr_names:
        assert mrmr_feature in features, (
            'Feature "{}" is not in the indexed version'.format(mrmr_feature)
        )
    return mrmr_indices