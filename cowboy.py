import os
import logging
import pprint

from collections import defaultdict

import numpy
from matplotlib import pyplot
from sklearn.manifold import Isomap

import experiments


def setup_logger(name):
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', name + '.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    ch.setFormatter(logging.Formatter(
        "%(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ] %("
        "message)s"
    ))
    # add the handlers to the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
logger = setup_logger('cnn')

try:
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30
from sklearn import tree
import pydotplus

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'downloads')

#def cowboy():
    # data preprocessing
    # features = [[0, 0], [1, 1]]
    # classes = [0, 1]
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(features, classes)
    #
    # # data processing
    # data = [[2., 2.]]
    # actual_class = 1
    # predicted_class = int(clf.predict(data))
    #
    # print(predicted_class)
    #
    #
    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                                 feature_names=None,
    #                                 class_names=None,
    #                                 filled=True,
    #                                 rounded=True,
    #                                 special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf(os.path.join(DATA_DIR, 'decision_tree.pdf'))


import dataset
import preprocess
from scipy.spatial import distance
from processing import jaccard
import pickle

class Homework3Experiments(object):

    pickle_file_fmt = 'hw3.{}.pickle'

    def __init__(self, n, dataset_dir, pickling, randomize=True):
        # load data
        logger.info('Looking for datasets in {}'.format(dataset_dir))
        self.n = n
        self.dataset_dir = dataset_dir
        self.output_dir = os.path.join('figures', 'kmeans')
        os.makedirs(self.output_dir, exist_ok=True)

        self.pickling = pickling

        # preprocess
        self.vectorizers = {
            'tf': 'Term Frequency',
            # 'existence': 'Existence',
            'tfidf': 'TF-IDF'
        }

        corpus_pickle = 'corpus.{}'.format(n)
        corpus_by_vectorizer = self._load_pickle(corpus_pickle)
        if not corpus_by_vectorizer:
            self.articles = dataset.get(n=n, from_=dataset_dir,
                                        randomize=randomize)

            corpus_by_vectorizer = {
                self.vectorizers[k]: preprocess.execute(corpus=self.articles,
                                                        exclude_stopwords=True,
                                                        method=k)
                for k in self.vectorizers
            }
            self._save_to_pickel(corpus_by_vectorizer, corpus_pickle)
        self.corpus_by_vectorizer = corpus_by_vectorizer

        self.corpus = self.corpus_by_vectorizer['Term Frequency']

        self.experiment = {}


    def _load_pickle(self, filename):
        if not self.pickling:
            return False

        pickle_path = os.path.join(
            self.output_dir,
            self.pickle_file_fmt.format(filename))
        if os.path.exists(pickle_path):
            logger.info('Loading from pickle: {}'.format(pickle_path))
            pkl = pickle.load(open(pickle_path, 'rb'))
            return pkl

        return False

    def _save_to_pickel(self, object, filename):
        pickle_path = os.path.join(
            self.output_dir,
            self.pickle_file_fmt.format(filename))
        pickle.dump(object, open(pickle_path, 'wb'))


    def dimensionality_reduction(self):
        output_path = 'dimreduce'
        os.makedirs(output_path, exist_ok=True)

        logger.info(hr('Dimensionality Reduction', '+'))

        dataset = self.corpus.matrix.toarray()
        labels = self.corpus.classes

        # check if pickle of transformed data exists
        pkl_filename = 'dim_reduction_{}'.format(self.n)
        reduced = self._load_pickle(pkl_filename)

        if not reduced:
            # map the dataset to 2 dimensions
            lower_dimension_mapper = Isomap(n_neighbors=5,
                                            n_components=2)
            lower_dimension_mapper.fit(X=dataset, y=labels)
            reduced = lower_dimension_mapper.transform(X=dataset)
            self._save_to_pickel(reduced, pkl_filename)

        # quick printing of reduced dataset
        logger.debug('Transformed dataset:')
        logger.debug(reduced)
        logger.debug('X[0]: {}'.format(reduced[0]))
        logger.debug('y:    {}'.format(reduced[1]))

        # plot the reduced dimension
        # create a color mapping of the labels
        #colors_mapping = defaultdict(lambda: next(experiments.colors))
        # colors_mapping = defaultdict(lambda: numpy.random.rand(3,1))
        colors_available = experiments.get_cmap(len(self.corpus.class_names))
        colors = numpy.array([colors_available(cls) for cls in labels])

        pyplot.scatter(x=reduced[:, 0], y=reduced[:, 1],
                       c=colors)

        pyplot.show()

    def archive(self):
        # news articles
        # data matrix
        # classification results
        return

    def plot(self):
        return


def main():
    experiments = Homework3Experiments(
        n=10,
        dataset_dir=DATA_DIR,
        pickling=False)
    experiments.dimensionality_reduction()
    experiments.archive()
    experiments.plot()


if __name__ == '__main__':
    main()
