import os
import logging
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

def cowboy():
    # data preprocessing
    features = [[0, 0], [1, 1]]
    classes = [0, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, classes)

    # data processing
    data = [[2., 2.]]
    actual_class = 1
    predicted_class = int(clf.predict(data))

    print(predicted_class)


    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=None,
                                    class_names=None,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.path.join(DATA_DIR, 'decision_tree.pdf'))


import dataset
import preprocess
from experiments import tree
from experiments import knn
import experiments.knn.utils as knn_utils
from experiments.knn import plot
from scipy.spatial import distance
from processing import jaccard
import pickle

class Homework2Experiments(object):
    distances = ['euclidean',
                 #'manhattan',
                 #'chebyshev',
                 #'hamming',
                 #'canberra',
                 #'braycurtis',
                 distance.cosine,
                 distance.jaccard]

    pickle_file_fmt = 'hw2.n{n}.pickle'

    def __init__(self, n, dataset_dir, randomize=True, method='tf'):
        # load data
        logger.info('Looking for datasets in {}'.format(dataset_dir))
        self.n = n
        self.dataset_dir = dataset_dir

        self.articles = dataset.get(n=n, from_=dataset_dir,
                                    randomize=randomize)

        # preprocess
        self.vectorizers = {
            'tf': 'Term Frequency',
            #'existence': 'Existence',
            'tfidf': 'TF-IDF'
        }
        self.corpus_by_vectorizer = {
            self.vectorizers[k]: preprocess.execute(corpus=self.articles,
                                               exclude_stopwords=True,
                                               method=k)
            for k in self.vectorizers
        }
        self.corpus = self.corpus_by_vectorizer['Term Frequency']

    def run(self, knn_neighbors_max):
        logger.info(hr('Beginning Experiments'))
        self.decision_tree(save_to='figures')
        self.knn(max_neighbors=knn_neighbors_max)

    """
    def _load_pickle(self):
        pickle_path = os.path.join(
            self.dataset_dir,
            self.pickle_file_fmt.format(n=self.n))
        if os.path.exists(pickle_path):
            logger.info('Loading from pickle: {}'.format(pickle_path))
            pkl = pickle.load(open(pickle_path, 'rb'))
            self.experiment = pkl
            return True

        return False

    def _save_to_pickel(self):
        pickle_path = os.path.join(
            self.dataset_dir,
            self.pickle_file_fmt.format(n=self.n))
        pickle.dump(self.experiment,
                    open(pickle_path, 'wb'))
    """

    def decision_tree(self, save_to):
        logger.info(hr('Decision Tree', '+'))
        tree.run(k=5, corpus=self.corpus, save_to=save_to)

    def knn(self, max_neighbors):
        logger.info(hr('k-Nearest Neighbors', '+'))
        # knn.run(k_neighbors=5, k_fold=5, corpus=self.corpus,
        #         distance_fn=distance.cosine, vote_weights=knn.inverse_squared)
        experiment = self.experiment = knn.experiment.Experiment(
            cross_validation_n=5,
            vote_weight='uniform',
            corpus_series=self.corpus_by_vectorizer)

        for corpus_key in self.vectorizers:
            selected_corpus_type = self.vectorizers[corpus_key]
            logger.info(hr('{0} article matrices'.format(selected_corpus_type),
                        '~'))
            for distance_fn in Homework2Experiments.distances:
                logger.info(hr('{0} distances'.format(distance_fn), "."))
                experiment.run(xvals=range(1, max_neighbors+1),
                               series=selected_corpus_type,
                               variation=distance_fn)

    def archive(self):
        # news articles
        # data matrix
        # classification results
        pass

    def plot(self):
        os.makedirs('figures/uniform/', exist_ok=True)
        plot.draw_accuracies(self.experiment, save_to='figures/uniform/')
        plot.draw_fmeasures(self.experiment,
            [('cosine', 'Term Frequency'), ('jaccard', 'TF-IDF')],
            save_to='figures')


def main(n=20):
    experiments = Homework2Experiments(n=n, dataset_dir=DATA_DIR)
    experiments.run(knn_neighbors_max=3)
    experiments.archive()
    experiments.plot()



if __name__ == '__main__':
    main()

"""

def experiment_round():
    partitioner = CrossValidation(k=5, dataset=corpus)
    for training, testing in partitioner:
        clf.fit(training.matrix, training.classes)
        predicted_classes = clf.predict(testing.matrix)
        successes = (testing.classes == predicted_classes)
        num_correct = numpy.sum(successes)
        correct_counts.append(num_correct)

    average_accuracy = sum(correct_counts) / partitioner.part_size


def max_features_vs_accuracy_vs_impurity_measure():
    experiment = ...
    plot_accuracy_by_impurity_measure(experiment)


def max_depth_vs_accuracy_vs_impurity_measure():
    experiment = ...
    plot_accuracy_by_impurity_measure(experiment)


def plot_accuracy_by_impurity_measure(experiment):
    accuracy_data = {}
    fmeasure_data = {}
    for impurity in impurity_measures:
        accuracies = []
        for i in range(len(features)):
            results = experiment.run(i, impurity, repeat=10)

            accuracy = results.get_average_accuracy()
            accuracies.append(accuracy)

            fmeasure = results.get_fmeasure()
            fmeasures.append(fmeasure)

        accuracy_data[impurity] = { 'X': numpy.arange(len(features)),
                                    'Y': numpy.array(accuracies) }
        fmeasure_data[impurity] = { 'X': numpy.arange(len(features)),
                                    'Y': numpy.array(fmeasures) }

    plot(accurracies=accuracy_data, fmeasures=fmeasure_data)
"""