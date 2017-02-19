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

    ch.setLevel(logging.INFO)

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

class Homework2Experiments(object):
    distances = ['euclidean',
                 'manhattan',
                 'chebyshev',
                 'hamming',
                 'canberra',
                 'braycurtis',
                 distance.cosine,
                 distance.jaccard]

    def __init__(self, n, dataset_dir, randomize=True, method='tf'):
        # load data
        logger.info('Looking for datasets in {}'.format(dataset_dir))
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

    def run(self):
        logger.info(hr('Beginning Experiments'))
        self.decision_tree()
        self.knn()

    def decision_tree(self):
        logger.info(hr('Decision Tree', '-'))
        tree.run(k=5, corpus=self.corpus)

    def knn(self):
        logger.info(hr('k-Nearest Neighbors', '-'))
        # knn.run(k_neighbors=5, k_fold=5, corpus=self.corpus,
        #         distance_fn=distance.cosine, vote_weights=knn.inverse_squared)
        experiment = knn.experiment.Experiment(
            cross_validation_n=5,
            vote_weight=knn_utils.inverse_squared,
            corpus_series=self.corpus_by_vectorizer)

        for corpus_key in self.vectorizers:
            selected_corpus_type = self.vectorizers[corpus_key]
            experiment.run(xvals=range(1, 3), series=selected_corpus_type,
                           variation=distance.cosine)
        plot.draw(experiment)

    def archive(self):
        # news articles
        # data matrix
        # classification results
        pass

    def plot(self):
        pass


def main(n=10):
    experiments = Homework2Experiments(n=n, dataset_dir=DATA_DIR)
    experiments.run()
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