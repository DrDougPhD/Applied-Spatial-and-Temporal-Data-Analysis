import os
import logging
import pprint

from collections import defaultdict


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
from experiments import tree
from experiments.tree import plot as tree_plots
from experiments import knn
import experiments.knn.utils as knn_utils
from experiments.knn import plot
from scipy.spatial import distance
from processing import jaccard
import pickle

class Homework2Experiments(object):
    distances = [
        distance.jaccard,
        'euclidean',
         #'manhattan',
         #'chebyshev',
         #'hamming',
         #'canberra',
         #'braycurtis',
         distance.cosine
    ]

    pickle_file_fmt = 'hw2.{}.pickle'

    def __init__(self, n, dataset_dir, pickling, randomize=True):
        # load data
        logger.info('Looking for datasets in {}'.format(dataset_dir))
        self.n = n
        self.dataset_dir = dataset_dir
        self.knn_vote_weight = None
        self.output_dir = os.path.join('results')
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
                                        randomize=randomize,
                                        archive_to=os.path.join(
                                            self.output_dir,
                                            'articles'
                                        ))

            corpus_by_vectorizer = {
                self.vectorizers[k]: preprocess.execute(
                    corpus=self.articles,
                    exclude_stopwords=True,
                    method=k,
                    save_csv_to=self.output_dir)
                for k in self.vectorizers
            }
            self._save_to_pickel(corpus_by_vectorizer, corpus_pickle)
        self.corpus_by_vectorizer = corpus_by_vectorizer

        self.corpus = self.corpus_by_vectorizer['Term Frequency']

        self.experiment = {}

    def run(self, knn_neighbors_max, dec_tree_max_leafs):
        logger.info(hr('Beginning Experiments'))
        self.decision_tree(max_leafs=dec_tree_max_leafs)
        self.knn_vote_weight = 'uniform'
        self.knn(max_neighbors=knn_neighbors_max)
        self.knn_vote_weight = 'distance'
        self.knn(max_neighbors=knn_neighbors_max)

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


    def decision_tree(self, max_leafs):
        output_path = os.path.join(self.output_dir, 'decision_tree')
        os.makedirs(output_path, exist_ok=True)

        logger.info(hr('Decision Tree', '+'))
        experiment = self.experiment['tree'] = tree.experiment.Experiment(
            cross_validation_n=5,
            corpus_series=self.corpus_by_vectorizer,
            save_to=output_path,
            criterion_options=['gini', 'entropy'],
            mrmr_file=os.path.join('data', 'mrmr.features.txt'))

        # Which method, gini or entropy, produces the most accurate results?
        # What is the precision, recall, and f-measure of these experiments?

        prec_n_rec_pkl_filename = 'dectree_{}'.format(self.n)
        prec_n_rec_results = self._load_pickle(prec_n_rec_pkl_filename)
        if not prec_n_rec_results:
            prec_n_rec_results = {}
            for vector_type in self.corpus_by_vectorizer:
                results_for_matrix_type = {}
                for criterion in experiment.criterion_options:
                    exp_results = \
                        experiment.criterion_based_accuracy(
                        criterion=criterion, vector_type=vector_type)
                    results_for_matrix_type[criterion] = exp_results
                prec_n_rec_results[vector_type] = results_for_matrix_type
            self._save_to_pickel(prec_n_rec_results, prec_n_rec_pkl_filename)

        tree_plots.prec_n_rec(prec_n_rec_results,
                              class_labels=self.corpus.class_names,
                              save_to=output_path)

        # Which articles were harder to classify?
        decision_path_pkl_filename = 'decpaths_{}'.format(self.n)
        decision_paths = self._load_pickle(decision_path_pkl_filename)
        logger.info(hr('Decision Path Lengths'))
        if not decision_paths:
            decision_paths = experiment.decision_path_lengths(
                classnames=self.corpus.class_names,
                save_to=output_path)
            self._save_to_pickel(decision_paths, decision_path_pkl_filename)
        #logger.debug(hr('Decision Length Experiments Complete'))
        #logger.debug('Results:')
        #logger.debug(pprint.pformat(decision_paths))
        tree_plots.path_lengths(decision_paths, save_to=output_path)

    def knn(self, max_neighbors):
        output_path = os.path.join(self.output_dir, 'knn', self.knn_vote_weight)
        os.makedirs(output_path, exist_ok=True)

        logger.info(hr('k-Nearest Neighbors', '+'))
        filename = 'neighbors.knn.{0}of{1}'.format(
            max_neighbors, self.n)
        neighbors = self._load_pickle(filename)
        if not neighbors:
            # knn.run(k_neighbors=5, k_fold=5, corpus=self.corpus,
            #         distance_fn=distance.cosine, vote_weights=knn.inverse_squared)
            experiment = self.experiment['knn'] = knn.experiment.Experiment(
                cross_validation_n=5,
                vote_weight=self.knn_vote_weight,
                corpus_series=self.corpus_by_vectorizer,
                save_to=output_path)

            neighbors = defaultdict(dict)
            for corpus_key in self.vectorizers:
                selected_corpus_type = self.vectorizers[corpus_key]
                logger.info(hr('{0} article matrices'.format(selected_corpus_type),
                            '~'))

                neighbors_for_vector_type = neighbors[corpus_key]
                for distance_fn in Homework2Experiments.distances:
                    if isinstance(distance_fn, str):
                        distance_key = distance_fn
                    else:
                        distance_key = distance_fn.__name__

                    if distance_key not in neighbors_for_vector_type:
                        neighbors_for_vector_type[distance_key] = {}

                    logger.info(hr('{0} distances'.format(distance_fn), "."))
                    neighbors_from_exp = experiment.run(
                        xvals=range(1, max_neighbors+1),
                        series=selected_corpus_type,
                        variation=distance_fn)

                    neighbors_for_vector_type[distance_key] = neighbors_from_exp
                    print('#'*80)
                    print(neighbors_for_vector_type.keys())
                neighbors[corpus_key].update(neighbors_for_vector_type)
                print('#' * 100)
                print(neighbors.keys())
            plot.draw_accuracies(self.experiment['knn'], save_to=output_path)
            plot.draw_fmeasures(self.experiment['knn'],
               [('cosine', 'Term Frequency'), ('euclidean', 'TF-IDF')],
               save_to=output_path)
            self._save_to_pickel(neighbors, filename)

        plot.neighbor_heatmap(neighbors,
                              feature_names=self.corpus.features,
                              save_to=output_path)

        plot.neighborhood_radii(neighbors, save_to=output_path)
        experiment.export_neighbor_file()

    def archive(self):
        # news articles
        # data matrix
        # classification results
        return

    def plot(self):
        return


def main():
    configuration = {
        'n': 100,
        'k': 10,
        'dectree_max_leafs': 10,
        'enable_pickling': False,
    }
    experiments = Homework2Experiments(
        n=configuration['n'],
        dataset_dir=DATA_DIR,
        pickling=configuration['enable_pickling'])
    experiments.run(knn_neighbors_max=configuration['k'],
                    dec_tree_max_leafs=configuration['dectree_max_leafs'])
    # experiments.archive()
    # experiments.plot()


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