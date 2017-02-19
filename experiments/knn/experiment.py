import logging
logger = logging.getLogger('cnn.'+__name__)
from sklearn.neighbors import KNeighborsClassifier
import numpy
import processing

class LoggingObject(object):
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)

    def debug(self, msg):
        logger.debug(msg)

    def info(self, msg):
        logger.info(msg)


class Experiment(LoggingObject):
    def __init__(self, cross_validation_n, vote_weight, corpus_series):
        super(Experiment, self).__init__('cnn.' + __name__)
        print('cnn.' + __name__)
        self.n_fold = cross_validation_n
        self.voting_weight = vote_weight


        # corpus datasets are keyed by the vectorizer used - e.g. term
        # frequency, existence, or tfidf
        self.corpus = corpus_series

    def run(self, xvals, series, variation):
        for x in xvals:
            self.run_single(x, series, variation)

    def run_single(self, x, series, variation):
        self.info('k = {0}, series = {1}, variation = {2}'.format(
            x, series, variation
        ))
        accuracies = []
        partitioner = processing.CrossValidation(
            k=self.n_fold,
            dataset=self.corpus[series])
        for training, testing in partitioner:
            self.info('Training KNN Model')
            clf = KNeighborsClassifier(n_neighbors=x,
                                       algorithm='brute',
                                       metric=variation,
                                       weights=self.voting_weight)
            clf.fit(training.matrix.toarray(), training.classes)

            logger.info('Predicting scores')
            successes = 0
            for m, label in testing:
                # logger.debug('-'*80)
                # logger.debug('Testing matrix:')
                # logger.debug(m)
                # logger.debug(type(m))
                predicted = clf.predict(m)
                if predicted == label:
                    successes += 1

                accuracy = 0
            accuracies.append(successes / len(testing.classes))
            # accuracy = clf.score(testing.matrix, testing.classes)

        average_accuracy = numpy.mean(accuracies)

        logger.info('Accuracy: {0} -- {1}'.format(average_accuracy, accuracies))
        logger.debug('-' * 120)

    def get_results_for(self, series, variation):
        pass