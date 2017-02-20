import logging
logger = logging.getLogger('cnn.'+__name__)
from sklearn.neighbors import KNeighborsClassifier
import numpy
import processing
import itertools

class LoggingObject(object):
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)

    def debug(self, msg):
        logger.debug(msg)

    def info(self, msg):
        logger.info(msg)

from collections import defaultdict
class Experiment(LoggingObject):
    def __init__(self, cross_validation_n, vote_weight, corpus_series):
        super(Experiment, self).__init__('cnn.' + __name__)
        print('cnn.' + __name__)
        self.n_fold = cross_validation_n
        self.voting_weight = vote_weight

        self.series = set()
        self.variations = set()
        self.results = {}
        self.precision_and_recalls = {}


        # corpus datasets are keyed by the vectorizer used - e.g. term
        # frequency, existence, or tfidf
        self.corpus = corpus_series

    def run(self, xvals, series, variation):
        self.series.add(series)

        variation_label = self._str_or_fn_name(variation)
        self.variations.add(variation_label)

        if series not in self.results:
            variations = {}
            self.results[series] = variations

        average_accuracies = []
        for x in xvals:
            accuracies, prec_recall = self.run_single(x, series, variation)
            average_accuracies.append(accuracies)

        self.info('Experiments finished')
        result = ExperimentResults(xvals=numpy.asarray(xvals),
                                   yvals=average_accuracies,
                                   label=variation_label)

        self.results[series][variation_label] = result

    def _str_or_fn_name(self, item):
        if not isinstance(item, str):
            if hasattr(item, '__name__'):
                return item.__name__
            else:
                raise ValueError('Label has no name: {}'.format(item))
        else:
            return item

    def run_single(self, x, series, variation):
        self.info('k = {0}, series = {1}, variation = {2}'.format(
            x, series, variation
        ))
        accuracies = []
        prec_and_rec = PrecisionAndRecall()

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

                # Record if this was a true positive or a false negative
                # for this class.
                prec_and_rec.record(str(label), str(int(predicted)))

            accuracies.append(successes / len(testing.classes))
            # accuracy = clf.score(testing.matrix, testing.classes)

        average_accuracy = numpy.mean(accuracies)

        logger.info('Accuracy: {0} -- {1}'.format(average_accuracy,
                                                  accuracies))
        logger.info('Precision and Recall:\n{}'.format(prec_and_rec))
        logger.debug('-' * 120)
        return average_accuracy, prec_and_rec

    def get_results_for(self, series, variation):
        return self.results[series][variation]


class ExperimentResults(LoggingObject):
    def __init__(self, xvals, yvals, label):
        self.x = xvals
        self.y = yvals
        self.label = label


class PrecisionAndRecall(LoggingObject):
    def __init__(self):
        int_defdict = lambda: defaultdict(int)
        self.data = defaultdict(int_defdict)
        self.observations = []

    def record(self, actual_class, predicted_class):
        class_data = self.data[actual_class]
        if predicted_class not in class_data:
            class_data[predicted_class] = 0

        class_data[predicted_class] += 1
        self.observations.append((int(actual_class), int(predicted_class)))

    def true_positives(self, for_):
        return self.data[for_][for_]

    def false_positives(self, for_):
        key_data = self.data[for_]
        fpositives = 0
        for key in key_data:
            if key == for_:
                continue
            fpositives += key_data[key]
        return fpositives

    def true_negatives(self, for_):
        tns = 0
        # for each class that is not this class...
        other_classes = set(self.data.keys()) - set(for_)
        logger.debug('True Negatives for {0}:'.format(for_))
        for i in list(other_classes):
            for j in list(other_classes):
                logger.debug('{0} to {1}'.format(i, j))
                tns += self.data[i][j]

        """
        for key in self.data:
            if key == for_:
                continue

            # count up the number of times that item was not
            # classed as this class
            tns += self.true_positives(for_=key)
        """

        return tns

    def false_negatives(self, for_):
        fnegs = 0
        for key in self.data:
            fnegs += self.data[key][for_]
        return fnegs

    def precision(self, for_):
        tp = self.true_positives(for_)
        fp = self.false_positives(for_)
        if tp == 0:
            return 0

        return tp / (tp+fp)

    def recall(self, for_):
        tp = self.true_positives(for_)
        fn = self.false_negatives(for_)
        if tp == 0:
            return 0

        return tp / (tp+fn)

    def fmeasure(self, for_):
        p = self.precision(for_)
        r = self.recall(for_)
        if p == 0 or r == 0:
            return 0

        f = 2 * (p*r)/(p+r)
        return f

    def __str__(self):
        self.observations.sort(key=lambda x: x[0])
        output_lines = []
        max_key_length = len(max(self.data, key=len))
        for key in sorted(self.data.keys()):
            output_lines.append(
                '{key}:\t True Positives: {tp}\n'
                '{pad} \tFalse Positives: {fp}\n'
                '{pad} \t True Negatives: {tn}\n'
                '{pad} \tFalse Negatives: {fn}\n'
                '{pad} \tPrecision:       {p}\n'
                '{pad} \tRecall:          {r}\n'
                '{pad} \tF-Measure:       {fm}\n'
                '-----------------------------------------'.format(
                    key=key.rjust(max_key_length+2),
                    pad=''.rjust(max_key_length+2),
                    tp=self.true_positives(for_=key),
                    fp=self.false_positives(for_=key),
                    tn=self.true_negatives(for_=key),
                    fn=self.false_negatives(for_=key),
                    fm=self.fmeasure(for_=key),
                    p=self.precision(for_=key),
                    r=self.recall(for_=key)
                )
            )

        output_lines.append('\n'.join([
            str(o) for o in self.observations
        ]))

        return '\n'.join(output_lines)
