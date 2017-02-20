import logging
logger = logging.getLogger('cnn.'+__name__)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy
import processing

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

        # corpus datasets are keyed by the vectorizer used - e.g. term
        # frequency, existence, or tfidf
        self.corpus = corpus_series
        sample_corpus_key = list(corpus_series.keys())[0]
        sample_corpus = corpus_series[sample_corpus_key]
        self.classnames = sample_corpus.class_names

    def run(self, xvals, series, variation):
        self.series.add(series)

        variation_label = self._str_or_fn_name(variation)
        self.variations.add(variation_label)

        if series not in self.results:
            variations = {}
            self.results[series] = variations

        average_accuracies = []
        precision_and_recalls = []
        for x in xvals:
            accuracies, prec_recall = self.run_single(x, series, variation)
            average_accuracies.append(accuracies)
            precision_and_recalls.append(prec_recall)

        self.info('Experiments finished')
        result = ExperimentResults(xvals=numpy.asarray(xvals),
                                   yvals=average_accuracies,
                                   label=variation_label,
                                   precision_and_recalls=precision_and_recalls)

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
        self.debug(','*40)
        self.info('k = {0}, series = {1}, variation = {2}'.format(
            x, series, variation
        ))
        accuracies = []
        dataset = self.corpus[series]
        partitioner = processing.CrossValidation(
            k=self.n_fold,
            dataset=dataset)

        actual_labels = []
        predicted_labels = []
        neighbors = []

        for training, testing in partitioner:
            self.info('Training KNN Model')
            clf = KNeighborsClassifier(n_neighbors=x,
                                       algorithm='brute',
                                       metric=variation,
                                       weights=self.voting_weight)
            clf.fit(training.matrix.toarray(), training.classes)

            logger.info('Predicting scores')
            successes = 0

            for m, label, article in testing:
                # logger.debug('-'*80)
                # logger.debug('Testing matrix:')
                # logger.debug(m)
                # logger.debug(type(m))

                predicted = int(clf.predict(m))
                if predicted == label:
                    successes += 1

                # Record if this was a true positive or a false negative
                # for this class.
                actual_labels.append(int(label))
                predicted_labels.append(int(predicted))

                # Let's look at what are the nearest neighbors of this guy
                distances, indices = clf.kneighbors(m)
                distances = distances[0]
                indices = indices[0]

                # logger.debug('Distances:')
                # logger.debug(distances)
                # logger.debug('Indices:')
                # logger.debug(indices)
                # logger.debug('-' * 30)
                # logger.debug('Neighbors of:{0: >15} -- "{1}"'.format(
                #     article.category, article.title,
                # ))

                predicted_class_name = partitioner.classnames[predicted]
                article_neighbors = []
                neighbor_entry = {
                    'article': article,
                    'predicted_class': predicted_class_name,
                    'neighbors': article_neighbors,
                }

                for d, i in zip(distances, indices):
                    # logger.debug('Distance: {}'.format(d))
                    # logger.debug('Index:    {}'.format(i))
                    d = float(d)
                    i = int(i)
                    neighbor = training.get_article(i)
                    # logger.debug('{0:.9f}: {1: >15} -- "{2}"'.format(
                    #     d, neighbor.category, neighbor.title
                    # ))
                    article_neighbors.append({
                        'neighbor': neighbor,
                        'distance': d,
                    })

                closest_neighbor_tuple = min(article_neighbors,
                                             key=lambda x: x['distance'])
                summed_neighbor_distances = sum([
                     art['distance'] for art in article_neighbors])

                neighbor_entry['closest_distance']\
                    = closest_neighbor_tuple['distance']
                neighbor_entry['summed_distances']\
                    = summed_neighbor_distances
                neighbors.append(neighbor_entry)

                # logger.debug('  Predicted: {: >15}'.format(
                # predicted_class_name))

            accuracies.append(successes / len(testing.classes))
            # accuracy = clf.score(testing.matrix, testing.classes)


        # print some nice strings about the neighbors
        neighbors.sort(key=lambda x: x['closest_distance'], reverse=True)
        neighbor_string_info = []
        for article_neighborinos in neighbors:
            printable_string = self.stringify_neighbor_info(article_neighborinos)
            neighbor_string_info.append(printable_string)

        logger.debug('\n'.join(neighbor_string_info))
        logger.debug('='*80)

        # analyze precision and recall
        labels = list(range(len(dataset.class_names)))
        # prec, rec, fscore, support = metrics.precision_recall_fscore_support(
        #     y_true=actual_labels,
        #     y_pred=predicted_labels,
        #     labels=labels,
        #     average=None,
        # )

        prec_n_rec = PrecisionAndRecalls(truth=actual_labels,
                                         predictions=predicted_labels,
                                         available_labels=labels,
                                         label_names=dataset.class_names)

        # print_fmt = '{l: >16}: {p:.4f} -- {r:.4f} -- {s:.4f} -- {f:.4f}'
        # print(
        #     '{label: >16}: {prec: ^6} -- {rec: ^6} -- {support: ^6} -- {fscore: ^6}\n'.format(
        #         label='Label',
        #         prec='Prec',
        #         rec='Recall',
        #         support='Supp',
        #         fscore='FScore'))
        # for p, r, f, s, l in zip(prec, rec, fscore, support, labels):
        #     print(print_fmt.format(**locals()))

        average_accuracy = numpy.mean(accuracies)

        logger.info('Accuracy: {0} -- {1}'.format(average_accuracy,
                                                  accuracies))
        logger.info('Precision and Recall: {}'.format(
            prec_n_rec)) #prec_and_rec.fmeasure()))
        logger.debug('-' * 120)
        return average_accuracy, None

    def get_results_for(self, series, variation):
        return self.results[series][variation]

    def stringify_neighbor_info(self, neighbor_dict):
        lines = ['-'*80]

        article = neighbor_dict['article']
        lines.append('Neighbors of:{0: >15} -- "{1}"'.format(
            article.category, article.title,
        ))

        neighbors = neighbor_dict['neighbors']
        for neighborino in neighbors:
            neighbor = neighborino['neighbor']
            d = neighborino['distance']
            lines.append('{0:.9f}: {1: >15} -- "{2}"'.format(
                d, neighbor.category, neighbor.title
            ))

        predicted_class_name = neighbor_dict['predicted_class']
        lines.append('  Predicted: {: >15}'.format(
            predicted_class_name))
        return '\n'.join(lines)





class ExperimentResults(LoggingObject):
    def __init__(self, xvals, yvals, label, precision_and_recalls):
        self.x = xvals
        self.y = yvals
        self.label = label
        self.pnc = precision_and_recalls


class PrecisionAndRecalls(LoggingObject):
    string_fmt = '{l: >16}: {p:.4f} -- {r:.4f} -- {s:.4f} -- {f:.4f}'
    def __init__(self, truth, predictions, available_labels,
                 label_names):
        super(PrecisionAndRecalls, self).__init__()
        self.truth = truth
        self.predictions = predictions
        self.label_indices = available_labels
        self.label_names = label_names

        self.precisions, self.recalls, self.fscores, self.supports\
            = metrics.precision_recall_fscore_support(
                y_true=truth,
                y_pred=predictions,
                labels=available_labels,
                average=None,
            )

    def __iter__(self):
        for tup in zip(self.precisions, self.recalls, self.fscores,
                       self.supports, self.label_names):
            yield tup

    def __str__(self):
        lines = [
            ('\n{label: >16}: {prec: ^6} -- {rec: ^6} --'
             ' {support: ^6} -- {fscore: ^6}\n'.format(
                label='Label',
                prec='Prec',
                rec='Recall',
                support='Supp',
                fscore='FScore'))]
        for p, r, f, s, l in self:
            lines.append(PrecisionAndRecalls.string_fmt.format(**locals()))

        return '\n'.join(lines)
