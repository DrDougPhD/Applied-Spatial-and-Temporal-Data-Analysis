#from experiments import LoggingObject
from sklearn import tree
import pydotplus
import os
import processing
import numpy
#from experiments import PrecisionAndRecalls
from sklearn import metrics

import logging
logger = logging.getLogger('cnn.'+__name__)

class Experiment(object):
    def __init__(self, cross_validation_n, corpus_series, save_to):
        super(Experiment, self).__init__()
        self.n = cross_validation_n
        self.datasets = corpus_series
        self.save_to = save_to

    def export(self, clf, feature_names, class_names, name):
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=feature_names,
                                        class_names=class_names,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(os.path.join(self.save_to,
                                     '{}.decision_tree.pdf'.format(name)))

    def accuracy(self, series, vector_type, splitting_criterion, x):
        return 4

    def criterion_based_accuracy(self, criterion, vector_type):
        accuracies = []
        predicted_labels = []
        actual_labels = []
        dataset = self.datasets[vector_type]
        partitioner = processing.CrossValidation(k=self.n,
            dataset=dataset)
        for i, (training, testing) in enumerate(partitioner):
            logger.info('Training Decision Tree')
            clf = tree.DecisionTreeClassifier()
            clf.fit(training.matrix, training.classes)

            logger.info('Predicting scores')
            accuracy = clf.score(testing.matrix, testing.classes)
            accuracies.append(accuracy)

            for vector, true_label in zip(testing.matrix, testing.classes):
                predicted_label = clf.predict(vector)
                predicted_labels.append(int(predicted_label))
                actual_labels.append(true_label)

            self.export(clf, feature_names=list(dataset.features),
                        class_names=list(dataset.class_names),
                        name=vector_type)

        average_accuracy = numpy.mean(accuracies)
        logger.info('Accuracy: {0} -- {1}'.format(average_accuracy,
                                                accuracies))
        logger.debug('-' * 120)
        predicted_labels = numpy.array(predicted_labels)
        actual_labels = numpy.array(actual_labels)
        return {'predicted': predicted_labels, 'actual': actual_labels}


# class ExperimentPerformance(LoggingObject):
#     def __init__(self, predicted_labels, actual_labels, classnames):
#         super(ExperimentPerformance, self).__init__()
#         self.class_names = classnames
#
#         vals = metrics.precision_recall_fscore_support(
#             y_true=actual_labels,
#             y_pred=predicted_labels,
#             labels=range(len(classnames)),
#             average=None,
#         )
#         self.precs, self.recs, self.fscores, self.supports = vals
#
#         self.data = {}
#         for k, p, r, f, s in zip(self.class_names, *vals):
#             self.data[k] = {
#                 'precision': p,
#                 'recall': r,
#                 'f-score': f,
#                 'support': s,
#             }
#
#         overall = metrics.precision_recall_fscore_support(
#             y_true=actual_labels,
#             y_pred=predicted_labels,
#             labels=range(len(classnames)),
#             average='micro',
#         )
#         p, r, f, s = overall
#         self.data['overall'] = {
#             'precision': p,
#             'recall': r,
#             'f-score': f,
#             'support': s,
#         }
#
#         # {
#         #     k: {
#         #         metric: random.random() for metric in ['accuracy',
#         #                                                'precision',
#         #                                                'recall',
#         #                                                'f-score']
#         #     } for k in self.class_names
#         # }
#
