#from experiments import LoggingObject
from collections import defaultdict
from sklearn import tree
import pydotplus
import os
import processing
import numpy
#from experiments import PrecisionAndRecalls
from sklearn import metrics

from lib.lineheaderpadded import hr

import logging
logger = logging.getLogger('cnn.'+__name__)
import pprint

class Experiment(object):
    def __init__(self, cross_validation_n, corpus_series, criterion_options,
                 save_to):
        super(Experiment, self).__init__()
        self.n = cross_validation_n
        self.datasets = corpus_series
        self.save_to = save_to
        self.criterion_options = criterion_options

    def export(self, clf, feature_names, class_names, name):
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=feature_names,
                                        class_names=class_names,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        tree.export_graphviz(clf, out_file=os.path.join(
                                      self.save_to,
                                      'decision_tree.{}.dot'.format(name)),
                             feature_names=feature_names,
                             class_names=class_names,
                             filled=True,
                             rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(os.path.join(self.save_to,
                                     'decision_tree.{}.pdf'.format(name)))

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
                        name='{0}.{1}.crossfold_{2}'.format(
                            vector_type, criterion, i+1
                        ))

        average_accuracy = numpy.mean(accuracies)
        logger.info('Accuracy: {0} -- {1}'.format(average_accuracy,
                                                accuracies))
        logger.debug('-' * 120)
        predicted_labels = numpy.array(predicted_labels)
        actual_labels = numpy.array(actual_labels)
        return {'predicted': predicted_labels, 'actual': actual_labels}

    def decision_path_lengths(self, classnames, save_to):
        logger.info(hr('Decision Paths', '+'))
        results = defaultdict(dict)
        classification_results = []
        archive_file = {}

        for vector_type in self.datasets:
            archive_file[vector_type] = {}
            logger.info(hr(vector_type, '~'))

            results_for_vector_type = defaultdict(dict)
            dataset = self.datasets[vector_type]
            for criterion in self.criterion_options:

                tree_path_file = open(os.path.join(
                    save_to, 'decision_tree.{0}'
                             '.{1}.article_classification_path.csv'.format(
                        vector_type, criterion
                    )), 'w'
                )
                archive_file[vector_type][criterion] = tree_path_file
                tree_path_file.write(
                    'article,category,predicted,tree_path\n')

                logger.info(hr(criterion, '.'))

                partitioner = processing.CrossValidation(k=self.n,
                                                         dataset=dataset)
                path_lengths_by_label = defaultdict(list)
                logger.info(hr('Decision Tree Testing', '#'))
                for i, (training, testing) in enumerate(partitioner):
                    archive_file[vector_type][criterion].write(
                        '# cross fold no. {}\n'.format(i+1))
                    clf = tree.DecisionTreeClassifier()
                    clf.fit(training.matrix, training.classes)

                    for vector, label, article in testing:
                        path = clf.decision_path(vector)
                        predicted = clf.predict(vector)
                        archive_file[vector_type][criterion].write(
                            '{article_no},{category},{predicted},'
                            '{tree_path}\n'.format(
                                article_no=article.filename,
                                category=label,
                                predicted=int(predicted),
                                tree_path=_convert_tree_path_to_list(path),
                            )
                        )

                        path_length = numpy.sum(path.toarray())
                        path_lengths_by_label[classnames[label]].append(
                            path_length)

                    # cross-fold break
                    classification_results.append(
                        (vector_type, criterion, None))

                for k in path_lengths_by_label:
                    results_by_class_label = results[k]
                    if vector_type not in results_by_class_label:
                        results_by_class_label[vector_type] = defaultdict(dict)

                    results_by_class_label[vector_type][criterion] = \
                        path_lengths_by_label[k]

                    path_lengths = path_lengths_by_label[k]
                    logger.debug('Path statistics for {}'.format(k))
                    logger.debug('Min path length: {}'.format(min(path_lengths)))
                    logger.debug('Max path length: {}'.format(max(path_lengths)))
                    logger.debug('Avg path length: {}'.format(numpy.mean(path_lengths)))

        # save to file of format:
        # article_no article_class_no predicted_class_no decision_tree_path
        # for classification_result in classification_results:
        #     output_file = archive_file[classification_result[0]]\
        #                               [classification_result[1]]
        #     if classification_result[2] is None:
        #
        #
        # with open(os.path.join(
        #           save_to, 'decision_tree_predicted_class_and_tree_path')) as f:


        return results



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

def _convert_tree_path_to_list(path):
    logger.debug(hr('Converting path to ->', '.'))
    adjacency_list = path.toarray()[0]
    logger.debug(adjacency_list)
    pretty_list = []
    for i, visited in enumerate(adjacency_list):
        if visited == 1:
            pretty_list.append(str(i))
    pretty_path = '->'.join(pretty_list)
    logger.debug(pretty_path)
    return pretty_path
