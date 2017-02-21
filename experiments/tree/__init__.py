from . import experiment

import logging
logger = logging.getLogger('cnn.'+__name__)
from sklearn import tree
import pydotplus
import os
import numpy
import processing

def run(k, corpus, save_to):
    accuracies = []
    partitioner = processing.CrossValidation(k=k, dataset=corpus)
    for i, (training, testing) in enumerate(partitioner):
        logger.info('Training Decision Tree')
        clf = tree.DecisionTreeClassifier()
        clf.fit(training.matrix, training.classes)

        logger.info('Predicting scores')
        accuracy = clf.score(testing.matrix, testing.classes)
        accuracies.append(accuracy)


        export(clf, feature_names=list(corpus.features),
               class_names=list(corpus.class_names),
               save_to=save_to, index=i)


    average_accuracy = numpy.mean(accuracies)

    logger.info('Accuracy: {0} -- {1}'.format(average_accuracy, accuracies))
    logger.debug('-'*120)


def export(clf, feature_names, class_names, save_to, index):
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.path.join(save_to,
                                 '{}.decision_tree.pdf'.format(index)))

# TODO: Save decision trees for visualization?
# Make pretty plots.