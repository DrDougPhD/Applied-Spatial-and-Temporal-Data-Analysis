import logging
logger = logging.getLogger('cnn.'+__name__)
from sklearn import tree
import numpy
import processing

def run(k, corpus):
    accuracies = []
    partitioner = processing.CrossValidation(k=k, dataset=corpus)
    for training, testing in partitioner:
        logger.info('Training Decision Tree')
        clf = tree.DecisionTreeClassifier()
        clf.fit(training.matrix, training.classes)

        logger.info('Predicting scores')
        accuracy = clf.score(testing.matrix, testing.classes)
        accuracies.append(accuracy)
    average_accuracy = numpy.mean(accuracies)

    logger.info('Accuracy: {0} -- {1}'.format(average_accuracy, accuracies))
    logger.debug('-'*120)

# TODO: Save decision trees for visualization?
# Make pretty plots.