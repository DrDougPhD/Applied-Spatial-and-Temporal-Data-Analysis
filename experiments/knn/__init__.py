import logging
logger = logging.getLogger('cnn.'+__name__)
from sklearn.neighbors import KNeighborsClassifier
import numpy
import processing

def run(k_neighbors, k_fold, corpus, distance_fn, vote_weights):
    accuracies = []
    partitioner = processing.CrossValidation(k=k_fold, dataset=corpus)
    for training, testing in partitioner:
        logger.info('Training KNN Model')
        clf = KNeighborsClassifier(n_neighbors=k_neighbors,
                                   algorithm='brute',
                                   metric=distance_fn,
                                   weights=vote_weights)
        clf.fit(training.matrix.toarray(), training.classes)

        logger.info('Predicting scores')
        successes = 0
        for m, label in testing:
            logger.debug('-'*80)
            logger.debug('Testing matrix:')
            logger.debug(m)
            logger.debug(type(m))
            predicted = clf.predict(m)
            if predicted == label:
                successes += 1

            accuracy = 0
        accuracies.append(successes / len(testing.classes))
        #accuracy = clf.score(testing.matrix, testing.classes)

    average_accuracy = numpy.mean(accuracies)

    logger.info('Accuracy: {0} -- {1}'.format(average_accuracy, accuracies))
    logger.debug('-'*120)


def inverse_squared(distances):
    if 0. in distances:
        return distances == 0

    return 1 / numpy.square(distances)

class Experiment(object):
    def __init__(self):
        pass

    def run(self):
        pass

    def plot(self):
        pass


"""
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print('Indicies')
print(indices)
print('='*80)
print('Distances')
print(distances)
"""


"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
"""