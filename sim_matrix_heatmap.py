import logging

import numpy
from matplotlib import pyplot

logger = logging.getLogger('cnn.' + __name__)

import utils

def plot(sorted_matrix, distance_metric):
    # Create an array of distance values between each pair
    similarities = utils.similarity_matrix(
        matrix=[article.vector for article in sorted_matrix],
        distance_metric=distance_metric,
    )
    logger.debug('\n{}'.format(similarities))

    x_vals = []
    y_vals = []
    colors = []
    for (x, y), color in numpy.ndenumerate(similarities):
        x_vals.append(x)
        y_vals.append(y)
        colors.append(color)

    logger.debug('X: {0} -- {1}'.format(len(x_vals), x_vals))
    logger.debug('Y: {0} -- {1}'.format(len(y_vals), y_vals))
    logger.debug('C: {0} -- {1}'.format(len(colors), colors))

    pyplot.pcolor(similarities)
    #pyplot.xticks(numpy.arange(sorted_matrix.shape[0]),
    # article_cluster_indices)
    pyplot.show()