import numpy as np
import matplotlib.pyplot as plt
from experiments import LoggingObject
from lib.lineheaderpadded import hr
import logging
logger = logging.getLogger('cnn.'+__name__)

def prec_n_rec(data):
    num_subplots = len(data)
    logger.debug('Creating {} subplots for Prec. and Rec. graphs'.format(
        num_subplots
    ))
    fig, ax = plt.subplots(ncols=num_subplots, sharex=True, sharey=True)
    for axes, tree_splitting_measure in zip(ax, data.keys()):
        axes.set_title(tree_splitting_measure.title())
        axes.set_ylabel('Data Class')
        axes.set_xlabel('Performance Metric')

    plt.tight_layout(h_pad=1.0)
    plt.show()

class ExperimentResults(LoggingObject):
    def __init__(self):
        super(ExperimentResults, self).__init__()
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.fmeasure = None

if __name__ == '__main__':
    entropy_results = ExperimentResults()
    gini_results = ExperimentResults()

    results = {
        'entropy': entropy_results,
        'gini': gini_results
    }
    prec_n_rec(results)