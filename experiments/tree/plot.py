import numpy as np
import matplotlib.pyplot as plt
from experiments import LoggingObject
from lib.lineheaderpadded import hr
import logging
def setup_logger(name):
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    ch.setFormatter(logging.Formatter(
        "%(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ] %("
        "message)s"
    ))
    # add the handlers to the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger
logger = setup_logger('cnn')

def prec_n_rec(data):
    num_subplots = len(data)
    logger.debug('Creating {} subplots for Prec. and Rec. graphs'.format(
        num_subplots
    ))
    fig, ax = plt.subplots(ncols=num_subplots, sharey=True)

    # flip the x-axis of the left subplot
    ax[0].invert_xaxis()
    ax[0].invert_yaxis()

    for axes, splitting_method in zip(ax, data):
        axes.set_title(splitting_method.title())
        axes.set_ylabel('Data Class')
        axes.set_xlabel('Performance Metric')

        performance_data = data[splitting_method]
        # for each class of the data, make a group of bars corresponding to
        # accuracy, precision, recall, f-measure
        # for perf_metric_type in data:
        #     # [ accuracy1, accuracy2, ... ]
        #     indices, perf_metrics, style = perf_metric_type.as_tuple()
        #     axes.barh(indices, perf_metrics, align='center',
        #               color='green', ecolor='black')

        # Set the tickmarks on the y-axis to correspond to the groups and the
        #  location where the tick should go.
        tickmark_locations, tick_labels = data.get_labels()
        axes.set_yticks(tickmark_locations)
        axes.set_yticklabels(tick_labels)
        #
        # for data_by_class in data:
        #     y_indices, perf_metrics, y_labels, y_design\
        #         = data_by_class.as_tuple()



    plt.tight_layout(h_pad=1.0)
    plt.show()

import numpy
class PlottableExperimentPerformance(LoggingObject):
    bar_width = 0.22

    def __init__(self, results):
        super(PlottableExperimentPerformance, self).__init__()

        # get the splitting methods used, e.g. gini, entropy
        self.node_splitting_methods = list(results.keys())

        # transform the results for easy plotting
        self.data = self._transform(results)

        # get the class names from one of the results
        rep_data = results[self.node_splitting_methods[0]]
        self.bar_group_names = ['Overall']
        self.bar_group_names.extend([c.title() for c in rep_data.class_names])

    def _transform(self, results):
        return results

    def __len__(self):
        """
        How many splitting methods are there?
        How many subplots should be made?
        :return:
        """
        return len(self.node_splitting_methods)

    def __getitem__(self, item_key):
        return self.data[item_key]

    def __iter__(self):
        """
        Iterate through the methods used to split nodes in the decision tree
        :return: name of splitting method (e.g. 'gini', 'entropy')
        """
        for k in self.node_splitting_methods:
            yield k

    def get_labels(self):
        """
        Get axis labels, regardless of the axes / splitting type being used
        :return:
        """
        label_indices = numpy.arange(6)\
                      + (2*PlottableExperimentPerformance.bar_width)
        self.debug('Tick labels:    {}'.format(self.bar_group_names))
        self.debug('Tick labels at: {}'.format(label_indices))
        return label_indices, self.bar_group_names



class ExperimentPerformance(LoggingObject):
    def __init__(self):
        super(ExperimentPerformance, self).__init__()
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.fmeasure = None

        self.class_names = ['crime', 'living', 'entertainment', 'politics']


if __name__ == '__main__':
    entropy_results = ExperimentPerformance()
    gini_results = ExperimentPerformance()

    results = {
        'entropy': entropy_results,
        'gini': gini_results
    }
    plottable_results = PlottableExperimentPerformance(results)
    prec_n_rec(plottable_results)