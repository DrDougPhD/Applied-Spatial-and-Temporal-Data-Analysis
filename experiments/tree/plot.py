import numpy as np
import matplotlib.pyplot as plt
from experiments import LoggingObject
from lib.lineheaderpadded import hr
import logging
import random

random.seed(0)


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

def prec_n_rec(results, title):
    data = PlottableExperimentPerformance(results)
    num_subplots = len(data)
    logger.debug('Creating {} subplots for Prec. and Rec. graphs'.format(
        num_subplots
    ))
    fig, ax = plt.subplots(ncols=num_subplots)

    ax[0].invert_xaxis()
    bar_handles = {}

    # each subplot corresponds to a node splitting method
    for axes, splitting_method_data in zip(ax, data):
        logger.debug(hr(splitting_method_data.title()))

        axes.set_title(splitting_method_data.title())
        #axes.set_ylabel('Data Class')
        axes.set_xlabel('Performance Metric')

        # for each class of the data, make a group of bars corresponding to
        # accuracy, precision, recall, f-measure
        for perf_metric_type in splitting_method_data.by_metric_type():

            # [ accuracy1, accuracy2, ... ]
            indices, perf_metrics, style, name = perf_metric_type
            logger.debug(hr(name, '+'))

            bars = axes.barh(indices, perf_metrics, align='center',
                             **style)
            logger.debug('Bar object: {}'.format([b for b in bars]))
            if name not in bar_handles:
                bar_handles[name] = bars[0]

        axes.invert_yaxis()
        #
        # for data_by_class in data:
        #     y_indices, perf_metrics, y_labels, y_design\
        #         = data_by_class.as_tuple()

    # flip the x-axis of the left subplot
    #ax[0].invert_xaxis()

    # apply labels appropriately
    tickmark_locations, tick_labels = data.get_labels()
    ax[0].set_yticks(tickmark_locations)
    ax[0].set_yticklabels(tick_labels)
    #ax[0].invert_yaxis()

    # hide tickmarks on the left-hand-side axis of right subplot
    right_axis = ax[-1]
    right_axis.set_yticks([])

    logger.debug('Items in the bar_handles list:\n{}'.format(bar_handles))
    bar_labels = [l.title() for l in bar_handles.keys()]
    bar_objects = [bar_handles[l] for l in bar_handles]
    ax[-1].legend(bar_objects, bar_labels,
                  loc='upper right')

    # apply tickmarks to the right-side of the left subplot
    #right = ax[-1].twinx()
    #right.set_yticks(tickmark_locations)
    #right.set_yticklabels(tick_labels)
    #right.invert_yaxis()

    fig.suptitle(title)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8)
    plt.show()

import numpy
class PlottableExperimentPerformance(LoggingObject):
    bar_width = 0.22

    def __init__(self, results):
        super(PlottableExperimentPerformance, self).__init__()

        # get the splitting methods used, e.g. gini, entropy
        self.node_splitting_methods = list(results.keys())

        # transform the results for easy plotting
        self.data = tx_results = {
            k: PlottableDataFromSplittingType(results[k],
                                              splitting_method_name=k)
            for k in self.node_splitting_methods
        }

        # get the class names from one of the results
        rep_data = results[self.node_splitting_methods[0]]
        self.bar_group_names = [c.title() for c in rep_data.class_names]

    def __len__(self):
        """
        How many splitting methods are there?
        How many subplots should be made?
        :return:
        """
        return len(self.node_splitting_methods)

    def __iter__(self):
        """
        Iterate through the methods used to split nodes in the decision tree
        :return: name of splitting method (e.g. 'gini', 'entropy')
        """
        for k in self.node_splitting_methods:
            yield self.data[k]

    def get_labels(self):
        """
        Get axis labels, regardless of the axes / splitting type being used
        :return:
        """
        label_indices = numpy.arange(len(self.bar_group_names))\
                      + (2.5*PlottableExperimentPerformance.bar_width)
        self.debug('Tick labels:    {}'.format(self.bar_group_names))
        self.debug('Tick labels at: {}'.format(label_indices))
        return label_indices, self.bar_group_names


from collections import defaultdict
import itertools
class PlottableDataFromSplittingType(LoggingObject):
    hatches = itertools.cycle('// * O \ | + x o .'.split())
    colors = itertools.cycle([
        'green', 'blue', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'
    ])
    style_for = {}

    def __init__(self, results, splitting_method_name):
        super(PlottableDataFromSplittingType, self).__init__()
        self.splitting_method_name = splitting_method_name

        self.metric_names = None
        self.performance_metrics_split_by_type = defaultdict(list)
        for class_name in results.data:
            results_by_class = results.data[class_name]
            if not self.metric_names:
                # build up the metric names (e.g. 'accuracy') on the fly
                self.metric_names = list(results_by_class.keys())

            for metric in self.metric_names:
                self.performance_metrics_split_by_type[metric].append(
                    results_by_class[metric]
                )
            self.performance_metrics_split_by_type['class_names'].append(
                class_name)
        logger.debug('Transformed results:')
        logger.debug(self.performance_metrics_split_by_type)

        for n in self.metric_names:
            if n not in self.style_for:
                PlottableDataFromSplittingType.style_for[n] = {
                    'hatch': next(PlottableDataFromSplittingType.hatches),
                    #'color': next(PlottableDataFromSplittingType.colors),
                    'height': PlottableExperimentPerformance.bar_width,
                }

    def by_metric_type(self):
        self.debug('Accessing result data by the performance metric type')
        offset = PlottableExperimentPerformance.bar_width
        for i, metric_name in enumerate(self.metric_names, start=1):
            values = self.performance_metrics_split_by_type[metric_name]
            indices = numpy.arange(len(values)) + offset*i

            logger.debug('Indices for {} bars'.format(metric_name))
            logger.debug(indices)

            style = self.style_for[metric_name]
            self.debug('Style of bar: {}'.format(style))

            yield indices, values, style, metric_name

    def title(self):
        return self.splitting_method_name.title()


class ExperimentPerformance(LoggingObject):
    def __init__(self):
        super(ExperimentPerformance, self).__init__()
        self.class_names = ['overall', 'crime', 'living', 'entertainment',
                            'politics']
        self.data = {
            k: {
                metric: random.random() for metric in ['accuracy',
                                                       'precision',
                                                       'recall',
                                                       'f-score']
            } for k in self.class_names
        }


if __name__ == '__main__':
    entropy_results = ExperimentPerformance()
    gini_results = ExperimentPerformance()

    results = {
        'entropy': entropy_results,
        'gini': gini_results
    }
    prec_n_rec(results, 'Term Frequency')