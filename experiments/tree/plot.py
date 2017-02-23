import matplotlib.pyplot as plt
from experiments import LoggingObject
from lib.lineheaderpadded import hr
import itertools
import random
import os
random.seed(0)

import logging
logger = logging.getLogger('cnn.'+__name__)

hatches = itertools.cycle('// * O \ | + x o .'.split())
hatch = {}

from sklearn import metrics
def prec_n_rec(results, class_labels, save_to=None):
    bar_width = 0.22

    logger.debug(results.keys())
    for vector_type in results:
        logger.info(hr(vector_type, '+'))
        results_for_vector_type = results[vector_type]
        logger.debug(results_for_vector_type.keys())

        num_subplots = len(results)
        fig, ax = plt.subplots(ncols=num_subplots)
        ax[0].grid(True, axis='x', zorder=2)
        ax[1].grid(True, axis='x', zorder=2)

        ax[0].invert_xaxis()
        bar_handles = {}
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F-Score']

        for i, splitting_method in enumerate(results_for_vector_type):
            logger.info(hr(splitting_method, '~'))
            results_for_splitting_method = results_for_vector_type[splitting_method]
            predictions = results_for_splitting_method['predicted']
            truths = results_for_splitting_method['actual']
            labels = numpy.arange(len(class_labels))

            # overall, how well did the prediction do?
            overall_correct = numpy.sum(truths == predictions)
            overall_accuracy = overall_correct / len(truths)
            logger.debug('Overall::')
            logger.debug('Accuracy:    {}'.format(overall_accuracy))

            vals = metrics.precision_recall_fscore_support(
                y_true=truths,
                y_pred=predictions,
                labels=labels,
                average=None,
            )

            precs, recs, fscores, supports = vals
            accuracies = []
            for cls in range(len(class_labels)):
                instances_of_class = (truths == cls)
                true_positives = (truths == predictions)
                true_positives_for_class = numpy.logical_and(true_positives,
                                                             instances_of_class)
                accuracy = numpy.sum(true_positives_for_class) / numpy.sum(
                    instances_of_class
                )
                accuracies.append(accuracy)

                #
                # logger.debug('Truths:      {}'.format(truths))
                # logger.debug('Predictions: {}'.format(predictions))
                # logger.debug('Instances of "{}"'.format(class_labels[cls]))
                # logger.debug(instances_of_class)
                # logger.debug('True positives:')
                # logger.debug(true_positives)
                # logger.debug('True positives for class:')
                # logger.debug(true_positives_for_class)
                # logger.debug('Accuracy: {}'.format(accuracy))


                # true_positives = (predictions == truths)
                # true_positives_count = numpy.sum(true_positives)
                # logger.debug('True Positives: {}'.format(true_positives_count))
                # logger.debug(true_positives)
                #
                # truths_for_class = (truths == cls)
                # truths_for_class_count = numpy.sum(truths_for_class)
                # logger.debug('Truths for cls: {}'.format(truths_for_class_count))
                # logger.debug(truths_for_class)
                #
                # true_pos_for_class = numpy.logical_and(truths_for_class, true_positives)
                # true_pos_for_class_count = numpy.sum(true_pos_for_class)
                # logger.debug('Truth Pos cls:  {}'.format(true_pos_for_class_count))
                # logger.debug(true_pos_for_class_count)

            accuracies = numpy.array(accuracies)
            # logger.debug('Precisions and Recalls:::')
            # logger.debug('Accuracies:  {}'.format(list(accuracies)))
            # logger.debug('Precision:   {}'.format(list(precs)))
            # logger.debug('Recall:      {}'.format(list(recs)))
            # logger.debug('F-Score:     {}'.format(list(fscores)))
            # logger.debug('Support:     {}'.format(list(supports)))

            metric_values = [accuracies, precs, recs, fscores]

            # Begin plotting
            axes = ax[i]
            axes.set_title(splitting_method.title())
            # axes.set_ylabel('Data Class')
            axes.set_xlabel('Performance Metric')

            indices = numpy.arange(start=0, stop=len(class_labels))
            for i, metric in enumerate(metric_names):
                logger.debug('#'*80)
                logger.debug('Plotting for {}'.format(metric))
                indices_for_metric = indices+(i*bar_width)
                fmt = '{0:.11f} -- {1:.11f} -- {2:.11f} -- {3:.11f} -- {4:.11f} ' \
                      '-- {5:.11f} -- {6:.11f}'
                header_fmt = '{0: ^13} -- {1: ^13} -- {2: ^13} -- {3: ^13} -- ' \
                             '{4: ^13} -- {5: ^13} -- {6: ^13}'
                logger.debug('Classes:  {}'.format(header_fmt.format(
                    *class_labels)))
                logger.debug('Indices:  {}'.format(fmt.format(
                    *indices_for_metric)))
                logger.debug('Values:   {}'.format(fmt.format(
                    *metric_values[i])))
                if metric not in hatch:
                    hatch[metric] = next(hatches)
                bars = axes.barh(indices_for_metric,
                                 width=metric_values[i],
                                 label=metric,
                                 height=bar_width,
                                 align='center',
                                 hatch=hatch[metric])
                if metric not in bar_handles:
                    bar_handles[metric] = bars[0]

        tickmark_locations = indices + bar_width*1.5
        ax[0].set_yticks(tickmark_locations)
        ax[0].set_yticklabels(class_labels)
        # ax[0].invert_yaxis()

        # hide tickmarks on the left-hand-side axis of right subplot
        right_axis = ax[-1]
        right_axis.set_yticks([])

        bar_labels = [l.title() for l in bar_handles.keys()]
        bar_objects = [bar_handles[l] for l in bar_handles]
        ax[-1].legend(bar_objects, bar_labels,
                      loc='upper right', bbox_to_anchor=(1.75, 1.))

        # apply tickmarks to the right-side of the left subplot
        # right = ax[-1].twinx()
        # right.set_yticks(tickmark_locations)
        # right.set_yticklabels(tick_labels)
        # right.invert_yaxis()

        fig.suptitle('Decision Tree Performance on {} vectors'.format(
            vector_type.title()))
        # plt.tight_layout()
        plt.subplots_adjust(left=0.2, right=0.8)
        if save_to is None:
            plt.show()
        else:
            plt.savefig(os.path.join(
                save_to,
                '{}.prec_n_rec.svg'.format(vector_type)))



def path_lengths(data, save_to):
    logger.debug(hr('Path Depths per Class'))
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True)
    fig.set_size_inches((10, 12))
    ax.shape = (1, 8)
    ax = ax[0]
    for i, class_label in enumerate(data):
        if class_label not in hatch:
            hatch[class_label] = next(hatches)

        data_for_class = data[class_label]

        bar_labels = []
        max_lengths = []
        min_lengths = []
        average_lengths = []
        for vector_type in data_for_class:
            data_class_vector_type = data_for_class[vector_type]
            for criterion in data_class_vector_type:
                bar_labels.append('{vector}\n{criterion}'.format(
                    vector=vector_type,
                    criterion=criterion.title()
                ))

                underlying_data = data_class_vector_type[criterion]
                max_length = max(underlying_data)
                min_length = min(underlying_data)
                average_length = numpy.mean(underlying_data)
                max_lengths.append(max_length)
                min_lengths.append(min_length)
                average_lengths.append(average_length)

        indices = range(len(bar_labels))
        axes = ax[i]
        axes.grid(True, axis='x', zorder=2)
        axes.barh(indices, average_lengths,
                xerr=[min_lengths, max_lengths],
                align='center',
                ecolor='black')

        axes.set_title(class_label.title())
        axes.set_yticks(indices)
        axes.set_yticklabels(bar_labels)

        plt.tight_layout(pad=1.8, h_pad=3)

    last_axes = ax[-1]
    last_axes.grid(False)
    last_axes.set_frame_on(False)
    last_axes.axes.get_yaxis().set_visible(False)
    last_axes.axes.get_xaxis().set_visible(False)
    #plt.suptitle('Decision Tree Path Lengths')

    if save_to is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_to,
                                 'path_depths.svg'.format(vector_type)))


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
