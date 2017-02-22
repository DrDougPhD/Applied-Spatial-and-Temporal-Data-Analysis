import pprint

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import os
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext
from matplotlib import cm

from matplotlib.axes import Axes
from matplotlib.ticker import IndexLocator

try:
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

import logging
logger = logging.getLogger('cnn.'+__name__)

import itertools
pixel = ','
point = '.'
circle = 'o'
triangle_down = 'v'
square = 's'
plus = '+'
star = '*'
cross_x = 'x'
diamond = 'd'
marker_list = [pixel,
               circle,
               triangle_down,
               square,
               plus,
               star,
               cross_x,
               diamond,
               point,
               ]
marker = itertools.cycle(marker_list)


class LoggingObject(object):
    def __init__(self, name=None):
        if name is None:
            name = 'cnn.{n}.{cn}'.format(n=__name__, cn=self.__class__.__name__)
        self.logger = logging.getLogger(name)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)


class ExperimentFigure(LoggingObject):
    def __init__(self, figure):
        self.figure = figure
        self.v_subplot_counts = 1
        self.h_subplot_counts = 1

        # add a big axes, hide frame
        self.hidden_axes = figure.add_subplot(111, frameon=False)

        # hide tick and tick label of the big axes
        self.hidden_axes.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                                     right='off')
        #self.hidden_axes.set_xlabel('Rank of Article Pair')
        #self.hidden_axes.yaxis.set_tick_params(pad=15)
        self.hidden_axes.set_ylabel('Accuracy')
        #self.hidden_axes.set_title('Article Length Distribution')

    def add_vertical_axes(self, axes):
        self.figure.add_subplot(self.v_subplot_counts,
                                self.h_subplot_counts,
                                1, axes=axes)
        self.v_subplot_counts += 1


class AccuracyLine(LoggingObject):
    def __init__(self, results):
        super(AccuracyLine, self).__init__()
        self.results = results

        self.line = lines.Line2D(xdata=results.x,
                                 ydata=results.y)
        self.x = results.x
        self.y = results.y
        self.label = results.label

        self.debug(self.results)

    def trigger_confidence_boxes(self, with_outliers=False):
        pass

    def get_last_point(self):
        last_point = self.results.x[-1], self.results.y[-1]
        self.debug('Last point: {}'.format(last_point))
        return last_point

    def get_line(self):
        return self.line

    def get_xmax(self):
        v = max(self.results.x)
        self.debug('Max X: {}'.format(v))
        return v

    def get_ymax(self):
        v = max(self.results.y)
        self.debug('Max Y: {}'.format(v))
        return v


class PlotAccuracyFromK(LoggingObject):
    markers = {}
    def __init__(self, axes, label):
        super(PlotAccuracyFromK, self).__init__()
        self.label = label
        self.axes = axes
        self.lines = []

        # Configure axes
        axes.set_title(label)
        axes.margins(0, tight=True)
        axes.set_xlim(left=0, auto=True)
        axes.set_ylim(bottom=0, auto=True)
        axes.set_xlabel('k (for kNN)')
        # axes.set_autoscale_on(True)

    def add_line(self, line):
        self.lines.append(line)
        #self.axes.add_line(line.get_line())
        #self.axes.set_xlim(left=0, right=line.get_xmax(), auto=True)
        #self.axes.set_ylim(bottom=0, top=1.05*line.get_ymax(), auto=True)
        self.debug('Plotting {}:'.format(line.label))
        self.debug('\tx := {}'.format(line.x))
        self.debug('\ty := {}'.format(line.y))
        if line.label not in PlotAccuracyFromK.markers:
            PlotAccuracyFromK.markers[line.label] = next(marker)
        m = PlotAccuracyFromK.markers[line.label]

        self.axes.plot(line.x, line.y, label=line.label,
                       marker=m)

        #self.axes.set_ylim(bottom=0, top=line.get_ymax())


        # add a text for the line
        x_min, x_max = self.axes.get_xlim()
        x_offset = 0.01 * (x_max - x_min)
        x_text = x_max + x_offset
        y = line.get_last_point()[1]
        #self.debug('Text for {0} at point: ({1}, {2})'.format(line.label,
        #                                                       x_text, y))
        #self.axes.text(x=x_text, y=y,
        #               s=line.label)

    def add_verticle_line_at_optimal(self):
        pass

    def add_verticle_line_at(self, x):
        pass


def draw_accuracies(experiment, save_to):
    logger.info(hr('Plotting Results'))
    fig, verticle_axes = plt.subplots(len(experiment.series),
                                      sharex=True,
                                      sharey=True)
    window = ExperimentFigure(fig)

    for axes, vectorizer_type in zip(verticle_axes, experiment.series):
        logger.info(hr('Matrix type: {}'.format(vectorizer_type), '-'))
        subplot = PlotAccuracyFromK(axes, label=vectorizer_type)

        for distance_metric in experiment.variations:
            logger.info('Distance metrix: {}'.format(distance_metric))
            l = AccuracyLine(experiment.get_results_for(
                    series=vectorizer_type,
                    variation=distance_metric))
            l.trigger_confidence_boxes(with_outliers=True)
            subplot.add_line(l)

        legend = axes.legend(loc='center right', shadow=True,
                             bbox_to_anchor=(1.33, 0.5))
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

    plt.tight_layout(h_pad=1.0)
    plt.subplots_adjust(right=0.87)
    plt.savefig(os.path.join(save_to, 'accuracies.pdf'),
                bbox_inches='tight')


def draw_fmeasures(experiment, best_metric_for_matrix, save_to):
    fmeasure_markers = {}

    logger.debug(hr('Drawing F-Measures Plot'))
    def get_fmeasures(results):
        precs_n_recs = results.pnc

        # just grab the label names from the first one
        labels = precs_n_recs[0].label_names

        logger.debug('Precision and Recall items: {}'.format(len(precs_n_recs)))

        return [f.fmeasure() for f in precs_n_recs], labels


    fig, ax = plt.subplots(len(best_metric_for_matrix), sharex=True)

    results = experiment.results
    for axes, (k, matrix) in zip(ax, best_metric_for_matrix):

        results_for_matrix_type = results[matrix]
        results_by_metric = results_for_matrix_type[k]
        xvals = results_by_metric.x
        fmeasure_results = results_by_metric.prec_n_recs

        #fmeasures, labels = get_fmeasures(results_by_metric)
        logger.debug('Just extracted fmeasures. What does it look like?')
        logger.debug(results_by_metric)
        for l in experiment.classnames:
            yvals = [pnc[l, 'fscore'] for pnc in fmeasure_results]
            if l not in fmeasure_markers:
                fmeasure_markers[l] = next(marker)
            m = fmeasure_markers[l]
            line = axes.plot(xvals, yvals, label=l, marker=m)

        axes.set_title('{distance} distances between {matrix} vectors'.format(
            distance=k.title(), matrix=matrix.title()))
        axes.set_xlabel('k (for kNN)')
        axes.set_ylabel('F-Measure')
        axes.set_xlim(left=xvals[0], right=xvals[-1])
        legend = axes.legend(loc='center right', shadow=True,
                             bbox_to_anchor=(1.33, 0.5))
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        # Set the fontsize
        # for label in legend.get_texts():
        #     label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

    """

    fig, ax = plt.subplot(len(plotnames), sharex=True)
    for i, axes in enumerate(ax):
        for linename in linenames_per_plot:
            line = axes.plot(None, None, label=linename)
        axes.set_title(subplots[i])
    """
    plt.tight_layout(h_pad=1.0)
    plt.subplots_adjust(right=0.87)
    plt.savefig(os.path.join(save_to, 'fmeasures.pdf'),
                bbox_inches='tight')

    # f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].plot(x, y)
    # axarr[0].set_title('Sharing X axis')
    # axarr[1].scatter(x, y)


def neighbor_heatmap(neighbors, save_to):
    # only interested in the neighbors for TDIDF, Cosine
    logger.info('######## Neighbor Heatmap')
    neighborinos = neighbors['tfidf']['cosine'][-1]
    logger.debug('First of the neighborinos')

    neighborhood = neighborinos[0]
    article = neighborhood['article']

    neighbors_of_article = neighborhood['neighbors']
    neighbor_vectors = [a['neighbor'].vector for a in neighbors_of_article]

    neighbors_as_matrix = numpy.concatenate(
        ([article.vector], [*neighbor_vectors]),
        axis=0
    )

    articles_considered = []
    for n in neighborinos:
        article = n['article']
        logger.debug(hr('Neighbor listing'))
        logger.debug(article.title)

        if article.title in articles_considered:
            logger.debug('Skipping "{}"'.format(article.title))
            continue
        else:
            # if we come across an article that is closest to the main article,
            # then we might come across that article as the main article soon
            # afterwards.
            articles_considered.append(n['neighbors'][0]['neighbor'].title)

        neighboring_articles = map(lambda x: x['neighbor'],
                                   n['neighbors'])

        for j, art in enumerate(neighboring_articles):
            logger.debug(' --> {0: .4f}   {1}'.format(
                n['neighbors'][j]['distance'],
                art.title))

    fig, ax = plt.subplots(3)
    fig.set_size_inches(12, 16)
    for i in range(1):
        neighborhood = neighborinos[i]
        article = neighborhood['article']

        neighbors_of_article = neighborhood['neighbors']
        neighbor_vectors = [a['neighbor'].vector for a in neighbors_of_article]

        neighbors_as_matrix = numpy.concatenate(
            ([article.vector], [*neighbor_vectors]),
            axis=0
        )
        article_labels = [
            '{title}\nCategory: {category}'.format(
                title=article.title,
                category=article.category.title())]
        article_labels.extend([
            '{title}\nCategory: {category}'.format(
                title=art['neighbor'].title,
                category=art['neighbor'].category.title())
            for art in neighbors_of_article])

        y_incides = range(neighbors_as_matrix.shape[0])
        axes = ax[i]
        c = axes.pcolor(neighbors_as_matrix, cmap=cm.gray_r)
        axes.grid(axis='y')
        axes.set_yticks(y_incides)
        axes.set_yticklabels(article_labels, verticalalignment='bottom')
        plt.colorbar(mappable=c, ax=ax[0])

    plt.title('Color Map of Shared Features')
    plt.show()


if __name__ == '__main__':
    neighbor_heatmap(None, None)
    # fig, verticle_axes = plt.subplots(3, sharex=True)
    # cosine = AccuracyLine(experiment.get_results_for(
    #     vectorizer='Term Frequency',
    #     distance='cosine'))
    # cosine.trigger_confidence_boxes(with_outliers=True)
    #
    # jaccard = AccuracyLine(experiment.get_results_for(
    #     vectorizer='Term Frequency',
    #     distance='jaccard'))
    # jaccard.trigger_confidence_boxes(with_outliers=True)
    #
    # euclidean = AccuracyLine(experiment.get_results_for(
    #     vectorizer='Term Frequency',
    #     distance='euclidean'))
    # euclidean.trigger_confidence_boxes(with_outliers=True)
    #
    # term_freq_subplot = PlotAccuracyFromK(
    #     verticle_axes[0],
    #     label=experiment.get_vectorizer_type())
    # lines = [cosine, jaccard, euclidean]
    # for l in lines:
    #     term_freq_subplot.add_line(l)


    # term_freq_subplot = PlotAccuracyFromK(
    #     verticle_axes[0],
    #   label=experiment.get_vectorizer_type())

    """
    term_freq_subplot.add_verticle_line_at_optimal()
    term_freq_subplot.add_verticle_line_at(x=3)
    """

    """
    #fig, (ax1, ax2, ax3) = plt.subplots(3)

    #fig = plt.figure()
    fig = ExperimentFigure()
    axes = AccuracyAxes(fig=fig, )
    fig.add_vertical_axes(axes)
    #axes.set_figure(fig=fig)

    x = np.arange(20)
    y = np.random.rand(20).T
    #x, y = np.random.rand(2, 20)
    print('X: {}'.format(x))
    print('Y: {}'.format(y))
    line = OptimalKNNbyDistanceMetric(x, y, mfc='red', ms=12, label='line label')
    #line.text.set_text('line label')
    line.text.set_color('red')
    line.text.set_fontsize(16)

    axes.add_line(line)
    """