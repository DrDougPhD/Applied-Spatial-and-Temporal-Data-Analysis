import pprint

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import os
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext
from matplotlib import cm
from matplotlib import gridspec

from matplotlib.axes import Axes
from matplotlib.ticker import IndexLocator

try:
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

import logging
logger = logging.getLogger('cnn.'+__name__)

from experiments import LoggingObject, color, colors, hatch, hatches
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


def neighbor_heatmap(neighbors, feature_names, save_to):
    #save_to = None
    # only interested in the neighbors for TDIDF, Cosine
    logger.info('######## Neighbor Heatmap')
    if len(neighbors['tfidf']['cosine']) > 5:
        neighborinos = neighbors['tfidf']['cosine'][4]
    else:
        neighborinos = neighbors['tfidf']['cosine'][-1]
    logger.debug('First of the neighborinos')

    neighborinos.sort(key=lambda x: x['summed_distances'])

    # neighborhood = neighborinos[0]
    # article = neighborhood['article']
    #
    # neighbors_of_article = neighborhood['neighbors']
    # neighbor_vectors = [a['neighbor'].vector for a in neighbors_of_article]
    #
    # neighbors_as_matrix = numpy.concatenate(
    #     ([article.vector], [*neighbor_vectors]),
    #     axis=0
    # )

    # Print article neighbors to console
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

    #fig, ax = plt.subplots(2, 3)

    # for i in [0, -1]:
    def heatmap_n_stuff(i, heatmap, distances, max_common_feat):
        heatmap.set_title('Feature Occurrences')
        #distances.set_title('Distance to\nNeighbors')
        max_common_feat.set_title('Most\nCommon\nFeature')

        heatmap.invert_yaxis()

        neighborhood = neighborinos[i]

        neighbors_of_article = neighborhood['neighbors']
        neighbor_vectors = [a['neighbor'].vector for a in
                            neighbors_of_article]

        article = neighborhood['article']
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
              category=art[
                  'neighbor'].category.title())
            for art in neighbors_of_article])

        c = heatmap.pcolor(neighbors_as_matrix > 0, cmap=cm.gray_r)
        #fig.colorbar(mappable=c, ax=heatmap)

        heatmap.grid(axis='y')

        y_incides = range(neighbors_as_matrix.shape[0])
        heatmap.set_yticks(y_incides)
        heatmap.set_yticklabels(article_labels, verticalalignment='top',
                                size='small')

        # create the subplot for the distances
        neighbor_distances = numpy.array([0,
            *[art['distance'] for art in neighbors_of_article]])
        indices = numpy.arange(len(neighbor_distances))

        distances.set_title('Distances')
        bars = distances.barh(indices+0.5,
                              width=neighbor_distances,
                              #edgecolor='black'
                              align='center')
        distances.set_ylim((0, len(neighbor_distances)))
        distances.set_yticks([])
        distances.set_xlim((0, 1))
        distances.invert_yaxis()


        # plt.colorbar(mappable=c, ax=ax[0])

        # Determine the common labels between the target article and each
        # neighbor
        max_common_feat.set_xlabel('Max.\nShared\nFeature')
        max_common_feat.set_ylim((0, len(neighbors_of_article)))

        unique_feature_indices = set()
        existence_in_article = article.vector > 0
        most_common_features = []
        for index, neighbor in enumerate(neighbors_of_article):
            existence_in_neighbor = neighbor['neighbor'].vector > 0
            shared_existence = numpy.logical_and(existence_in_article,
                                                 existence_in_neighbor)

            indices = numpy.arange(len(shared_existence))
            existence_indices = indices[shared_existence]

            # find the feature that has the greatest summed value
            summed_feature_values = (
                article.vector+neighbor['neighbor'].vector)[existence_indices]
            v, i = max(zip(summed_feature_values, existence_indices),
                       key=lambda x: x[0])
            max_feat = feature_names[i]
            logger.debug(neighbor['neighbor'].title)
            logger.debug('{0: ^15} -- {1: .5f} -- {2: .5f}'.format(
                max_feat, article.vector[i], neighbor['neighbor'].vector[i]
            ))

            most_common_features.append(max_feat)
            max_common_feat.text(x=0.5, y=index+0.5, s=max_feat,
                                 horizontalalignment='center')

        max_common_feat.invert_yaxis()
        max_common_feat.axis('off')



        #     logger.debug('Shared indices between article and neighbor:')
        #     logger.debug(existence_indices)
        #     unique_feature_indices.update(existence_indices)
        #
        # logger.debug('Shared indices with at least one neighbor:')
        # unique_feature_indices = numpy.array(sorted(unique_feature_indices))
        # logger.debug(unique_feature_indices)
        #
        # feature_names = numpy.array(feature_names, dtype=numpy.unicode_)
        # logger.debug(feature_names[unique_feature_indices])
        #
        # top_axis = axes.twiny()
        # top_axis.invert_xaxis()
        # top_axis.set_xticks(unique_feature_indices)
        # top_axis.set_xticklabels(feature_names[unique_feature_indices],
        #                          rotation='vertical')

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])
    heatmap_n_stuff(i=0, heatmap=plt.subplot(gs[0]),
                    distances=plt.subplot(gs[1]),
                    max_common_feat=plt.subplot(gs[2]))

    heatmap_n_stuff(i=-1, heatmap=plt.subplot(gs[3]),
                    distances=plt.subplot(gs[4]),
                    max_common_feat=plt.subplot(gs[5]))

    plt.subplots_adjust(left=0.35)
    #plt.tight_layout(pad=2)

    plt.suptitle('Heatmap of Shared Features')
    if save_to:
        plt.savefig(os.path.join(save_to, 'knn_heatmaps.png'),
                    bbox_inches='tight')
    else:
        plt.show()


def neighborhood_radii(neighbors, save_to=None):
    logger.info(hr('Neighborhood Radii'))
    neighborinos = neighbors['tfidf']['cosine'][5]

    #neighborinos.sort(key=lambda x: x['summed_distances'], reverse=True)
    fig = plt.figure()
    ax = plt.subplot()

    scatter_colors = {}
    class_colors = {}
    colors_from_which_to_choose = itertools.cycle('rgbcmykw')

    def scatterplot_average_distance(neighborinos, k):
        neighborinos.sort(key=lambda x: x['summed_distances'])

        num_neighbors_with_same_class = []
        for x, neighborhood in enumerate(neighborinos):
            article_class = neighborhood['article'].category
            if article_class not in class_colors:
                class_colors[article_class] = next(colors_from_which_to_choose)

            common_category_neighbors = 0
            for neighbor in neighborhood['neighbors']:
                if neighbor['neighbor'].category == article_class:
                    common_category_neighbors += 1

            #num_neighbors_with_same_class.append(common_category_neighbors)
            ax.scatter(x, common_category_neighbors, color=class_colors[article_class])

        # ax.scatter(range(len(neighborinos)),
        #            num_neighbors_with_same_class)

        ## normalized summed distances
        # scatter_colors[k] = next(colors)
        #
        # min_distance = min(neighborinos,
        #                    key=lambda x: x['summed_distances'])[
        #     'summed_distances']
        # max_distance = max(neighborinos,
        #                    key=lambda x: x['summed_distances'])[
        #     'summed_distances']
        # normalized_distances = [
        #     (x['summed_distances'] - min_distance)/(max_distance-min_distance)
        #     for x in neighborinos
        # ]
        # ax.scatter(range(len(neighborinos)), normalized_distances)


        # for x, neighborhood in enumerate(neighborinos):
        #     average_distance = sum([
        #         n['distance'] for n in neighborhood['neighbors']
        #     ]) / len(neighborhood['neighbors'])
        #     ax.scatter(x, average_distance, color=scatter_colors[k])

    logger.info(neighbors.keys())
    logger.info(neighbors['tfidf'].keys())
    scatterplot_average_distance(neighbors['tfidf']['euclidean'][5], 5)
    #scatterplot_average_distance(neighbors['tfidf']['cosine'][5], 5)
    #scatterplot_average_distance(neighbors['tf']['cosine'][5], 5)
    #scatterplot_average_distance(neighbors['tf']['euclidean'][5], 5)

    #plt.show()


if __name__ == '__main__':
    pass