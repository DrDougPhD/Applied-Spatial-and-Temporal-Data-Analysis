import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext

from matplotlib.axes import Axes
try:
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

import logging
logger = logging.getLogger('cnn.'+__name__)

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
        # axes.set_autoscale_on(True)

    def add_line(self, line):
        self.lines.append(line)
        #self.axes.add_line(line.get_line())
        #self.axes.set_xlim(left=0, right=line.get_xmax(), auto=True)
        #self.axes.set_ylim(bottom=0, top=1.05*line.get_ymax(), auto=True)
        self.debug('Plotting {}:'.format(line.label))
        self.debug('\tx := {}'.format(line.x))
        self.debug('\ty := {}'.format(line.y))
        self.axes.plot(line.x, line.y)

        #self.axes.set_ylim(bottom=0, top=line.get_ymax())


        # add a text for the line
        x_min, x_max = self.axes.get_xlim()
        x_offset = 0.01 * (x_max - x_min)
        x_text = x_max + x_offset
        y = line.get_last_point()[1]
        self.debug('Text for {0} at point: ({1}, {2})'.format(line.label,
                                                               x_text, y))
        self.axes.text(x=x_text, y=y,
                       s=line.label)

    def add_verticle_line_at_optimal(self):
        pass

    def add_verticle_line_at(self, x):
        pass


class ExperimentDummy(LoggingObject):
    def __init__(self, vectorizer, distance):
        self.vectorizer = vectorizer
        self.distance = distance

        self.x = np.arange(50)
        self.y = np.random.rand(50).T
        self.label = distance

    def get_results_for(self, vectorizer, distance):
        self.x = np.arange(50)
        self.y = np.random.rand(50).T
        self.vectorizer = vectorizer
        self.distance = distance
        return self

    def get_vectorizer_type(self):
        return self.vectorizer

    def __str__(self):
        return (
            'X: {x}\n'
            'Y: {y}'.format(x=self.x, y=self.y)
        )


def draw(experiment):
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

    plt.tight_layout(h_pad=1.0)
    plt.subplots_adjust(right=0.87)
    plt.show()


if __name__ == '__main__':
    experiment = ExperimentDummy('Term Frequency', 'cosine')
    draw(experiment)


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