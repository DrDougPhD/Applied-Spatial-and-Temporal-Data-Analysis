import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext

from matplotlib.axes import Axes
from matplotlib.figure import Figure


class OptimalKNNbyDistanceMetric(lines.Line2D):
    def __init__(self, *args, **kwargs):
        # we'll update the position when the line data is set
        self.text = mtext.Text(0, 0, '')
        lines.Line2D.__init__(self, *args, **kwargs)

        # we can't access the label attr until *after* the line is
        # inited
        self.text.set_text(self.get_label())

    def set_figure(self, figure):
        self.text.set_figure(figure)
        lines.Line2D.set_figure(self, figure)

    def set_axes(self, axes):
        self.text.set_axes(axes)
        lines.Line2D.set_axes(self, axes)

    def set_transform(self, transform):
        # 2 pixel offset
        texttrans = transform + mtransforms.Affine2D().translate(2, 2)
        self.text.set_transform(texttrans)
        lines.Line2D.set_transform(self, transform)

    def set_data(self, x, y):
        lines.Line2D.set_data(self, x, y)
        if len(x):
            text_location = self.get_last_point()
            print('Text location: {0}'.format(text_location))
            self.text.set_position(text_location)

    def draw(self, renderer):
        # draw my label at the end of the line with 2 pixel offset
        lines.Line2D.draw(self, renderer)
        self.text.draw(renderer)

    def get_last_point(self):
        return self.get_xdata()[-1], self.get_ydata()[-1]


class AccuracyAxes(Axes):
    def __init__(self, *args, **kwargs):
        Axes.__init__(self, *args, **kwargs)


###############################################################################


class ExperimentFigure(Figure):
    def __init__(self, figure):
        self.figure = figure

    def add_vertical_axes(self, axes):
        self.add_subplot(1, 1, 1, axes=axes)


class AccuracyLine(object):
    def __init__(self, results):
        self.results = results

    def trigger_confidence_boxes(self, with_outliers=False):
        pass

    def get_last_point(self):
        return self.get_xdata()[-1], self.get_ydata()[-1]


class PlotAccuracyFromK(object):
    def __init__(self, axes, label):
        self.label = label
        self.axes = axes

    def add_line(self, line):
        pass

    def add_verticle_line_at_optimal(self):
        pass

    def add_verticle_line_at(self, x):
        pass


class ExperimentDummy:
    def __init__(self, vectorizer, distance):
        self.vectorizer = vectorizer
        self.distance = distance

        self.x = np.arange(50)
        self.y = np.random.rand(50).T

    def get_results_for(self, vectorizer, distance):
        return self

    def get_vectorizer_type(self):
        return self.vectorizer


if __name__ == '__main__':

    experiment = ExperimentDummy('Term Frequency', 'cosine')
    """
    fig, verticle_axes = plt.subplots(3)
    window = ExperimentFigure(fig)

    cosine = AccuracyLine(experiment.get_results_for(vectorizer='tf',
                                                     distance='cosine'))
    cosine.trigger_confidence_boxes(with_outliers=True)

    term_freq_subplot = PlotAccuracyFromK(verticle_axes[0],
                                          label=experiment.get_vectorizer())
    term_freq_subplot.add_line(cosine)
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



    plt.show()