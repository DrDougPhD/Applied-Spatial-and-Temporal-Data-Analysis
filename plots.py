#!/usr/bin/env python
# make a horizontal bar chart

import string
import random
random.seed(0)
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('cnn.plots')

def store_to(directory, data):
    # highest similar article pairs for each fn
    """
    for fn in data:
    similarities = data[fn]
    first_10 = similarities[:10]
    most_similar_articles(first_10, fn)
    """
    article_length_distribution(data)


def most_similar_articles(similarities, function_name):
    fig, ax = plt.subplots(1, figsize=(20, 6), dpi=80)

    max_length = 80
    def short_titles(similar):
        titles = map(lambda a: a.title, similar.article)
        shorter_titles = []
        for t in titles:
            shortened = t[:max_length]
            if len(shortened) != len(t):
                shortened = '{}...'.format(shortened)
            shorter_titles.append(shortened)
        return shorter_titles

    article_titles = ['{0}\n{1}'.format(*short_titles(s)) for s in similarities]

    n = len(similarities)
    y_pos = np.arange(n)
    performance = map(lambda s: s.normalized, similarities)

    ax.barh(y_pos, list(performance), align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(article_titles)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Pairwise Similarity Scores')
    ax.set_title('{} Similarities'.format(function_name.title()))

    # add left axis
    left = ax.twinx()
    plt.text(1, 1, 'Most frequent\ncommon words:',
             transform=ax.transAxes)
    left.barh(y_pos, np.zeros(n), align='center', ecolor='black')
    left.set_yticks(y_pos)
    left_labels = ['"{0}" ({1} times)'.format(s.highest_common_feat.name,
                                              s.highest_common_feat.score)
                   for s in similarities]
    left.set_yticklabels(left_labels)
    left.invert_yaxis()  # labels read top-to-bottom

    #plt.xlim([0, 1])
    plt.tight_layout()
    plt.subplots_adjust(left=0.3, right=0.9)
    plt.show()


def article_length_distribution(data):
    fig, axes = plt.subplots(3, sharex=True, sharey=True,
                             figsize=(7, 8))

    # add a big axes, hide frame
    ax = fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axes
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                    right='off')
    ax.set_xlabel('Rank of Article Pair')
    ax.yaxis.set_tick_params(pad=15)
    ax.set_ylabel('Summed Length of Article Pair')
    ax.set_title('Article Length Distribution')

    for i, fn in enumerate(data):
        ax = axes[i]
        similarities = data[fn]
        n = len(similarities)
        x = np.arange(1, n+1)

        article_lengths = map(lambda c: sorted([c.article[0].length,
                                                c.article[1].length]),
                              similarities)

        y_lines = np.array(list(article_lengths)).T
        ax.vlines(x, y_lines[0], y_lines[1])

        ax.annotate(fn.title(), xy=(1, 0), xytext=(-5, 5),
                    xycoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    textcoords='offset pixels')
        ax.set_ylim(ymin=0)
        #axes[i].set_ylabel('Function {}'.format(i))

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    plt.tight_layout()
    plt.show()


def scatterplot(n=100):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, sharex=True, sharey=True)

    # add a big axes, hide frame
    ax = fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axes
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                    right='off')

    for i in range(3):
        np.random.seed(0)
        x = np.arange(1, n+1)
        y = 20 + 3*x + np.random.normal(0, 60, n)

        axes[i].plot(x, y, "o")
        axes[i].annotate('XLabel', xy=(1, 0), xytext=(-5, 5),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom',
                         textcoords='offset pixels')
        #axes[i].set_ylabel('Function {}'.format(i))

    ax.set_xlabel('Rank of Article Pair')
    ax.set_ylabel('Summed Length of Article Pair')
    ax.set_title('Article Length Distribution')

    plt.tight_layout()
    plt.show()


def oo_graph():
  from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  from matplotlib.figure import Figure

  fig = Figure()
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)
  ax.plot([1, 2, 3])
  ax.set_title('hi mom')
  ax.grid(True)
  ax.set_xlabel('time')
  ax.set_ylabel('volts')
  canvas.print_figure('test')

def horizontal_graph(n=10):

  plt.rcdefaults()
  fig, ax = plt.subplots(1, figsize=(20, 6), dpi=80)

  # Example data
  ylabels = [ '{0}\n{1}'.format(random_word(), random_word()) 
              for i in range(n) ]

  y_pos = np.arange(n)
  performance = 3 + 10 * np.random.rand(n)
  error = np.random.rand(n)

  ax.barh(y_pos, performance, xerr=error, align='center',
          color='green', ecolor='black')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(ylabels)
  ax.invert_yaxis()  # labels read top-to-bottom
  ax.set_xlabel('Performance')
  ax.set_title('How fast do you want to go today?')

  # add left axis
  left = ax.twinx()
  plt.text(13.61, .5, 'Most frequent\ncommon word')
  left.barh(y_pos, np.zeros(n), align='center',
          color='green', ecolor='black')
  left.set_yticks(y_pos)
  left_labels = [ '"{0}"'.format(random_word(10)) 
                  for i in range(n) ]
  left.set_yticklabels(left_labels)
  left.invert_yaxis()  # labels read top-to-bottom

  plt.tight_layout()
  plt.subplots_adjust(left=0.12, right=0.9)
  plt.show()


def two_scales():
  fig, ax1 = plt.subplots()
  t = np.arange(0.01, 10.0, 0.01)
  s1 = np.exp(t)
  ax1.plot(t, s1, 'b-')
  ax1.set_xlabel('time (s)')
  # Make the y-axis label, ticks and tick labels match the line color.
  ax1.set_ylabel('exp', color='b')
  ax1.tick_params('y', colors='b')

  ax2 = ax1.twinx()
  s2 = np.sin(2 * np.pi * t)
  ax2.plot(t, s2, 'r.')
  ax2.set_ylabel('sin', color='r')
  ax2.tick_params('y', colors='r')

  fig.tight_layout()
  plt.show()


def random_word(n=64):
  return "".join([
    random.choice(string.ascii_letters) for i in range(random.randint(3, n))
  ]).title()


if __name__ == '__main__':
  #horizontal_graph()
  #two_scales()
  scatterplot(n=100)
