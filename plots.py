#!/usr/bin/env python
# make a horizontal bar chart

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

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


from pylab import *
import string
import random
random.seed(0)
import matplotlib.pyplot as plt
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
  left.set_yticks(y_pos)
  left_labels = [ '"{0}"'.format(random_word(10)) 
                  for i in range(n) ]
  left.set_yticklabels(left_labels)
  left.invert_yaxis()  # labels read top-to-bottom

  plt.tight_layout()
  plt.subplots_adjust(left=0.12, right=0.9)
  plt.show()


  """
  val = 3+10*rand(n)    # the bar lengths
  pos = arange(n)+.5    # the bar centers on the y axis
  ylabels = [ '{0}\n{1}'.format(random_word(), random_word()) 
              for i in range(n) ]
  fig = plt.figure(1)
  barh(pos,val, align='center')
  yticks(pos, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))
  xlabel('Performance')
  title('How fast do you want to go today?')
  grid(True)
  plt.figure(2, figsize=(20, 6), dpi=80)
  barh(pos, val, xerr=rand(n), ecolor='r', align='center')
  yticks(pos, ylabels)
  xlabel('Performance')

  show()
  """


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
  horizontal_graph()
  #two_scales()
