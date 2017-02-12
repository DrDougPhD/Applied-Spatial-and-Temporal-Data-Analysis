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
def horizontal_graph(n=15):
  import matplotlib.pyplot as plt
  val = 3+10*rand(n)    # the bar lengths
  pos = arange(n)+.5    # the bar centers on the y axis
  ylabels = [ '{0}\n{1}'.format(random_word(), random_word()) 
              for i in range(n) ]
  """
  fig = plt.figure(1)
  barh(pos,val, align='center')
  yticks(pos, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))
  xlabel('Performance')
  title('How fast do you want to go today?')
  grid(True)
  """
  plt.figure(2, figsize=(20, 6), dpi=80)
  barh(pos, val, xerr=rand(n), ecolor='r', align='center')
  yticks(pos, ylabels)
  xlabel('Performance')
  plt.tight_layout()
  plt.subplots_adjust(left=0.12)

  show()



def random_word():
  return "".join([
    random.choice(string.ascii_letters) for i in range(random.randint(3, 64))
  ]).title()


if __name__ == '__main__':
  horizontal_graph()
