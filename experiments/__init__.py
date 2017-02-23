from sklearn import metrics

import logging
logger = logging.getLogger('cnn.'+__name__)
class LoggingObject(object):
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)

    def debug(self, msg):
        logger.debug(msg)

    def info(self, msg):
        logger.info(msg)

class PrecisionAndRecalls(LoggingObject):
    string_fmt = '{l: >16}: {p:.4f} -- {r:.4f} -- {s:.4f} -- {f:.4f}'
    def __init__(self, truth, predictions, available_labels,
                 label_names):
        super(PrecisionAndRecalls, self).__init__()
        self.truth = truth
        self.predictions = predictions
        self.label_indices = available_labels
        self.label_names = label_names

        self.precisions, self.recalls, self.fscores, self.supports\
            = metrics.precision_recall_fscore_support(
                y_true=truth,
                y_pred=predictions,
                labels=available_labels,
                average=None,
            )

        self.values_by_label_idx = list(zip(self.precisions,
                                            self.recalls,
                                            self.fscores,
                                            self.supports,
                                            self.label_names))
        self.values_by_label = {
            label_names[i]: {
                'precision': x[0],
                'recall': x[1],
                'fscore': x[2],
                'support': x[3]
            } for i, x in enumerate(self.values_by_label_idx)
        }

    def __getitem__(self, key):
        label_name, attribute_name = key
        return self.values_by_label[label_name][attribute_name]

    def __iter__(self):
        for tup in self.values_by_label_idx:
            yield tup

    def __str__(self):
        lines = [
            ('\n{label: >16}: {prec: ^6} -- {rec: ^6} --'
             ' {support: ^6} -- {fscore: ^6}\n'.format(
                label='Label',
                prec='Prec',
                rec='Recall',
                support='Supp',
                fscore='FScore'))]
        for p, r, f, s, l in self:
            lines.append(PrecisionAndRecalls.string_fmt.format(**locals()))

        return '\n'.join(lines)


import itertools
hatches = itertools.cycle('// * O \ | + x o .'.split())
hatch = {}
# -*- noplot -*-
"""
Some simple functions to generate colours.
"""
import numpy as np
from matplotlib import colors as mcolors

def pastel(colour, weight=2.4):
    """ Convert colour into a nice pastel shade"""
    rgb = np.asarray(mcolors.to_rgba(colour)[:3])
    # scale colour
    maxc = max(rgb)
    if maxc < 1.0 and maxc > 0:
        # scale colour
        scale = 1.0 / maxc
        rgb = rgb * scale
    # now decrease saturation
    total = rgb.sum()
    slack = 0
    for x in rgb:
        slack += 1.0 - x

    # want to increase weight from total to weight
    # pick x s.t.  slack * x == weight - total
    # x = (weight - total) / slack
    x = (weight - total) / slack

    rgb = [c + (x * (1.0 - c)) for c in rgb]

    return rgb


def get_colours(n):
    """ Return n pastel colours. """
    base = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if n <= 3:
        return base[0:n]

    # how many new colours to we need to insert between
    # red and green and between green and blue?
    needed = (((n - 3) + 1) / 2, (n - 3) / 2)

    colours = []
    for start in (0, 1):
        for x in np.linspace(0, 1, needed[start] + 2):
            colours.append((base[start] * (1.0 - x)) +
                           (base[start + 1] * x))

    return [pastel(c) for c in colours[0:n]]

colors = itertools.cycle(get_colours(15))
color = {}
