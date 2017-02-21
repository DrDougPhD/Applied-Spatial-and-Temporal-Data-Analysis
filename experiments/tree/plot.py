import numpy as np
import matplotlib.pyplot as plt
from experiments import LoggingObject
from lib.lineheaderpadded import hr
import logging
logger = logging.getLogger('cnn.'+__name__)

def prec_n_rec(data):
    fig, ax = plt.subplots(nrows=2)

    plt.show()

class ExperimentResults(LoggingObject):
    def __init__(self):
        super(ExperimentResults, self).__init__()
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.fmeasure = None

if __name__ == '__main__':
    dummy_results = ExperimentResults()
    prec_n_rec(dummy_results)