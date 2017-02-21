from experiments import LoggingObject

class Experiment(LoggingObject):
    def __init__(self, cross_validation_n, corpus_series):
        super(Experiment, self).__init__()

    def run(self, series, vector_type, splitting_criterion, x):
        pass