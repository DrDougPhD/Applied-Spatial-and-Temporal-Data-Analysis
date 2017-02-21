from experiments import LoggingObject
from sklearn import tree
import pydotplus
import os

class Experiment(LoggingObject):
    def __init__(self, cross_validation_n, corpus_series):
        super(Experiment, self).__init__()
        self.n = cross_validation_n
        self.datasets = corpus_series

    def export(clf, feature_names, class_names, save_to, name):
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=feature_names,
                                        class_names=class_names,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(os.path.join(save_to,
                                     '{}.decision_tree.pdf'.format(name)))

    def accuracy(self, series, vector_type, splitting_criterion, x):
        return 4