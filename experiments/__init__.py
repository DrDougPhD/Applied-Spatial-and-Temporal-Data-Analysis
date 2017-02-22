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
