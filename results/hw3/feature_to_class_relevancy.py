import pprint

import numpy as np
import csv

from progressbar import ProgressBar
from scipy.stats import rv_continuous
from scipy.stats import uniform

filename = 'corpus.tfidf.105.csv'


def main():
    matrix = []
    labels = []
    with open(filename) as f:
        csv_file = csv.reader(f)
        header = next(csv_file)[1:]
        for row in csv_file:
            labels.append(row[0])
            matrix.append(np.array(row[1:], dtype=np.float_))
    matrix = np.array(matrix).T
    unique_labels = set(labels)
    labels = np.array(labels, dtype=np.str_)
    label_binaries = {
        l: labels == l for l in unique_labels
    }
    print(matrix)

    progress = ProgressBar(max_value=len(header)*len(unique_labels))
    correlations = []
    index = 0
    for i, feat_vector in enumerate(matrix):
        for class_label in unique_labels:
            corr = np.correlate(label_binaries[class_label],
                                feat_vector)
            correlations.append((class_label, header[i], corr[0]))

            progress.update(index)
            index += 1

    progress.finish()

    print('Sorting features by mutual information')
    correlations.sort(key=lambda x: x[-1], reverse=True)
    print('First 100 max relevant features:')
    pprint.pprint(correlations[:100])

    print('Writing out to file')
    with open('class_to_features.tfidf.sorted.txt', 'w') as f:
        progress = ProgressBar(max_value=len(correlations))
        for i, corr in enumerate(correlations):
            f.write('{0: >15} {1: <15}\t{2}\n'.format(*corr))
            progress.update(i)

        progress.finish()


def mutual_information(x, y, unique_classes):
    mi = {}
    y_ent = continuous_entropy(y)
    for cls in unique_classes:
        binary_classes = x == cls
        return discrete_entropy(binary_classes, unique_classes)\
             + y_ent - joint_entropy(binary_classes, y)


def joint_entropy(x, y):
    probs = []
    for c1 in set(x):
        for c2 in set(y):
            probs.append(np.mean(
                np.logical_and(x == c1, y == c2)
            ))

    return np.sum(-p * np.log(p) for p in probs)


def continuous_entropy(x):
    pdf = rv_continuous.pdf()


def discrete_entropy(x, possible_values):
    probs = [np.mean(x == c) for c in possible_values]
    return np.sum(-p * np.log(p) for p in probs)


class TfIdfDistribution(rv_continuous):
    "Gaussian distribution"
    def __init__(self, training, *args, **kwargs):
        self.discrete_observations = np.array(training)
        self.n = len(training)
        super(TfIdfDistribution, self).__init__(*args, **kwargs)

    def _cdf(self, x):
        single_x = x[0]
        observations_less_than = self.discrete_observations <= single_x
        percent_less_than = np.sum(observations_less_than) / self.n

        print('RV X =', x, single_x)
        print('Discrete observations :=', self.discrete_observations)
        print('Observatiosn <= {0}\t{1}'.format(single_x, observations_less_than))
        print('Num observations <=', np.sum(observations_less_than))
        print('Total observations =', self.n)
        print('Percentage =', percent_less_than)
        print('-' * 80)

        return percent_less_than


def test_distribution():
    values = np.random.rand(30000)
    distribution = TfIdfDistribution(training=values,
                                     a=0,
                                     name='Feature')
    max_val = max(values)

    print('Maximum value: {}'.format(max_val))
    pdf = distribution.pdf(x=max_val)
    print('PDF at {0}: {1}'.format(values[0], pdf))


if __name__ == '__main__':
    main()

