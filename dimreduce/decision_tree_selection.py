import pprint
import numpy as np
import csv

import sys
from progressbar import ProgressBar


def node_impurity(targets, records):
    num_records = len(records)
    probabilities = []
    for cls in set(targets):
        records_of_class = records[targets == cls]
        num_records_matching = len(records_of_class)

        # print('\t{0} records: {1} out of {2} ({3} percent)'.format(
        #     cls, num_records_matching, num_records,
        #     100*num_records_matching/num_records))

        probabilities.append( (num_records_matching/num_records)**2 )

    summed_impurities = 1 - np.sum(probabilities)

    # print('\tSquared Probabilities:', probabilities)
    # print('\tImpurity:', summed_impurities)
    # print('.'*40)
    return summed_impurities


def main(filename):
    matrix = []
    labels = []
    with open(filename) as f:
        csv_file = csv.reader(f)
        header = next(csv_file)[1:]
        for row in csv_file:
            labels.append(row[0])
            matrix.append(np.array(row[1:], dtype=np.float_))

    article_count = len(labels)
    matrix = np.array(matrix).T
    unique_labels = set(labels)
    labels = np.array(labels, dtype=np.str_)
    label_binaries = {
        l: labels == l for l in unique_labels
    }
    print(matrix)

    progress = ProgressBar(max_value=len(header)*len(unique_labels))
    index = 0

    print('### Calculating Gini Scores')
    feature_scores = []
    for i, feat_vector in enumerate(matrix):
        observed_attr_values = set(feat_vector)

        #print('Base impurity for feature', header[i])
        base_impurity = node_impurity(targets=labels,
                                      records=feat_vector)

        split_impurities = []
        percent_incoming = []
        for attr_val in observed_attr_values:
            attr_val_mask = feat_vector == attr_val

            # print('Number of articles with "{0}" == {1}: {2} out of {3}'.format(
            #     header[i], attr_val, np.sum(attr_val_mask), article_count
            # ))

            records_with_attr_val = feat_vector[attr_val_mask]
            labels_of_attr_val_records = labels[attr_val_mask]

            attr_val_impurity = node_impurity(targets=labels_of_attr_val_records,
                                              records=records_with_attr_val)

            split_impurities.append(attr_val_impurity)
            percent_incoming.append(len(records_with_attr_val)/article_count)

            progress.update(index)
            index += 1

        # gini index
        score = base_impurity - np.sum(percent_incoming[j]
                                       * np.sum(split_impurities[j])
                                       for j in range(len(observed_attr_values)))
        #print('Gini Index splitting on {0}: {1}'.format(header[i], score))
        feature_scores.append((header[i], score))
        #print('-'*80)

    progress.finish()

    print('Sorting features by dectree score')
    feature_scores.sort(key=lambda x: x[-1], reverse=True)
    print('First 100 max score features:')
    pprint.pprint(feature_scores[:100])

    print('Writing out to file')
    with open('dectree_scores.tfidf.sorted.txt', 'w') as f:
        progress = ProgressBar(max_value=len(feature_scores))
        for i, score in enumerate(feature_scores):
            f.write('{0: >15}\t{1}\n'.format(*score))
            progress.update(i)

        progress.finish()

    return feature_scores


def cond_prob(x, y):
    pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[-1]

    else:
        filename = 'dectreesample.csv' #'corpus.tfidf.105.csv'

    main(filename)

