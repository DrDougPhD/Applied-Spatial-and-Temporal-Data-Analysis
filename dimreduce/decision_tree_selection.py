import pprint
import numpy as np
import csv
import utils
import sys
from progressbar import ProgressBar


def node_impurity(targets, records):
    num_records = len(records)
    probabilities = []
    classes = sorted(set(targets))
    for cls in classes:
        records_of_class = records[targets == cls]
        num_records_matching = len(records_of_class)

        print('\tClass {0} records: {1} out of {2} ({3} percent)'.format(
            cls, num_records_matching, num_records,
            100*num_records_matching/num_records))

        probabilities.append( (num_records_matching/num_records)**2 )

    summed_impurities = 1 - np.sum(probabilities)

    print('\tSquared Probabilities:', probabilities)
    print('\tImpurity:', summed_impurities)
    print('\t' + '.'*40)
    return summed_impurities


@utils.pickled('filename', 'n')
def select_by_dectree_gini_splitting(n, filename):
    scores = decision_tree_selection(filename=filename)
    return scores[:n]


@utils.pickled('filename')
def decision_tree_selection(filename):
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

    progress = ProgressBar(max_value=len(header)*len(unique_labels))
    index = 0

    feature_scores = []
    for i, feat_vector in enumerate(matrix):
        print('Considering Attribute "{}"'.format(header[i]))
        observed_attr_values = sorted(set(feat_vector))

        split_impurities = []
        percent_incoming = []
        for attr_val in observed_attr_values:
            # isolate articles that have a value greater than or eq to a
            # specific value for the given attribute
            feats_gt_mask = feat_vector >= attr_val
            records_with_attr_val_gt = feat_vector[feats_gt_mask]
            labels_of_attr_val_records_gt = labels[feats_gt_mask]

            print('Number of articles with "{0}" >= {1}: {2} out of {3}'.format(
                header[i], attr_val, np.sum(feats_gt_mask), article_count
            ))

            attr_val_impurity_gt = node_impurity(
                targets=labels_of_attr_val_records_gt,
                records=records_with_attr_val_gt)

            # isolate articles that have a value less than a specific value
            # for the given attribute
            feats_lt_mask = feat_vector < attr_val
            records_with_attr_val_lt = feat_vector[feats_lt_mask]
            labels_of_attr_val_records_lt = labels[feats_lt_mask]

            print('Number of articles with "{0}" <= {1}: {2} out of {3}'.format(
                header[i], attr_val, np.sum(feats_lt_mask), article_count
            ))

            attr_val_impurity_lt = node_impurity(
                targets=labels_of_attr_val_records_lt,
                records=records_with_attr_val_lt)

            score_of_split = (len(records_with_attr_val_gt)/article_count\
                           * attr_val_impurity_gt)\
                           + (len(records_with_attr_val_lt)/article_count\
                           * attr_val_impurity_lt)

            split_impurities.append((score_of_split, attr_val))

            print('Splitting on {0} >= {1}, {0} < {1}: {2}'.format(
                header[i], attr_val, score_of_split
            ))
            print('='*60)

            progress.update(index)
            index += 1

        # gini index
        split_with_min_score, attr_val_split = min(split_impurities,
                                                   key=lambda x: x[0])
        print('Gini Index splitting on {0} (>= and < {1}): {2}'.format(
            header[i], attr_val_split, split_with_min_score))
        feature_scores.append((header[i], split_with_min_score))
        print('#'*80)

    progress.finish()

    print('Sorting features by dectree score')
    feature_scores.sort(key=lambda x: x[-1])
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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[-1]

    else:
        filename = 'corpus.tfidf.105.csv'# 'dectreesample.csv'
        # #'corpus.tfidf.105.csv'

    n = 100
    best_feats = select_by_dectree_gini_splitting(n=n, filename=filename)

    scores_file = 'feat_n_scores.dectree.100arts.{}terms.txt'.format(n)
    with open(scores_file, 'w') as f:
        f.write('\n'.join(
            '{feat} & {score} \\\\'.format(feat=feat, score=score)
            for feat, score in best_feats
        ))

    output = 'dectree.tfidf.100articles.{n}terms.txt'.format(n=n)
    with open(output, 'w') as f:
        f.write('\n'.join(map(lambda x: x[0], best_feats)))


