import pprint
import numpy as np
import csv
import collections

from progressbar import ProgressBar
from scipy.special import comb
import utils

import logging
logger = logging.getLogger('cnn.'+__name__)


@utils.pickled('n', 'filename')
def select_correlated_features(n, filename):
    print('Sorting correlations by the number of associated keys')

    dict_of_feature_correlations = collections.defaultdict(dict)

    correlations = correlated_features(filename=filename)
    for attr1, attr2, corr in correlations:
        corr_for_attr = dict_of_feature_correlations[attr1]
        if 'attrs' not in corr_for_attr:
            corr_for_attr['attrs'] = []
            corr_for_attr['corrs'] = []
        corr_for_attr['attrs'].append(attr2)
        corr_for_attr['corrs'].append(corr)

    ordered_list_of_corrs = []
    for attr, associated_attrs in dict_of_feature_correlations.items():
        corrs_to_others = associated_attrs['corrs']

        ordered_list_of_corrs.append((attr, np.sum(corrs_to_others)))

    ordered_list_of_corrs.sort(key=lambda x: x[-1], reverse=True)

    print('Top {} best features:'.format(n))
    pprint.pprint(ordered_list_of_corrs[:n])
    return [x[0] for x in ordered_list_of_corrs[:n]]


@utils.pickled('filename')
def correlated_features(filename):
    matrix = []
    labels = []
    with open(filename) as f:
        csv_file = csv.reader(f)
        header = next(csv_file)[1:]
        for row in csv_file:
            labels.append(row[0])
            matrix.append(np.array(row[1:], dtype=np.float_))
    matrix = np.array(matrix).T

    progress = ProgressBar(max_value=comb(len(header), 2, exact=True))
    correlations = []
    index = 0
    for i, feature_vector in enumerate(matrix):
        for j, other_feat_vector in enumerate(matrix):
            if j <= i:
                continue

            corr = np.correlate(feature_vector, other_feat_vector)
            correlations.append((header[i], header[j], corr[0]))

            progress.update(index)
            index += 1

    logger.debug('Sorting features by correlation')
    correlations.sort(key=lambda x: x[-1], reverse=True)
    logger.debug('First 100 highly redundant features:')
    logger.debug(pprint.pformat(correlations[:100]))

    with open('features.tfidf.sorted_correlations.txt', 'w') as f:
        progress = ProgressBar(max_value=len(correlations))
        for i, corr in enumerate(correlations):
            f.write('{0: >15} {1: <15}\t{2}\n'.format(*corr))
            progress.update(i)

        progress.finish()

    return correlations


if __name__ == '__main__':
    highest_correlated = select_correlated_features(
        n=100, filename='corpus.tfidf.104.csv')
    pprint.pprint(highest_correlated)
    with open('correlated_features.tfidf.100articles.100terms.txt', 'w') as f:
        f.write('\n'.join(highest_correlated))


# import csv
# import pprint
#
# import numpy
# import sys
# import os
#
# from scipy.special import comb
# from progressbar import ProgressBar
#
#
#
#
# coinfo_csv = 'coinformation.tfidf.105.csv'
# assert os.path.isfile(coinfo_csv), 'No file at {}'.format(coinfo_csv)
#
# print('Loading file')
# # coinfo = numpy.genfromtxt(coinfo_csv,
# #                           converters={0: lambda x: x.decode()})
# pairwise_coinfo = []
# with open(coinfo_csv) as f:
#     reader = csv.reader(f)
#     header = next(reader)[1:]
#     num_features = len(header)
#     print(header)
#
#     progress = ProgressBar(max_value=comb(num_features, 2, exact=True))
#     index = 0
#     for i, row in enumerate(reader):
#         #print(i)
#         for j in range(i+1, num_features):
#             #print(header[i], header[j], row[j])
#             # print('{0: >15} {1: <15}\t{2}'.format(header[i],
#             #                                       header[j],
#             #                                       row[j]))
#             pair_ci = (float(row[j]) if row[j] != 'NA' else float('-inf'),
#                        header[i],
#                        header[j])
#             if pair_ci[0] == float('inf'):
#                 continue
#
#             pairwise_coinfo.append(pair_ci)
#             progress.update(index)
#             index += 1
#
#     progress.finish()
#
# print('Sorting...')
# pairwise_coinfo.sort(key=lambda x: x[0], reverse=True)
#
# print('Writing to file')
# with open('sorted_coinfo.txt', 'w') as f:
#     for coinfo, f1, f2 in pairwise_coinfo:
#         f.write('{0:.5f}\t{1: >15} {2: <15}\n'.format(coinfo, f1, f2))
# #pprint.pprint(pairwise_coinfo[:100])
