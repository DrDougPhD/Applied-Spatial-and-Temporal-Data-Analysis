import pprint
import numpy as np
import csv
import collections

from progressbar import ProgressBar
from scipy.special import comb
from scipy.stats import pearsonr

import utils

import logging
logger = logging.getLogger('cnn.'+__name__)


@utils.pickled('n', 'filename')
def select_correlated_features(n, filename):
    correlations, avg_tfidfs = correlated_features(filename=filename)

    print('Beginning feature removal')
    blacklist = set()
    dict_of_feature_correlations = collections.defaultdict(dict)
    for attr1, attr2, corr in correlations:
        if attr1 in blacklist or attr2 in blacklist:
            continue

        lower_scoring_attr = min([attr1, attr2],
                                 key=lambda attr: avg_tfidfs[attr])
        higher_scoring_attr = max([attr1, attr2],
                                 key=lambda attr: avg_tfidfs[attr])

        blacklist.add(lower_scoring_attr)

        corr_for_attr = dict_of_feature_correlations[higher_scoring_attr]
        if 'corrs' not in corr_for_attr:
            corr_for_attr['corrs'] = []
        corr_for_attr['corrs'].append(corr)
    print('Feature removal complete')


    ordered_list_of_corrs = []
    for attr, associated_attrs in dict_of_feature_correlations.items():
        corrs_to_others = associated_attrs['corrs']

        ordered_list_of_corrs.append((attr, np.sum(corrs_to_others),
                                      avg_tfidfs[attr]))

    print('Sorting features by sum of correlations to others')
    ordered_list_of_corrs.sort(key=lambda x: x[1])

    print('Writing feats/correlation sum/average tfidf to file')
    with open('features.tfidf.avgcorr_and_tfidf.txt', 'w') as f:
        f.write('{0: >20}\t{1: <7}\t{2: <7}\n'.format('Attribute',
                                                      'AvgCorr',
                                                      'AvgTfidf'))
        print('{0: >20}\t{1: <7}\t{2: <7}'.format('Attribute',
                                                      'AvgCorr',
                                                      'AvgTfidf'))
        for attr, avg_corr, avg_tfidf in ordered_list_of_corrs:
            line = '{0: >20}\t & {1: <.5f}\t & {2: <.5f} \\\\ \n'.format(
                attr, avg_corr, avg_tfidf
            )
            print(line[:-1])
            f.write(line)

    print('Top {0} best features out of {1}:'.format(
        n, len(ordered_list_of_corrs)))
    pprint.pprint(ordered_list_of_corrs[:n])
    return [x[0] for x in ordered_list_of_corrs[:n]]


@utils.pickled('filename')
def correlated_features(filename):
    matrix = []
    labels = []
    with open(filename) as f:
        csv_file = csv.reader(f)
        header = next(csv_file)[1000:2000]
        for row in csv_file:
            labels.append(row[0])
            matrix.append(np.array(row[1000:2000], dtype=np.float_))
    matrix = np.array(matrix).T

    print('Computing pairwise correlation (drink a beer in the meantime...)')
    progress = ProgressBar(max_value=comb(len(header), 2, exact=True))
    correlations = []
    average_tfidfs = {}
    index = 0
    for i, feature_vector in enumerate(matrix):
        non_zero_mask = feature_vector > 0
        average_tfidfs[header[i]] = np.mean(feature_vector[non_zero_mask])
        for j, other_feat_vector in enumerate(matrix):
            if j <= i:
                continue

            # ignore instances where both are 0
            mask = np.logical_or(non_zero_mask,
                                 other_feat_vector > 0)
            print(np.sum(mask))
            if int(np.sum(mask)) == 0:
                pass

            else:
                corr, pval = pearsonr(feature_vector[mask], other_feat_vector[mask])
                correlations.append((header[i], header[j], corr))

            progress.update(index)
            index += 1

    progress.finish()

    logger.debug('Sorting features by correlation')
    correlations.sort(key=lambda x: x[-1], reverse=True)
    logger.debug('First 100 highly redundant features:')
    logger.debug(pprint.pformat(correlations[:100]))

    print('Writing out correlations to file')
    with open('features.tfidf.sorted_correlations.txt', 'w') as f:
        progress = ProgressBar(max_value=len(correlations))
        for i, corr in enumerate(correlations):
            f.write('{0: >15} {1: <15}\t{2}\n'.format(*corr))
            progress.update(i)

        progress.finish()

    return correlations, average_tfidfs


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
