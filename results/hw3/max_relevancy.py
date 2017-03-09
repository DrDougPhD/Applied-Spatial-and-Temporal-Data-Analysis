import csv
import pprint

import numpy
import sys
import os

from scipy.special import comb
from progressbar import ProgressBar

coinfo_csv = 'coinformation.tfidf.105.csv'
assert os.path.isfile(coinfo_csv), 'No file at {}'.format(coinfo_csv)

print('Loading file')
# coinfo = numpy.genfromtxt(coinfo_csv,
#                           converters={0: lambda x: x.decode()})
pairwise_coinfo = []
with open(coinfo_csv) as f:
    reader = csv.reader(f)
    header = next(reader)[1:]
    num_features = len(header)
    print(header)

    progress = ProgressBar(max_value=comb(num_features, 2, exact=True))
    index = 0
    for i, row in enumerate(reader):
        #print(i)
        for j in range(i+1, num_features):
            #print(header[i], header[j], row[j])
            # print('{0: >15} {1: <15}\t{2}'.format(header[i],
            #                                       header[j],
            #                                       row[j]))
            pairwise_coinfo.append(
                (float(row[j]) if row[j] != 'NA' else float('-inf'),
                 header[i],
                 header[j]))

            progress.update(index)
            index += 1

    progress.finish()

print('Sorting...')
pairwise_coinfo.sort(lambda x: x[0], reverse=True)

print('Writing to file')
with open('sorted_coinfo.txt', 'w') as f:
    for coinfo, f1, f2 in pairwise_coinfo:
        f.write('{0:.f5}\t{1: >15} {2: <15}\n'.format(coinfo, f1, f2))
#pprint.pprint(pairwise_coinfo[:100])