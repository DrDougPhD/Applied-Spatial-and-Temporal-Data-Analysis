import pprint
import numpy as np
import csv
from progressbar import ProgressBar

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

    return correlations


if __name__ == '__main__':
    main()

