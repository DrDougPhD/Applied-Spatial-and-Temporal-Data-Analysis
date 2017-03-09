import csv
import pprint
import numpy


filename = 'corpus.tfidf.105.csv'


def main():
    matrix = []
    labels = []
    with open(filename) as f:
        csv_file = csv.reader(f)
        header = next(csv_file)[1:]
        for row in csv_file:
            labels.append(row[0])
            matrix.append(numpy.array(row[1:], dtype=numpy.float_))
    matrix = numpy.array(matrix).T
    unique_labels = set(labels)
    labels = numpy.array(labels, dtype=numpy.str_)
    label_binaries = {
        l: labels == l for l in unique_labels
        }

    # rank words in terms of their average tf-idf scores
    avg_feature_scores = []
    for feature_scores, feature_name in zip(matrix,
                                            header):
        # logger.debug(
        #     'Feature scores {0}: {1}'.format(feature_name, feature_scores))
        non_zero_scores = numpy.sum(feature_scores > 0)
        avg_feature_score = numpy.sum(feature_scores) / non_zero_scores
        avg_feature_scores.append((feature_name,
                                   non_zero_scores,
                                   avg_feature_score))

    avg_feature_scores.sort(key=lambda x: x[2], reverse=True)
    print('Average feature scores:')
    pprint.pprint(avg_feature_scores)
    with open('tfidf.ordered_score.txt', 'w') as f:
        f.write('{0: >20}\t{1: ^15}\t{2}\n'.format(
            'Feature', 'Document Freq', 'Average TF-IDF score'))
        f.writelines(['{0: >20}\t{1: ^15}\t{2}\n'.format(*x)
                      for x in avg_feature_scores])


if __name__ == '__main__':
    main()
