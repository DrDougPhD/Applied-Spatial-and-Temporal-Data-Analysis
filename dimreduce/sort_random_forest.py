import csv
import pprint
import subprocess

if __name__ == '__main__':

    # print('Running R script')
    # subprocess.run(['Rscript', 'random_forest_selection.R'])

    scores = []
    with open('randomforest.tfidf.104.csv') as f:
        csv_file = csv.reader(f)
        header = next(csv_file)
        for row in csv_file:
            feat = row[0]
            if 'X' == feat[0]:
                feat = feat[1:]

            scores.append((feat, float(row[1])))

    scores.sort(key=lambda x: x[-1], reverse=True)

    pprint.pprint(scores[:100])

    print('Writing to file')
    with open('randforest.sorted.tfidf.104.txt', 'w') as f:
        f.write('{0: >15} {1}\n'.format(*header))
        for score in scores:
            f.write('{0: >15} {1}\n'.format(*score))

    output = 'randforest.tfidf.100articles.{n}terms.txt'.format(n=100)
    with open(output, 'w') as f:
        f.write('\n'.join([feat for feat, score in scores[:101]]))

    # with open('corpus.tfidf.105.csv') as f:
    #     header = next(f)
    #
    # with open('features.txt', 'w') as f:
    #     f.write('\n'.join(header.split(',')))
