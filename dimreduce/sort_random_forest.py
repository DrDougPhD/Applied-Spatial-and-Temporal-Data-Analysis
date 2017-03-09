import sys
import csv
import pprint

scores = []
with open(sys.argv[-1]) as f:
    csv_file = csv.reader(f)
    header = next(csv_file)
    for row in csv_file:
        scores.append((row[0], float(row[1])))

scores.sort(key=lambda x: x[-1], reverse=True)

pprint.pprint(scores[:100])

print('Writing to file')
with open('randforest.sorted.tfidf.105.txt', 'w') as f:
    f.write('{0: >15} {1}\n'.format(*header))
    for score in scores:
        f.write('{0: >15} {1}\n'.format(*score))
