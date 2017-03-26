import csv
import pprint


with open('results.csv') as f:
    reader = csv.DictReader(f, delimiter=';')
    for line in reader:
        pprint.pprint(line)
