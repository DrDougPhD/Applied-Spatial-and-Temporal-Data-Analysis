from surprise import Dataset
from surprise import Reader
from surprise import evaluate
from surprise import print_perf
import os
import pprint

from surprise import KNNBasic # collaborative filtering

import threading
import time
import statistics
import json
import collections

exitFlag = 0

PERF_MEASURES = ['RMSE', 'MAE']

#load data from a file
FILE_PATH = os.path.join('data', 'restaurant_ratings.txt')
FILE_READER = Reader(line_format='user item rating timestamp', sep='\t')


def main():
    data = Dataset.load_from_file(FILE_PATH,
                                  reader=FILE_READER)
    data.split(n_folds=3)

    methods = [ucf, icf]
    performances = collections.defaultdict(list)
    for i, method in enumerate(methods):
        print('Running experiments for', method.__doc__)
        print('-'*40)
        for k in range(2, 101):
            print('k =', k)
            results = method(data, k)
            performances[method.__doc__].append(results)
            print('-'*20)
        
    
    print('='*80)
    pprint.pprint(performances)
    
    print('Writing to file')
    with open('kvaried.json', 'w') as f:
        json.dump(performances, f)


class ParallelRecommenders(threading.Thread):
   def __init__(self, threadID, counter, fn, data, k):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = fn.__doc__
      self.counter = counter
      self.method = fn
      self.params = data
      self.k = k
      
   def run(self):
      print("Starting {0} (k={1}".format(self.name, self.k))
      self.perf = (self.k, self.method(self.params, self.k))
      print("Exiting  {0} (k={1}".format(self.name, self.k))

def _collaborative_filtering(data, is_user_based, kneighbors):
    algo = KNNBasic(k=kneighbors,
                    sim_options = {
        'user_based': is_user_based,
        'name': 'MSD',
    })
    return evaluate(algo, data,
                    measures=PERF_MEASURES)


def ucf(data, k):
    """User-based Collaborative Filtering"""
    return _collaborative_filtering(data, True, k)


def icf(data, k):
    """Item-based Collaborative Filtering"""
    return _collaborative_filtering(data, False, k)


if __name__ == '__main__':
    main()
