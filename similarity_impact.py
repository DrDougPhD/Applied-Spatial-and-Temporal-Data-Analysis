from surprise import Dataset
from surprise import Reader
from surprise import evaluate
from surprise import print_perf
import os
import pprint

from surprise import SVD # singular value decomposition
from surprise import NMF # non-negative matrix factorization
from surprise import KNNBasic # collaborative filtering

import threading
import time
import statistics
import json

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
    jobs = []
    for i, method in enumerate(methods):
        job = ParallelRecommenders(threadID=i,
                                   counter=1,
                                   fn=method,
                                   data=data)
        job.start()
        jobs.append(job)
        
    
    performances = {}
    for j in jobs:
        j.join()
        performances[j.name] = j.perf
    
    print('='*80)
    pprint.pprint(performances)
    
    print('Writing to file')
    with open('similarities.json', 'w') as f:
        json.dump(performances, f)


class ParallelRecommenders(threading.Thread):
   def __init__(self, threadID, counter, fn, data):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = fn.__doc__
      self.counter = counter
      self.method = fn
      self.params = data
      
   def run(self):
      print("Starting " + self.name)
      self.perf = self.method(self.params)
      print("Exiting " + self.name)


def _collaborative_filtering(data, is_user_based, sim_name):
    algo = KNNBasic(sim_options = {
        'user_based': is_user_based,
        'name': sim_name,
    })
    return evaluate(algo, data,
                    measures=PERF_MEASURES)


def ucf(data):
    """User-based Collaborative Filtering"""
    return {
        'Cosine': _collaborative_filtering(data, True, 'cosine'),
        'MSD': _collaborative_filtering(data, True, 'MSD'),
        'Pearson': _collaborative_filtering(data, True, 'pearson'),
    }


def icf(data):
    """Item-based Collaborative Filtering"""
    return {
        'Cosine': _collaborative_filtering(data, False, 'cosine'),
        'MSD': _collaborative_filtering(data, False, 'MSD'),
        'Pearson': _collaborative_filtering(data, False, 'pearson'),
    }


if __name__ == '__main__':
    main()
