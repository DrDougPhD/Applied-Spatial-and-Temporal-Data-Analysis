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

    methods = [svd, pmf, nmf, ucf, icf]
    performances = {}
    for method in methods:
        print('Method:', method.__doc__)
        performances[method.__doc__] = method(data)
    
    print('='*80)
    pprint.pprint(performances)
    
    print('Writing to file')
    with open('results.json', 'w') as f:
        json.dump(performances, f)


def svd(data):
    """Singular Value Decomposition"""
    recommender = SVD()
    results =  evaluate(recommender, data,
                        measures=PERF_MEASURES)
    print('SVD internals')
    print(dir(recommender))
    print(recommender)
    return results


def pmf(data):
    """Probabilistic Matrix Factorization"""
    recommender = SVD(biased=False)
    results = evaluate(recommender, data,
                       measures=PERF_MEASURES)
    return results


def nmf(data):
    """Non-negative Matrix Factorization"""
    recommender = NMF()
    results = evaluate(recommender, data,
                       measures=PERF_MEASURES)
    return results

def _collaborative_filtering(data, is_user_based):
    algo = KNNBasic(sim_options = {
        'user_based': is_user_based
    })
    return evaluate(algo, data,
                    measures=PERF_MEASURES)


def ucf(data):
    """User-based Collaborative Filtering"""
    return _collaborative_filtering(data, True)


def icf(data):
    """Item-based Collaborative Filtering"""
    return _collaborative_filtering(data, False)


if __name__ == '__main__':
    main()
