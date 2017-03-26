from surprise import Dataset
from surprise import Reader
from surprise import evaluate
from surprise import print_perf
import os
import pprint

from surprise import SVD # singular value decomposition
from surprise import NMF # non-negative matrix factorization
from surprise import KNNBasic # collaborative filtering


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
        print('='*80)
        print(method.__doc__)
        print('~'*len(method.__doc__))

        perf = method(data)
        performances[method.__doc__] = perf

    print('='*80)
    pprint.pprint(performances)


def svd(data):
    """Singular Value Decomposition"""
    return evaluate(SVD(), data,
                    measures=PERF_MEASURES)


def pmf(data):
    """Probabilistic Matrix Factorization"""
    return evaluate(SVD(biased=False), data,
                    measures=PERF_MEASURES)


def nmf(data):
    """Non-negative Matrix Factorization"""
    return evaluate(NMF(), data,
                    measures=PERF_MEASURES)

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
