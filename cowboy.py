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

    perf = svd(data)
    print_perf(perf)


    print('='*80)
    pprint.pprint(perf)


def svd(data):
    return evaluate(SVD(), data,
                    measures=PERF_MEASURES)


def pmf(data):
    return evaluate(SVD(biased=False), data,
                    measures=PERF_MEASURES)


def nmf(data):
    return evaluate(NMF(), data,
                    measures=PERF_MEASURES)

def _collaborative_filtering(data, is_user_based):
    algo = KNNBasic(sim_options = {
        'user_based': is_user_based
    })
    return evaluate(algo, data,
                    measures=PERF_MEASURES)


def ucf(data):
    # user-based collaborative filtering
    return _collaborative_filtering(data, True)


def icf(data):
    # item-based collaborative filtering
    return _collaborative_filtering(data, False)


if __name__ == '__main__':
    main()
