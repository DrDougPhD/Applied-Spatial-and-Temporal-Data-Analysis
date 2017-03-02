import os
import pickle
import csv
import logging

import numpy as np
import pprint


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_STORAGE = os.path.join(CURRENT_DIR, 'cache')

CSV_STORAGE = os.path.join(CACHE_STORAGE, 'csv')
PICKLE_STORAGE = os.path.join(CACHE_STORAGE, 'pickles')
os.makedirs(PICKLE_STORAGE, exist_ok=True)
os.makedirs(CACHE_STORAGE, exist_ok=True)

logger = logging.getLogger(__name__)


def pickled(func):
    def func_wrapper(*args, **kwargs):
        pickle_filename = 'pickle.{fn}.args={args}.kwargs={kwargs}.bin'.format(
            fn=func.__name__,
            args=str(args),
            kwargs=str(kwargs))
        pickle_path = os.path.join(PICKLE_STORAGE, pickle_filename)

        try:
            with open(pickle_path, 'rb') as pkl:
                result = pickle.load(pkl)

        except:
            logger.exception('No pickle for {0} at "{1}".'
                             ' It will be created after execution.'.format(
                                func.__name__, pickle_path))
            result = func(*args, **kwargs)

            with open(pickle_path, 'wb') as pkl:
                pickle.dump(result, pkl)

        return result
    
    return func_wrapper


def csvcached(func):
    def func_wrapper(*args, **kwargs):
        filename = 'cache.{fn}.args={args}.kwargs={kwargs}.csv'.format(
            fn=func.__name__,
            args=str(args),
            kwargs=str(kwargs))
        filepath = os.path.join(CACHE_STORAGE, filename)

        try:
            with open(filepath, 'r') as file:
                csvfile = csv.DictReader(file)
                result = [row for row in csvfile]

        except:
            logger.exception('No csv cache for {0} at "{1}".'
                             ' It will be created after execution.'.format(
                                func.__name__, filepath))
            result = func(*args, **kwargs)

            # result is expected to be a list of dictionaries, or an iterable
            # object with each item capable of acting as a dictionary
            with open(filepath, 'w') as file:
                csvfile = csv.DictWriter(file, fieldnames=result[0].keys())
                csvfile.writeheader()
                csvfile.writerows(result)

        return result
    
    return func_wrapper


###############
### Example ###
###############

#@pickled
@csvcached
def run():
    row = lambda first, second, third: dict(first=first,
                                            second=second,
                                            third=third)
    result = [row(x,y,z) for x,y,z in np.random.rand(10, 3)]
    logger.debug('\n'+pprint.pformat(result))
    return result


def setup_logger():
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    line_numbers_and_function_name = logging.Formatter(
        "%(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ]"
        "%(message)s")
    ch.setFormatter(line_numbers_and_function_name)
    logger.addHandler(ch)


def main():
    # When a function completes, its return value should be output to a 
    # csv so that it can be loaded from csv in the next script.
    # If data is not passed as input to a function, then it should be loaded
    # from a csv of an expected filename.
    setup_logger()

    logger.info('Testing cache decorator')
    run()


if __name__ == '__main__':
    main()

