import os

NUM_ARTICLES = 100

_current_directory = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_current_directory,
                        'data', 'downloads')


RESULTS_DIR = os.path.join(_current_directory,
                           'results', 'hw3')
SELECTED_ARTICLE_ARCHIVE = os.path.join(RESULTS_DIR,
                                        'articles')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SELECTED_ARTICLE_ARCHIVE, exist_ok=True)


CACHE_STORAGE = os.path.join(_current_directory, 'cache')
CSV_STORAGE = os.path.join(CACHE_STORAGE, 'csv')
PICKLE_STORAGE = os.path.join(CACHE_STORAGE, 'pickles')

os.makedirs(CACHE_STORAGE, exist_ok=True)
os.makedirs(PICKLE_STORAGE, exist_ok=True)
os.makedirs(CSV_STORAGE, exist_ok=True)

PICKLING_ENABLED = True
UPDATE_PICKLES = False

VECTORIZER_METHOD = 'tf'
