import os
_current_directory = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_current_directory,
                        'data', 'downloads')


RESULTS_DIR = os.path.join(_current_directory,
                           'results', 'hw3')
SELECTED_ARTICLE_ARCHIVE = os.path.join(RESULTS_DIR,
                                        'articles')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SELECTED_ARTICLE_ARCHIVE, exist_ok=True)


NUM_ARTICLES = 100