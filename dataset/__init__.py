import os

import utils
from . import load
from .articles import ArticleSelector

@utils.pickled
def load_dataset(from_, n, randomize):
    dataset_dir = load.get_dataset_dir(from_)
    archive_files = load.get_datasets(indir=dataset_dir)
    decompressed_dataset_directories = {}
    for f in archive_files:
        filename = os.path.basename(f)
        filename_prefix = '.'.join(filename.split('.')[:-1])
        extract_to = os.path.join(dataset_dir, filename_prefix)
        load.decompress(f, to=extract_to, dataset_dir=dataset_dir)
        decompressed_dataset_directories[filename] = extract_to

    # randomly select articles
    selector = ArticleSelector(decompressed_dataset_directories)
    selected_articles = selector.get(n, randomize=randomize)
    return selected_articles