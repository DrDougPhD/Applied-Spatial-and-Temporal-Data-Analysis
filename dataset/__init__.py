import os

import shutil

from . import load
from .articles import ArticleSelector
import logging

logger = logging.getLogger(__name__)


def get(from_, n=10, archive_to=None, randomize=True):
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

    if archive_to is not None:
        logger.info('Copying {0} articles to {1}'.format(
            len(selected_articles),
            archive_to
        ))
        os.makedirs(archive_to, exist_ok=True)
        for article in selected_articles:
            shutil.copy(article.path, archive_to)

    return selected_articles