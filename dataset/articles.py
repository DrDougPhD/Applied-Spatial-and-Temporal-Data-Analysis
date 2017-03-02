import os
import logging
import glob
import string
import random
from bs4 import BeautifulSoup
from collections import defaultdict

logger = logging.getLogger('cnn.'+__name__)
import utils



try:  # this is my own package, but it might not be present
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30


class ArticleSelector(object):
    """
    Given a dataset of articles, select a subset for further processing.
    Obtain the article's title, category, and plain text.
    """

    class BaseDatasetAccessor(object):
        """
        Base class for accessing articles within a given directory.
        """

        def __init__(self, dir):
            assert os.path.isdir(dir), \
                "Directory doesn't exist: {}".format(dir)
            self.stored_in = dir

        def retrieve(self):
            raise NotImplemented('.retrieve() method has not been implemented')

    class QianDataset(BaseDatasetAccessor):
        """
        Retrieve article files from the Qian CNN dataset located in a specified
        directory.
        """

        def retrieve(self):
            logger.info('Retrieving articles from within {}'.format(
                self.stored_in))
            article_categories_in = os.path.join(self.stored_in, 'Raw')
            categories = os.listdir(article_categories_in)
            logger.info('Category directories: {}'.format(categories))
            articles = []
            for category in categories:
                subcat_files = []
                abspath = os.path.join(article_categories_in, category)
                for article_path in self._retrieve_from_category(abspath):
                    article = QianArticle(path=article_path)
                    if article:
                        subcat_files.append(article)
                    else:
                        article.delete_from_dataset()
                logger.info(
                    'Files within {0}: {1}'.format(category, len(subcat_files)))
                articles.extend(subcat_files)
            return articles

        def _retrieve_from_category(self, category_directory):
            glob_path = os.path.join(category_directory, 'cnn_*.txt')
            files = []
            for filename in glob.glob(glob_path):
                # logger.debug('\t -->  {}'.format(filename))
                files.append(os.path.join(category_directory, filename))
            return files

    article_accessor = {
        # access articles by the file's bytesize
        136208660: lambda dir: ArticleSelector.QianDataset(dir),
    }

    def __init__(self, datasets):
        logger.info(hr('Article Selection'))
        file_sizes = [(file, os.stat(datasets[file] + '.zip').st_size)
                      for file in datasets]
        self.accessors = [ArticleSelector.article_accessor[size](datasets[k])
                          for k, size in file_sizes]

    def get(self, count, randomize=True, equal_dist=True):
        articles = self._load_all_articles()
        logger.info('Dataset size: {}'.format(len(articles)))

        # partition articles by their categories
        logger.info('Partitioning dataset by category')
        partitioning = partition_by_category(articles)
        if equal_dist:
            logger.info('Equal-sized category partitions')
            truncate_to_equal_category_distribution(partitioning)


        # flatten article selection, either randomly sampling from each
        # category or guaranteeing equal samples for each category
        selected_articles = self._select_and_flatten(randomize, equal_dist,
                                                     count, partitioning)

        self._check_category_diversity(selected_articles)
        random.shuffle(selected_articles)
        return selected_articles

    def _select_and_flatten(self, randomize, equal_dist, count, partitioning):
        selected_articles = []
        if randomize:
            random.seed(randomize)
            logger.debug('Random selection of {} articles'.format(count))

            if equal_dist:
                num_articles_per_partition = 1+(count // len(partitioning))
                logger.debug('Equally sampling {} articles per '
                             'category'.format(num_articles_per_partition))
                for key, category_partition in partitioning.items():
                    selected_articles.extend(
                        random.sample(category_partition,
                                      num_articles_per_partition)
                    )
                return selected_articles

            else:
                logger.debug('Random sampling without equal sampling per '
                             'partition')
                for c, category_partition in partitioning.items():
                    selected_articles.extend(category_partition)

                return random.sample(selected_articles, count)

        else:
            logger.debug('Non-random selection of {} articles'.format(count))

            for c, category_partition in partitioning.items():
                selected_articles.extend(partitioning)
                if len(selected_articles) > count:
                    break
            return selected_articles[:count]

    def _load_all_articles(self):
        articles = []
        for selector in self.accessors:
            subset = selector.retrieve()
            articles.extend(subset)
            assert len(articles) > 0, 'No articles found for {}'.format(
                selector.__class__.__name__)
        return articles


    def _check_category_diversity(self, selected_articles):
        categories = set()
        category_counts = defaultdict(int)
        for a in selected_articles:
            categories.add(a.category)
            category_counts[a.category] += 1
        logger.info('Categories chosen: {}'.format(categories))
        logger.info(category_counts)


def partition_by_category(articles):
    categories = set([utils.get_category(p.filename)
                      for p in articles])

    partitions = {c: [] for c in categories}
    for f in articles:
        category_of_art = utils.get_category(f.filename)
        partitions[category_of_art].append(f)

    logger.debug('Articles per category:')
    for key, category_partition in partitions.items():
        logger.debug('{key: >15}: {count} articles'.format(
            key=key,
            count=len(category_partition)
        ))

    return partitions


def truncate_to_equal_category_distribution(partitions):
    # partition articles by class
    min_articles = float('inf')
    for c in partitions:
        article_count = len(partitions[c])
        min_articles = min(article_count, min_articles)

    logger.debug('Smallest category has {} articles.'
                 ' Truncating other categories to this size.'.format(
        min_articles
    ))

    # strip away articles from classes that have more than others
    for c in partitions:
        articles = partitions[c]
        if len(articles) == min_articles:
            continue

        logger.debug('Truncating {0: >15}: {1} to {2}'.format(
            c, len(articles), min_articles
        ))
        partitions[c] = random.sample(articles, min_articles)

    logger.debug('New dataset size: {}'.format(
        sum([len(partitions[c]) for c in partitions])))


class NewspaperArticle(object):
    def __init__(self, path):
        assert os.path.isfile(path), 'File not found: {}'.format(path)
        self.path = path
        self.filename = os.path.basename(path)
        self.title = None
        self.abstract = None
        self.category = None
        self.vector = None
        self.text = None
        self.length = 0
        self._setup_quick_vars()

    def __radd__(self, other):
        return other + self.vector

    def __str__(self):
        """
        Return the plaintext of the article as a string.
        :return: string The plaintext of the article.
        """
        # simply iterate over every word in the document, removing newlines
        # and bad characters, and return as one long string
        return ' '.join(self)

    def __repr__(self):
        return '"{0.title}"\n' \
               '\tcategory: {0.category}\n' \
               '\tvector:   {1}'.format(self, self.vector)

    def __iter__(self):
        """
        Iterate through each word in this article.
        :return: string Next word in the article.
        """
        self._setup_reader()
        # logger.debug('Parsing through {}'.format(
        #     os.path.basename(self.path)))
        for w in self._next_word():
            self.length += 1
            yield w

    def __bool__(self):
        # load file
        self._setup_reader()
        return bool(self.text.strip())

    def delete_from_dataset(self):
        logger.warning('Deleting empty file: {}'.format(self.path))
        os.remove(self.path)

    def __len__(self):
        return self.length


class QianArticle(NewspaperArticle):
    punctuation_remover = str.maketrans('', '', string.punctuation)

    def _setup_quick_vars(self):
        category_dir = os.path.basename(os.path.dirname(self.path))
        self.category = category_dir.split('_')[-1]

    def _setup_reader(self):
        soup = BeautifulSoup(open(self.path), 'html.parser')
        self.title = soup.doc.title.text
        self.abstract = soup.doc.abstract.text
        self.text = soup.doc.find('text').text

    def _next_word(self):
        for line in self.text.split('\n'):
            if self._matches_useless_line(line):
                continue

            if '(CNN)' in line:
                # first line of the article
                prefix_removed = line.split('(CNN)')[-1]
                line = prefix_removed

            for word in line.split():
                # word lowering is done internally by scikit-learn
                # word = word.lower()
                word = word.translate(QianArticle.punctuation_remover)
                if not word or word.isspace():
                    continue

                yield word.lower()

    def _matches_useless_line(self, line):
        if line.startswith('Watch Anderson Cooper'):
            # it's not a diss, Mr. Cooper. But the line containing that
            # text does not pertain to the article's contents
            return True

        return False

