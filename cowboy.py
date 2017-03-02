import os
import logging
import numpy
import shutil
from matplotlib import pyplot
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import SparsePCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.random_projection import SparseRandomProjection

import config
import preprocess


def setup_logger(name):
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', name + '.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    ch.setFormatter(logging.Formatter(
        "%(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ] %("
        "message)s"
    ))
    # add the handlers to the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
logger = setup_logger('cnn')

try:
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

import dataset


def main():
    logger.info(hr('Loading Articles'))
    articles = dataset.load_dataset(n=config.NUM_ARTICLES,
                                    from_=config.DATA_DIR,
                                    randomize=True)
    logger.info(hr('Vectorizing Corpus'))
    corpus = preprocess.preprocess(corpus=articles,
                                   exclude_stopwords=True,
                                   method=config.VECTORIZER_METHOD)

#
# class Homework3Experiments(object):
#
#     pickle_file_fmt = 'hw3.{}.pickle'
#
#     def __init__(self, n, dataset_dir, pickling, randomize=True):
#         # load data
#         logger.info('Looking for datasets in {}'.format(dataset_dir))
#         self.n = n
#         self.dataset_dir = dataset_dir
#         self.output_dir = os.path.join('figures', 'kmeans')
#         os.makedirs(self.output_dir, exist_ok=True)
#
#         self.pickling = pickling
#
#         # preprocess
#         self.vectorizers = {
#             'tf': 'Term Frequency',
#             # 'existence': 'Existence',
#             'tfidf': 'TF-IDF'
#         }
#
#         corpus_pickle = 'corpus.{}'.format(n)
#         corpus_by_vectorizer = self._load_pickle(corpus_pickle)
#         if not corpus_by_vectorizer:
#             self.articles = dataset.get(n=n, from_=dataset_dir,
#                                         randomize=randomize)
#
#             corpus_by_vectorizer = {
#                 self.vectorizers[k]: preprocess.execute(corpus=self.articles,
#                                                         exclude_stopwords=True,
#                                                         method=k)
#                 for k in self.vectorizers
#             }
#             self._save_to_pickel(corpus_by_vectorizer, corpus_pickle)
#         self.corpus_by_vectorizer = corpus_by_vectorizer
#
#         self.corpus = self.corpus_by_vectorizer['Term Frequency']
#
#         self.experiment = {}
#
#
#     def _load_pickle(self, filename):
#         if not self.pickling:
#             return False
#
#         pickle_path = os.path.join(
#             self.output_dir,
#             self.pickle_file_fmt.format(filename))
#         if os.path.exists(pickle_path):
#             logger.info('Loading from pickle: {}'.format(pickle_path))
#             pkl = pickle.load(open(pickle_path, 'rb'))
#             return pkl
#
#         return False
#
#     def _save_to_pickel(self, object, filename):
#         pickle_path = os.path.join(
#             self.output_dir,
#             self.pickle_file_fmt.format(filename))
#         pickle.dump(object, open(pickle_path, 'wb'))
#
#
#     def dimensionality_reduction(self):
#         output_path = 'dimreduce'
#         os.makedirs(output_path, exist_ok=True)
#
#         logger.info(hr('Dimensionality Reduction', '+'))
#
#         dataset = self.corpus.matrix.toarray()
#         labels = self.corpus.classes
#         classnames = self.corpus.class_names
#         masks = {cls_idx: labels == cls_idx
#                  for cls_idx in range(len(classnames))}
#         for mask_key in masks:
#             logger.debug('{classname: >15}: {count} articles'.format(
#                 classname=classnames[mask_key],
#                 count=numpy.sum(masks[mask_key])
#             ))
#
#         # check if pickle of transformed data exists
#         pkl_filename = 'dim_reduction_{}'.format(self.n)
#         reduced = self._load_pickle(pkl_filename)
#
#         if not reduced:
#             # map the dataset to 2 dimensions
#             reduced = {}
#
#             # Isomap
#             key = 'Isometric Mapping'
#             logger.debug(hr(key, '.'))
#
#             mapper = Isomap(n_neighbors=5,
#                             n_components=2)
#             reduced[key] = mapper.fit_transform(X=dataset, y=labels)
#             # quick printing of reduced dataset
#             logger.debug('Transformed dataset:')
#             logger.debug(reduced[key])
#             logger.debug('X[0]: {}'.format(reduced[key][0]))
#             logger.debug('y:    {}'.format(reduced[key][1]))
#
#
#             # Local linear embedding (LLE)
#             key = 'Locally linear embedding'
#             logger.debug(hr(key, '.'))
#
#             mapper = LocallyLinearEmbedding(n_neighbors=5,
#                                             n_components=2)
#             reduced[key] = mapper.fit_transform(X=dataset, y=labels)
#             # quick printing of reduced dataset
#             logger.debug('Transformed dataset:')
#             logger.debug(reduced[key])
#             logger.debug('X[0]: {}'.format(reduced[key][0]))
#             logger.debug('y:    {}'.format(reduced[key][1]))
#
#
#             try:
#                 # Spectral Embedding
#                 key = 'Spectral Embedding'
#                 logger.debug(hr(key, '.'))
#
#                 mapper = SpectralEmbedding(n_neighbors=5,
#                                               n_components=2,
#                                               eigen_solver='amg')
#                 reduced[key] = mapper.fit_transform(X=dataset)
#                 # quick printing of reduced dataset
#                 logger.debug('Transformed dataset:')
#                 logger.debug(reduced[key])
#                 logger.debug('X[0]: {}'.format(reduced[key][0]))
#                 logger.debug('y:    {}'.format(reduced[key][1]))
#             except:
#                 pass
#
#
#             # Principal Component Analysis
#             key = 'Principal Component Analysis'
#             logger.debug(hr(key, '.'))
#
#             mapper = SparsePCA(n_components=2)
#             reduced[key] = mapper.fit_transform(X=dataset)
#             # quick printing of reduced dataset
#             logger.debug('Transformed dataset:')
#             logger.debug(reduced[key])
#             logger.debug('X[0]: {}'.format(reduced[key][0]))
#             logger.debug('y:    {}'.format(reduced[key][1]))
#
#
#             # Random Projections
#             key = 'Random Projections'
#             logger.debug(hr(key, '.'))
#
#             mapper = SparseRandomProjection(n_components=2)
#             reduced[key] = mapper.fit_transform(X=dataset)
#             # quick printing of reduced dataset
#             logger.debug('Transformed dataset:')
#             logger.debug(reduced[key])
#             logger.debug('X[0]: {}'.format(reduced[key][0]))
#             logger.debug('y:    {}'.format(reduced[key][1]))
#
#
#             # Feature Agglomeration
#             key = 'Featue Agglomeration'
#             logger.debug(hr(key, '.'))
#
#             mapper = FeatureAgglomeration(
#                 n_clusters=len(classnames))
#             reduced[key] = mapper.fit_transform(X=dataset)
#             # quick printing of reduced dataset
#             logger.debug('Transformed dataset:')
#             logger.debug(reduced[key])
#             logger.debug('X[0]: {}'.format(reduced[key][0]))
#             logger.debug('y:    {}'.format(reduced[key][1]))
#
#
#             self._save_to_pickel(reduced, pkl_filename)
#
#         # plot the reduced dimension
#         # create a color mapping of the labels
#         # colors_mapping = defaultdict(lambda: next(experiments.colors))
#         # colors_mapping = defaultdict(lambda: numpy.random.rand(3,1))
#
#
#
#
#         markers_available = {}
#         colors_available = experiments.get_cmap(len(classnames)+4)
#         colors_assigned = {}
#         for cls in labels:
#             markers_available[cls] = next(experiments.markers)
#             colors_assigned[cls] = colors_available(cls)
#
#         markers = numpy.array([markers_available[cls] for cls in labels])
#         colors = numpy.array([colors_assigned[cls] for cls in labels])
#
#         # for each mapping, create a subplot and plot the results
#         fig, ax = pyplot.subplots(nrows=3, ncols=2)
#         ax.shape = (6,1)
#         fig.set_size_inches((10, 20))
#         legend_handles = []
#         for key, axes in zip(reduced, ax[:,0]):
#             for cls_idx in masks:
#                 class_mask = masks[cls_idx]
#                 dots = axes.scatter(x=reduced[key][:, 0][class_mask],
#                                     y=reduced[key][:, 1][class_mask],
#                                     c=colors[cls_idx],
#                                     marker=markers[cls_idx])
#                 legend_handles.append(dots)
#
#             axes.set_title(key)
#             axes.grid(True)
#
#
#         # legend_handles = [
#         #     patches.Patch(color=colors_available(cls),
#         #                   label=classnames[cls],
#         #                   marker=markers[cls])
#         #     for cls in range(len(classnames))]
#         pyplot.legend(handles=legend_handles,
#                       loc='upper left')
#         pyplot.tight_layout(h_pad=4)
#         pyplot.show()


if __name__ == '__main__':
    main()
