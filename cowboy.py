import collections

import itertools
import pprint

import numpy
from matplotlib import pyplot
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

import kmeans
import metrics
import sim_matrix_heatmap
import utils
logger = utils.setup_logger('cnn')

import config
import preprocess
import dimreduce


try:
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

import dataset
import os


def main():
    logger.info(hr('Loading Articles'))
    articles = dataset.load_dataset(n=config.NUM_ARTICLES,
                                    from_=config.DATA_DIR,
                                    randomize=True)

    distance_func = distance.cosine
    feature_subset_selection_methods = {
        'Feature Correlation': 'correlated_features.tfidf.100articles.100terms.txt',
        'Feature Relevancy': 'maxrel.tfidf.100articles.100terms.txt',
        'Average TF-IDF': 'avgtfidf.tfidf.100articles.100terms.txt',
        'Random Forest': 'randforest.tfidf.100articles.100terms.txt',
        'Gini Split Gain': 'dectree.tfidf.100articles.100terms.txt',
    }
    for k in feature_subset_selection_methods:
        feature_subset_selection_methods[k] = os.path.join(
            'dimreduce', feature_subset_selection_methods[k])

    fig, axes = pyplot.subplots(nrows=3, ncols=2)
    sse_axes = axes[0,0]
    sse_axes.set_ylabel('SSE')
    sil_axes = axes[1,0]
    sil_axes.set_ylabel('Silhouette\nCoefficient')
    ideal_correlation_axes = axes[2,0]
    ideal_correlation_axes.set_ylabel('Ideal\nCorrelation')

    ticks = []
    index = 1
    feature_subset_method = 'Feature Correlation'


    logger.info(hr('Vectorizing Corpus'))
    feature_selection_file = os.path.join(
        'dimreduce',
        'correlated_features.tfidf.100articles.100terms.txt')
    corpus = preprocess.preprocess(corpus=articles,
                                   exclude_stopwords=True,
                                   feature_subset=feature_selection_file,
                                   method=config.VECTORIZER_METHOD)

    logger.info(hr('K-Means Clustering'))
    articles_np = numpy.array(corpus.corpus)
    clustering, centroids = kmeans.kmeans(vectors=corpus.matrix.toarray(),
                                          k=7,
                                          distance=distance_func,
                                          initial_centroid_method='kpp')

    clusters = []
    for cluster_index, article_indices in enumerate(clustering):
        print('{0}: {1}'.format(cluster_index, article_indices))
        clusters.append(list(articles_np[article_indices]))

    articles_sorted_by_cluster = []
    article_cluster_indices = []
    for cluster_index, cluster in enumerate(clusters):
        # sort the cluster based on category labels
        cluster.sort(key=lambda a: a.category)

        # flatten cluster
        articles_sorted_by_cluster.extend([article.vector
                                           for article in cluster])
        article_cluster_indices.extend([cluster_index
                                        for _ in range(len(cluster))])

        logger.debug('{0: >5}:'.format(cluster_index))
        for article in cluster:
            logger.debug('    {0: >15} -- {1}'.format(article.category,
                                                      article.title))

    indices = numpy.arange(corpus.count)
    cart_product_indices = itertools.product(indices,
                                             repeat=2)

    logger.info(hr('Similarity / Distance Matrix'))
    # sim_matrix_heatmap.plot(sorted_matrix=articles_sorted_by_cluster,
    #                         distance_metric=distance.cosine,
    #                         cart_prod_indices=cart_product_indices)

    ## Ideal Cluster to Ideal Class Similarity Matrix correlation
    logger.info(hr('Calculating Ideal Similarity Correlation'))
    article_class_indices = [corpus.class_to_index[article.category]
                             for article in corpus]
    correlation = metrics.ideal_correlation(
        cluster_indices=article_cluster_indices,
        class_indices=article_class_indices,
        n=corpus.count)
    logger.info('Ideal Correlation: {}'.format(correlation))
    ideal_correlation_axes.bar(index, correlation)

    ## SSE
    logger.debug(hr('Calculating SSE'))
    sse = metrics.calculate_sse(centroids,
                                clustering,
                                corpus.matrix,
                                distance_func)
    logger.debug('SSE: {}'.format(sse))
    sse_axes.bar(index, sse)

    ## Silhouette coefficient
    logger.info(hr('Calculating Silhouette Coefficient'))
    silhouette = metrics.silhouette_coeff(clustering,
                                          corpus.matrix.toarray(),
                                          distance_func)
    logger.debug('Silhouette Coefficient: {}'.format(silhouette))
    sil_axes.bar(index, silhouette)

    ticks.append((index, feature_subset_method))


    axes[2,0].set_xticks([t[0] for t in ticks])
    axes[2, 0].set_xticklabels([t[1] for t in ticks])

    pyplot.tight_layout(h_pad=2.0)
    pyplot.show()

if __name__ == '__main__':
    main()

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

