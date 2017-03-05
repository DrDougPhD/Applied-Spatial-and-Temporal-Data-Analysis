import collections

import itertools
import pprint

import numpy
from scipy.spatial import distance
from sklearn.cluster import KMeans

import kmeans
import utils
logger = utils.setup_logger('cnn')

import config
import preprocess


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

    logger.info(hr('K-Means Clustering'))
    articles_np = numpy.array(corpus.corpus)


    # clustered = KMeans(n_clusters=7,
    #                    random_state=1,
    #                    init='random').fit(corpus.matrix)
    clustering, centroids = kmeans.it(vectors=corpus.matrix.toarray(),
                                      k=7,
                                      distance=distance.euclidean,
                                      initial_centroid_method='random')

    print('='*30 + 'Final Clusters' + '='*30)
    clusters = []
    for cluster_index, article_indices in enumerate(clustering):
        print('{0}: {1}'.format(cluster_index, article_indices))
        clusters.append(list(articles_np[article_indices]))

    articles_sorted_by_cluster = []
    for cluster_index, cluster in enumerate(clusters):
        # sort the cluster based on category labels
        cluster.sort(key=lambda a: a.category)

        # flatten cluster
        articles_sorted_by_cluster.extend([
            (article, cluster_index)
            for article in cluster
        ])

        logger.debug('{0: >5}:'.format(cluster_index))
        for article in cluster:
            logger.debug('    {0: >15} -- {1}'.format(article.category,
                                                      article.title))

    logger.info(hr('Similarity / Distance Matrix'))
    # make an index array augmented with the article categories of each article
    # index_category_array = numpy.array([
    #     numpy.arange(len(corpus.count))
    # ])

    # Create a list of index pairs to correspond to entries in the array
    cart_product_of_article_indices = itertools.product(
        numpy.arange(corpus.count), repeat=2)
    #logger.debug(pprint.pformat(list(cart_product_of_article_indices)))

    # Create an array of distance values between each pair
    distances = distance_matrix(matrix=list(map(lambda x: x[0].vector,
                                                articles_sorted_by_cluster)),
                                distance_func=distance.euclidean,
                                n=corpus.count)
    logger.debug(distances)
    # similarities = utils.euclidean_similarities(distances)
    # similarities.shape = (corpus.count, corpus.count)
    #
    #
    # logger.debug('\n{}'.format(similarities))


@utils.pickled('n','distance_func')
def distance_matrix(matrix, distance_func, n):
    return MemoizedDistances(matrix=matrix,
                             distance_func=distance_func)


class MemoizedDistances(object):
    def __init__(self, matrix, distance_func):
        n = len(matrix)
        self.distance_fn = distance_func

        indices = numpy.arange(n)
        self.memoized = {i: {} for i in indices}

        cart_product_indices = itertools.product(indices,
                                                 repeat=2)
        self.distance_matrix = numpy.array([
            self._get_distance(matrix[i], matrix[j],
                               i, j)
            for i, j in cart_product_indices
        ])
        self.distance_matrix.shape = (n, n)

    def _get_distance(self, u, v, u_idx, v_idx):
        if u_idx == v_idx:
            return 0

        min_idx = min(u_idx, v_idx)
        max_idx = max(u_idx, v_idx)
        memoized_for_u = self.memoized[min_idx]

        if max_idx not in memoized_for_u:
            memoized_for_u[max_idx] = self.distance_fn(u, v)

        return memoized_for_u[max_idx]

    def __str__(self):
        return pprint.pformat(self.distance_matrix)



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

