import collections

import itertools
import pprint

import numpy
from matplotlib import pyplot
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import random
random.seed(0)

import kmeans
import metrics
import sim_matrix_heatmap
import utils
from dataset.articles import QianArticle

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
    path = 'results/hw3/articles/'
    articles = []
    for f in os.listdir(path):
        a = QianArticle(path=os.path.join(path, f))
        articles.append(a)

    corpus = preprocess.preprocess(corpus=articles,
                                   exclude_stopwords=True,
                                   method=config.VECTORIZER_METHOD)
    corpus.to_csv('dimreduce')

    distance_func = distance.euclidean
    feature_subset_selection_methods = {
        'Feature\nCorrelation':
            'correlated_features.tfidf.100articles.100terms.txt',
        'Feature\nRelevancy': 'maxrel.tfidf.100articles.100terms.txt',
        'Average\nTF-IDF': 'avgtfidf.tfidf.100articles.1000terms.txt',
        'Random\nForest': 'randforest.tfidf.100articles.100terms.txt',
        'Decision\nTree Scores': 'dectree.tfidf.100articles.100terms.txt',
        'Stopwords': None,
    }
    for k in feature_subset_selection_methods:
        v = feature_subset_selection_methods[k]
        if v:
            feature_subset_selection_methods[k] = os.path.join('dimreduce', v)

    fig, axes = pyplot.subplots(ncols=3)
    fig.set_size_inches((10, 5))
    sse_axes = axes[0]
    sse_axes.set_ylabel('SSE')
    sil_axes = axes[1]
    sil_axes.set_ylabel('Silhouette\nCoefficient')
    ideal_correlation_axes = axes[2]
    ideal_correlation_axes.set_ylabel('Ideal\nCorrelation')

    heatmap_fig, headmap_axes = pyplot.subplots(ncols=3, nrows=2)
    headmap_axes.shape = (6,)
    for a in headmap_axes:
        a.set_aspect(aspect='equal', adjustable='box')
    #heatmap_fig.set_size_inches((15, 3))

    ticks = []
    scores = {}
    #index = 1

    for index, feature_subset_method in enumerate(feature_subset_selection_methods):
        logger.info(hr('Clustering on {}'.format(feature_subset_method), '#'))
        scores[feature_subset_method] = {}
        feature_selection_file = feature_subset_selection_methods[feature_subset_method]

        logger.info(hr('Vectorizing Corpus'))
        corpus = preprocess.preprocess(corpus=articles,
                                       exclude_stopwords=True,
                                       feature_subset=feature_selection_file,
                                       method=config.VECTORIZER_METHOD)

        logger.info(hr('K-Means Clustering'))
        articles_np = numpy.array(corpus.corpus)
        clustering, centroids = kmeans.kmeans(vectors=corpus.matrix.toarray(),
                                              k=7,
                                              distance=distance_func,
                                              initial_centroid_method='random')

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
        cm = sim_matrix_heatmap.plot(sorted_matrix=articles_sorted_by_cluster,
                                distance_metric=distance_func,
                                cart_prod_indices=cart_product_indices,
                                axes=headmap_axes[index])
        headmap_axes[index].set_xlabel(feature_subset_method)

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
        scores[feature_subset_method]['ideal'] = correlation

        ## SSE
        logger.debug(hr('Calculating SSE'))
        sse = metrics.calculate_sse(centroids,
                                    clustering,
                                    corpus.matrix,
                                    distance_func)
        logger.debug('SSE: {}'.format(sse))
        scores[feature_subset_method]['sse'] = sse
        sse_axes.bar(index, sse)

        ## Silhouette coefficient
        logger.info(hr('Calculating Silhouette Coefficient'))
        silhouette = metrics.silhouette_coeff(clustering,
                                              corpus.matrix.toarray(),
                                              distance_func)
        logger.debug('Silhouette Coefficient: {}'.format(silhouette))
        scores[feature_subset_method]['silhouette'] = silhouette
        sil_axes.bar(index, silhouette)

        ticks.append((index, feature_subset_method))

    logger.debug(pprint.pformat(scores))

    for a in axes:
        a.set_xticks([t[0] for t in ticks])
        a.set_xticklabels([t[1] for t in ticks], rotation='vertical')

    logger.info(hr('VARIABLE K-MEANS', '#'))

    sse_scores = []
    sil_scores = []
    average_purity = []
    variable_k_fig, (sse_axes, sil_axes, avg_purity_axes) = pyplot.subplots(
        nrows=3, sharex=True)
    variable_k_fig.set_size_inches((6,9))
    sse_axes.set_xticks(range(3, 22))
    sse_axes.set_ylabel('SSE')
    sse_axes.set_xlabel('K (for K-means)')
    sil_axes.set_ylabel('Silhouette\nCoefficient')
    sil_axes.set_xticks(range(3, 22))
    sil_axes.set_xlabel('K (for K-means)')
    avg_purity_axes.set_ylabel('Average\nCluster\nPurity')
    avg_purity_axes.set_xlabel('K (for K-means)')
    avg_purity_axes.set_xticks(range(3, 22))

    feature_subset_method = 'Random\nForest'

    range_of_k = list(range(3, 22))
    for index, num_clusters in enumerate(range_of_k):
        logger.info(hr('{0} clusters'.format(num_clusters)))
        feature_selection_file = feature_subset_selection_methods[
            feature_subset_method]

        logger.info(hr('Vectorizing Corpus'))
        corpus = preprocess.preprocess(corpus=articles,
                                       exclude_stopwords=True,
                                       feature_subset=feature_selection_file,
                                       method=config.VECTORIZER_METHOD)

        logger.info(hr('K-Means Clustering'))
        articles_np = numpy.array(corpus.corpus)
        clustering, centroids = kmeans.kmeans(vectors=corpus.matrix.toarray(),
                                              k=num_clusters,
                                              distance=distance_func,
                                              initial_centroid_method='random')

        clusters = []
        dominating_category_size = []
        for cluster_index, article_indices in enumerate(clustering):
            print('{0}: {1}'.format(cluster_index, article_indices))
            articles_in_cluster = list(articles_np[article_indices])
            clusters.append(articles_in_cluster)

            cluster_category_histogram = collections.defaultdict(int)
            for art in articles_in_cluster:
                cluster_category_histogram[art.category] += 1

            max_num_articles = float('-inf')
            for cat, num in cluster_category_histogram.items():
                if num > max_num_articles:
                    max_num_articles = num

            dominating_category_size.append(max_num_articles)

        avg_purity = numpy.sum(dominating_category_size) / corpus.count
        average_purity.append(avg_purity)


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

        # logger.info(hr('Similarity / Distance Matrix'))
        # cm = sim_matrix_heatmap.plot(sorted_matrix=articles_sorted_by_cluster,
        #                              distance_metric=distance_func,
        #                              cart_prod_indices=cart_product_indices,
        #                              axes=headmap_axes[index])
        # headmap_axes[index].set_xlabel(feature_subset_method)

        ## SSE
        logger.debug(hr('Calculating SSE'))
        sse = metrics.calculate_sse(centroids,
                                    clustering,
                                    corpus.matrix,
                                    distance_func)
        logger.debug('SSE: {}'.format(sse))
        #sse_axes.bar(index, sse)
        sse_scores.append(sse)

        ## Silhouette coefficient
        logger.info(hr('Calculating Silhouette Coefficient'))
        silhouette = metrics.silhouette_coeff(clustering,
                                              corpus.matrix.toarray(),
                                              distance_func)
        logger.debug('Silhouette Coefficient: {}'.format(silhouette))
        #sil_axes.bar(index, silhouette)
        sil_scores.append(silhouette)

    sse_axes.plot(range_of_k, sse_scores)
    sil_axes.plot(range_of_k, sil_scores)
    avg_purity_axes.plot(range_of_k, average_purity)
    logger.debug('Silhouette scores:')
    logger.debug(pprint.pformat(list(zip(range_of_k, sil_scores))))
    logger.debug('SSE scores:')
    logger.debug(pprint.pformat(list(zip(range_of_k, sse_scores))))
    logger.debug('Cluster purity: {}'.format(average_purity))

    # fig.suptitle('Feature Selection Methods', y=1.08)
    # heatmap_fig.suptitle('Similarity Matrices', y=1.08)
    # variable_k_fig.suptitle('Natural Clusters:\nVarying the Number of '
    #                         'Clusters', y=1.08)

    #heatmap_fig.colorbar(cm)


    ### Difference in initial centroid selection
    # sse_axes = axes[0, 1]
    # sse_axes.set_ylabel('SSE')
    # sil_axes = axes[1, 1]
    # sil_axes.set_ylabel('Silhouette\nCoefficient')
    # ideal_correlation_axes = axes[2, 1]
    # ideal_correlation_axes.set_ylabel('Ideal\nCorrelation')
    # scores = {}
    # ticks = []
    #
    # feature_subset_method = 'Random\nForest'
    # for index, initial_centroid_method in enumerate(['random', 'kpp']):
    #     logger.info(hr('Clustering on {}'.format(feature_subset_method), '#'))
    #     scores[initial_centroid_method] = {}
    #     feature_selection_file = feature_subset_selection_methods[
    #         feature_subset_method]
    #
    #     logger.info(hr('Vectorizing Corpus'))
    #     corpus = preprocess.preprocess(corpus=articles,
    #                                    exclude_stopwords=True,
    #                                    feature_subset=feature_selection_file,
    #                                    method=config.VECTORIZER_METHOD)
    #
    #     logger.info(hr('K-Means Clustering'))
    #     articles_np = numpy.array(corpus.corpus)
    #     clustering, centroids = kmeans.kmeans(
    #         vectors=corpus.matrix.toarray(),
    #         k=7,
    #         distance=distance_func,
    #         initial_centroid_method=initial_centroid_method)
    #
    #     clusters = []
    #     for cluster_index, article_indices in enumerate(clustering):
    #         print('{0}: {1}'.format(cluster_index, article_indices))
    #         clusters.append(list(articles_np[article_indices]))
    #
    #     articles_sorted_by_cluster = []
    #     article_cluster_indices = []
    #     for cluster_index, cluster in enumerate(clusters):
    #         # sort the cluster based on category labels
    #         cluster.sort(key=lambda a: a.category)
    #
    #         # flatten cluster
    #         articles_sorted_by_cluster.extend([article.vector
    #                                            for article in cluster])
    #         article_cluster_indices.extend([cluster_index
    #                                         for _ in range(len(cluster))])
    #
    #         logger.debug('{0: >5}:'.format(cluster_index))
    #         for article in cluster:
    #             logger.debug('    {0: >15} -- {1}'.format(article.category,
    #                                                       article.title))
    #
    #     logger.info(hr('Similarity / Distance Matrix'))
    #     # sim_matrix_heatmap.plot(sorted_matrix=articles_sorted_by_cluster,
    #     #                         distance_metric=distance.cosine,
    #     #                         cart_prod_indices=cart_product_indices)
    #
    #     ## Ideal Cluster to Ideal Class Similarity Matrix correlation
    #     logger.info(hr('Calculating Ideal Similarity Correlation'))
    #     article_class_indices = [corpus.class_to_index[article.category]
    #                              for article in corpus]
    #     correlation = metrics.ideal_correlation(
    #         cluster_indices=article_cluster_indices,
    #         class_indices=article_class_indices,
    #         n=corpus.count)
    #     logger.info('Ideal Correlation: {}'.format(correlation))
    #     ideal_correlation_axes.bar(index, correlation)
    #     scores[initial_centroid_method]['ideal'] = correlation
    #
    #     ## SSE
    #     logger.debug(hr('Calculating SSE'))
    #     sse = metrics.calculate_sse(centroids,
    #                                 clustering,
    #                                 corpus.matrix,
    #                                 distance_func)
    #     logger.debug('SSE: {}'.format(sse))
    #     scores[initial_centroid_method]['sse'] = sse
    #     sse_axes.bar(index, sse)
    #
    #     ## Silhouette coefficient
    #     logger.info(hr('Calculating Silhouette Coefficient'))
    #     silhouette = metrics.silhouette_coeff(clustering,
    #                                           corpus.matrix.toarray(),
    #                                           distance_func)
    #     logger.debug('Silhouette Coefficient: {}'.format(silhouette))
    #     scores[initial_centroid_method]['silhouette'] = silhouette
    #     sil_axes.bar(index, silhouette)
    #
    #     ticks.append((index, initial_centroid_method))
    #
    # logger.debug(pprint.pformat(scores))
    # axes[2,1].set_xticks([t[0] for t in ticks])
    # axes[2,1].set_xticklabels([t[1] for t in ticks], rotation='vertical')


    fig.tight_layout(w_pad=2.0)
    heatmap_fig.tight_layout(h_pad=1)
    variable_k_fig.tight_layout(h_pad=2)
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

