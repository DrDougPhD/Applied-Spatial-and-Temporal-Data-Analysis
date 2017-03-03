import collections

import itertools
import numpy
# from scipy.spatial.distance import cosine as proximity
from matplotlib.patches import Circle
from scipy.spatial.distance import euclidean as proximity
from scipy.stats import zscore

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
                                   method='tfidf')
    logger.debug(corpus)

    # produce circle graphs to show diameter and separation between a pair of
    #  classes

    # 1. calculate centroid of every cluster
    categories = collections.defaultdict(list)
    centroids = {}
    for article in corpus:
        categories[article.category].append(article)

    for category, articles in categories.items():
        logger.debug(hr(category))
        article_matrix = numpy.array([article.vector for article in articles])
        # for row in article_matrix:
        #     logger.debug(list(row))
        centroid = numpy.sum(article_matrix, axis=0) / len(articles)
        centroids[category] = centroid

    # 2. Calculate distances to the centroid for each item in a class
    distances_to_centroid = collections.defaultdict(list)
    zscores = {}
    average_distances_to_centroid = {}
    logger.debug(hr('Z-scores of Article Distance to Centroid', '-'))
    for category, centroid in centroids.items():
        for article in categories[category]:
            distances_to_centroid[category].append(
                proximity(article.vector, centroid)
            )
        average_distances_to_centroid[category] = float(numpy.mean(
            distances_to_centroid[category]))

        zscores[category] = sorted(
            zip(categories[category],
                zscore(distances_to_centroid[category]),
                distances_to_centroid[category]),
            key=lambda x: x[1]
        )
        logger.debug(hr(category, '.'))
        for article, zscore_centroid_distance, distance in zscores[category]:
            logger.debug('{0:+.5f}\t{1:.5f}\t{2}'.format(
                zscore_centroid_distance,
                distance,
                article.title))

    # 3. Calculate separation between clusters
    separations = []
    for (category1, articles1), (category2, articles2)\
            in itertools.combinations(categories.items(), 2):
        separations_between_classes = []

        for art1 in articles1:
            for art2 in articles2:
                separations_between_classes.append(
                    proximity(art1.vector, art2.vector)
                )

        avg_proximity = numpy.mean(separations_between_classes)
        separations.append((min(category1, category2),
                            max(category1, category2),
                            avg_proximity))
    separations.sort(key=lambda x: x[2])
    logger.debug(hr('Class Separations', '-'))
    for (k1, k2, separation) in separations:
        logger.debug('{0: >15}, {1: <15}: {2}'.format(k1, k2, separation))

    # 4. Create bubble chart
    import matplotlib.pyplot as plt
    figure, bubbles = plt.subplots(nrows=7, ncols=3, sharex=True)
    figure.set_size_inches(h=14, w=10)
    bubbles.shape = (21,)
    logger.debug(hr('Plotting', '-'))
    for axes, (k1, k2, separation) in zip(bubbles, separations):
        axes.set_title('{0} and {1}'.format(k1.title(), k2.title()))
        cat1_diameter = average_distances_to_centroid[k1]
        cat2_diameter = average_distances_to_centroid[k2]

        logger.debug('{0: >15}, {1: <15}: {2}'.format(k1, k2, separation))

        cat1_circle = Circle((0, 0), cat1_diameter, alpha=0.5)
        cat2_circle = Circle((separation, 0), cat2_diameter, alpha=0.5)
        axes.add_patch(cat1_circle)
        axes.add_patch(cat2_circle)

    plt.tight_layout(h_pad=1.5)
    plt.show()


    #
    #
    #
    #
    #
    #
    # categories = collections.defaultdict(list)
    # distances = collections.defaultdict(list)
    # centroids = {}
    # for article in corpus:
    #     categories[article.category].append(article.vector)
    #
    # for category, articles in categories.items():
    #     logger.debug(hr(category))
    #     article_matrix = numpy.array(articles)
    #     # for row in article_matrix:
    #     #     logger.debug(list(row))
    #     centroid = numpy.sum(article_matrix, axis=0) / len(articles)
    #     centroids[category] = centroid
    #
    #     # logger.debug('Centroid:')
    #     # logger.debug(list(centroid))
    #     # logger.debug('-'*80)
    #     for (i, u), (j, v) in itertools.combinations(
    #             enumerate(article_matrix), 2):
    #         distance = proximity(u, v)
    #         distances[category].append(distance)
    #
    #         # logger.debug('{0}:\t{1}'.format(i, u))
    #         # logger.debug('{0}:\t{1}'.format(j, v))
    #         logger.debug('{0}, {1}:\t{2}'.format(i, j, distance))
    #
    # cohesions = {}
    # logger.debug(hr('Class Cohesions', '-'))
    # for category, distance in distances.items():
    #     cohesion = numpy.mean(distance)
    #     cohesions[category] = cohesion
    #     logger.debug('{0: >15}: {1}'.format(category, cohesion))
    #
    # separations = []
    # for (category1, centroid1), (category2, centroid2) in \
    #         itertools.combinations(centroids.items(), 2):
    #     separation = proximity(centroid1, centroid2)
    #     # logger.debug('\n{0: >15}\n{1: >15}:\t{2}'.format(category1,
    #     #                                                  category2,
    #     #                                                  separation))
    #     separations.append((category1, category2, separation))
    #
    # separations.sort(key=lambda x: x[-1])
    # logger.debug(hr('Class Separations', '-'))
    # for (k1, k2, separation) in separations:
    #     logger.debug('{0: >15}, {1: <15}: {2}'.format(k1, k2, separation))






if __name__ == '__main__':
    main()