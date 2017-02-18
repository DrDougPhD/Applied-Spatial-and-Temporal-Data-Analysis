from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import logging
from progressbar import ProgressBar
import numpy
import itertools
import os
import csv
from scipy.spatial import distance
try:  # this is my own package, but it might not be present
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

logger = logging.getLogger('cnn.'+__name__)

CREATED_FILES = []


# note: jaccard from scipy is not jaccard similarity, but rather computing
#  the jaccard dissimilarity! i.e. numerator is cTF+cFT, not cTT
def jaccard(u, v):
    equal = (v == u)
    are_zero = (u == 0)
    equal_nonzero = (are_zero == False) * equal
    both_zeros = equal * are_zero
    results_not_zeros = (both_zeros == False)
    return numpy.sum(equal_nonzero) / numpy.sum(results_not_zeros)

ACTIVATED_DISTANCE_FNS = [distance.euclidean, jaccard, distance.cosine]


def go(calc, funcs, store_in):
    if funcs is None:
        distance_fns = ACTIVATED_DISTANCE_FNS
    else:
        distance_fns = [fn for fn in ACTIVATED_DISTANCE_FNS
                        if fn.__name__ in funcs]

    data = {}
    for fn in distance_fns:
        logger.info(hr(fn.__name__, line_char='-'))
        similarities = calc.pairwise_compare(
            by=fn, save_to=store_in)
        data[fn.__name__] = similarities

    return data


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)

class PairwiseSimilarity(object):
    def __init__(self, corpus, method, stopwords):
        self.corpus = corpus
        self.method = method

        # specify method in which corpus is repr'd as matrix:
        #  1. an existence matrix (0 if token is abscent, 1 if present)
        #  2. a term freq matrix (element equals token count in doc)
        #  3. Tf-Idf matrix
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=1,
                                              stop_words=stopwords)
        else:
            # matrix will be converted to binary matrix further down
            self.vectorizer = CountVectorizer(min_df=1,
                                              stop_words=stopwords)

        plain_text = [str(document) for document in self.corpus]
        self._matrix = self.vectorizer.fit_transform(plain_text)
        for i in range(len(corpus)):
            vector = self._matrix.getrow(i).toarray()[0]
            doc = corpus[i]
            if method == 'existence':
                # convert vector into a binary vector (only 0s and 1s)
                # vector = [ int(bool(e)) for e in vector ]
                vector = (vector != 0).astype(int).toarray()[0]
            doc.vector = vector

        self.features = self.vectorizer.get_feature_names()
        logger.info('{} unique tokens'.format(len(self.features)))

    def pairwise_compare(self, by, save_to=None):
        i = 0
        n = int(nCr(len(self.corpus), 2))

        progress = ProgressBar(
            max_value=n)

        similarity_calculations = []
        for u, v in itertools.combinations(self.corpus, 2):

            progress.update(i)
            i += 1

            comparison = ComparedArticles(u, v, by, self.features)
            logger.debug(comparison)
            logger.debug('-' * 80)
            similarity_calculations.append(comparison)

        progress.finish()

        # sort similarities by their normalized scores
        similarity_calculations.sort(key=lambda c: c.normalized, reverse=True)

        if save_to:
            similarities_file = os.path.join(
                save_to,
                '{method}.{distance}.{n}.tsv'.format(distance=by.__name__,
                                                     method=self.method,
                                                     n=n))
            with open(similarities_file, 'w') as f:
                CREATED_FILES.append(similarities_file)

                # find the length of the feature which occurs most commonly in
                # both articles. for pretty printing
                highest_feat_max_len_obj = max(
                    similarity_calculations,
                    key=lambda x: len(x.highest_common_feat.name))
                highest_feat_max_length = len(
                    highest_feat_max_len_obj.highest_common_feat.name)

                # find the article title that is the shortest. make pretty
                art_w_short_title = max([c.article[0]
                                         for c in similarity_calculations],
                                        key=lambda r: len(r.title))
                short_title_len = len(art_w_short_title.title) + 4

                f.write('{score:^10}\t'
                        '{normalized:^10}\t'
                        '{highest_common_feature}\t'
                        '{highest_common_feature_score:^10}\t'
                        '{title}\t'
                        'Article #2\n'.format(
                    title='Article #1'.ljust(short_title_len),
                    score='score',
                    normalized='similarity',
                    highest_common_feature='mcf'.center(
                        highest_feat_max_length),
                    highest_common_feature_score='# occurs',
                ))

                for calc in similarity_calculations:
                    f.write('{score:10.5f}\t'
                            '{normalized:10.5f}\t'
                            '{highest_common_feature}\t'
                            '{highest_common_feature_score:10.5f}\t'
                            '{art1}\t'
                            '"{art2}"\n'.format(
                        art1='"{}"'.format(calc.article[0].title)
                            .ljust(short_title_len),
                        art2=calc.article[1].title,
                        score=calc.score,
                        normalized=calc.normalized,
                        highest_common_feature=calc.highest_common_feat
                            .name.ljust(
                            highest_feat_max_length),
                        highest_common_feature_score=calc.highest_common_feat
                            .score
                    ))

        return similarity_calculations

    def save_matrix_to(self, directory):
        logger.info('Saving TF matrix to file')
        matrix_file = os.path.join(directory, self.method + '_matrix.csv')
        CREATED_FILES.append(matrix_file)
        with open(matrix_file, 'w') as f:
            csvfile = csv.writer(f, delimiter='|')
            csvfile.writerow(self.features)
            csvfile.writerows(self._matrix.toarray())

    def save_aggregate_feature_counts(self, directory):
        features_file = os.path.join(directory, 'aggregate_feature_counts.csv')
        CREATED_FILES.append(features_file)
        with open(features_file, 'w') as counts_file:
            csvfile = csv.writer(counts_file)
            csvfile.writerow(['token', 'count'])

            summed_vector = sum(self._matrix).toarray()[0]
            csvfile.writerows(sorted(
                zip(self.features, summed_vector),
                key=lambda e: e[1],
                reverse=True,
            ))


class ComparedArticles(object):
    class HighestCommonFeature(object):
        def __init__(self, articles, features, max_or_min=max):
            u, v = map(lambda x: x.vector, articles)
            # only sum up token occurrences for tokens that appear in both documents
            shared_appearances = (u + v) * (u != 0) * (v != 0)

            (i,), score = max_or_min(numpy.ndenumerate(shared_appearances),
                                     key=lambda e: e[1])
            self.name = features[i]
            self.score = score

    def __init__(self, art1, art2, fn, features):
        # sort articles by title
        if len(art1.title) < len(art2.title):
            self.article = [art1, art2]
        else:
            self.article = [art2, art1]
        self.score = fn(art1.vector, art2.vector)
        self.distance_fn = fn.__name__

        # normalize the score based on the distance function used
        if self.distance_fn == 'euclidean':
            # [ 0, +inf ) --(flipped)-> ( 0, +inf ) -> ( 0, 1 ] --(flip)-> [ 0, 1 )
            self.normalized = 1 / (self.score + 1)
        elif self.distance_fn == 'cosine':
            # [ -1, 1 ] -> [ 0, 2 ] -> [ 0, 1 ]
            self.normalized = 1 - self.score
        elif self.distance_fn == 'jaccard':
            self.normalized = self.score  # no need to normalize

        self.highest_common_feat = ComparedArticles.HighestCommonFeature(
            articles=self.article,
            features=features)

    def __str__(self):
        return '{0}\n' \
               'vs.\n' \
               '{1}\n' \
               '== SCORE ({2}): {3}'.format(
            repr(self.article[0]), repr(self.article[1]),
            self.distance_fn,
            self.score)
