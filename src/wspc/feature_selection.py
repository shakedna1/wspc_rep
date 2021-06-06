import numpy as np
import sklearn
import pandas as pd
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, SelectorMixin
from sklearn.pipeline import Pipeline


class SelectHierarchicalClustering(SelectorMixin, BaseEstimator):
    """
    A transformer that clusters the features in X according to dist_matrix, and selects a feature from each cluster with
    the highest chi2 score of X[feature] versus y
    """

    def __init__(self, dist_matrix=None, threshold=1):

        self.dist_matrix = dist_matrix
        self.threshold = threshold

    def _phi_coef(self, x, y):
        """Calculates phi coefficient between features

        Parameters:
        x - feature x column
        y - feature y column

        Returns:
        phi coefficient value
        """

        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        corr = np.sqrt(chi2 / n)

        return corr

    def _calc_dist_matrix(self, X):
        """Calculate distance matrix between each two features in X, each value is 1-phi_correlation"""

        X_df = pd.DataFrame.sparse.from_spmatrix(X)
        X_corr_mat = X_df.corr(method=self._phi_coef)

        feature_corr_dist_matrix = 1 - X_corr_mat

        feature_corr_dist_matrix_condensed = ssd.squareform(feature_corr_dist_matrix)

        self.dist_matrix = feature_corr_dist_matrix_condensed

    def _corr_linkage(self, method='average'):

        linkage = hierarchy.linkage(self.dist_matrix, method=method)

        return linkage

    def _hierarchical_clustering(self, linkage):
        """ Perform hierarchical clustering

        Parameters:
        linkage - linkage dendogram created by hierarchy.linkage(self.distance_matrix, method=method)

        Returns:
        a list of lists, each list represents a cluster and contains the indexes of features belonging
                 to the cluster
        """

        # array of len(X) - array[i] is the cluster number to which sample i belongs
        cluster_ids = hierarchy.fcluster(linkage, self.threshold, criterion='distance')

        cluster_id_to_feature_idx = {}
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_idx.setdefault(cluster_id, []).append(idx)

        return list(cluster_id_to_feature_idx.values())

    def fit(self, X, y):
        """
        Clusters the features (X columns) using self.dist_matrix and self.threshold, and selects a feature from each
        cluster with the highest chi2 score versus y.
        The attribute self.n_features_ represents the number of features selected (=number of clusters)
        The attribute self.selected_features_ is a list of indexes that correspond to the selected features
        """

        if not self.dist_matrix:
            self._calc_dist_matrix(X)

        linkage = self._corr_linkage()
        clusters = self._hierarchical_clustering(linkage)

        chi2_vals, __ = sklearn.feature_selection.chi2(X, y)
        chi2_vals = pd.Series(chi2_vals)

        # fitted attributes
        self.n_features_ = X.shape[1]
        self.selected_features_ = [chi2_vals[cluster].idxmax() for cluster in clusters]
        self.clusters_ = clusters

        print(f'threshold={self.threshold:.2f}, selected_features={len(self.selected_features_)}')

        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected

        Returns:
        mask - boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

        # Checks if the estimator is fitted by verifying the presence of fitted attributes (ending with a trailing
        # underscore) and otherwise raises a NotFittedError with the given message.
        sklearn.utils.validation.check_is_fitted(self)

        mask = np.zeros((self.n_features_, ), dtype=bool)

        mask[self.selected_features_] = 1

        return mask


def get_fs_pipeline(k, threshold, random_state=0):

    pipeline = Pipeline(steps=[('vectorize', CountVectorizer(lowercase=False, binary=True)),
                               ('k_best', SelectKBest(score_func=sklearn.feature_selection.chi2, k=k)),
                               ('cluster', SelectHierarchicalClustering(threshold=threshold)),
                               ('rf', RandomForestClassifier(random_state=random_state))])

    return pipeline
