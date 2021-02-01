import numpy as np
import pandas as pd

from scipy.cluster import hierarchy
import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from scipy.stats import chi2_contingency
import scipy.spatial.distance as ssd

DATE_INSERTED = 'Date Inserted'
AUROC = 'roc_auc'
BACC = 'balanced_accuracy'
MEASURES = [BACC, AUROC]


class SelectHierarchicalClustering(SelectorMixin, BaseEstimator):
    """
    A transformer that clusters the features in X according to dist_matrix, and selects a feature from each cluster with
    the highest chi2 score of X[feature] versus y
    """

    def __init__(self, dist_matrix, threshold=1):

        self.dist_matrix = dist_matrix
        self.threshold = threshold

    def _corr_linkage(self, method='average'):

        linkage = hierarchy.linkage(self.dist_matrix, method=method)

        return linkage

    def _hierarchical_clustering(self, linkage):
        """ Perform hierarchical clustering

        :param linkage: linkage dendogram created by hierarchy.linkage(self.distance_matrix, method=method)
        :return: a list of lists, each list represents a cluster and contains the indexes of features belonging
                 to the cluster
        """

        # array of len(X) - array[i] is the cluster number to which sample i belongs
        cluster_ids = hierarchy.fcluster(linkage, self.threshold, criterion='distance')

        cluster_id_to_feature_idx = {}
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_idx.setdefault(cluster_id, []).append(idx)

        return cluster_id_to_feature_idx.values()

    def fit(self, X, y):

        linkage = self._corr_linkage()
        clusters = self._hierarchical_clustering(linkage)

        chi2_vals, __ = sklearn.feature_selection.chi2(X, y)
        chi2_vals = pd.Series(chi2_vals)

        # fitted attributes
        self.n_features_ = X.shape[1]
        self.selected_features_ = [chi2_vals[cluster].idxmax() for cluster in clusters]

        print(f'threshold={self.threshold}, selected_features= {len(self.selected_features_)}')

        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        mask : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

        # Checks if the estimator is fitted by verifying the presence of fitted attributes (ending with a trailing
        # underscore) and otherwise raises a NotFittedError with the given message.
        sklearn.utils.validation.check_is_fitted(self)

        mask = np.zeros((self.n_features_, ), dtype=bool)

        mask[self.selected_features_] = 1

        return mask


def get_split_value(genome_id, train_genome_ids, validation_genome_ids):

    if genome_id in train_genome_ids:
        return -1
    elif genome_id in validation_genome_ids:
        return 0
    return None


def split_by_insertion_date(genomes_data, proportion=0.2):
    """
    Returns a predefined split of genomes_data to train and validation according to proportion, where the validation
    are genomes with the latest insertion date

    :param genomes_data: a GenomesData object
    :param proportion: the proportion size of validation
    :return: PredefinedSplit object
    """

    metadata = genomes_data.metadata
    # late-> early
    genomes_sorted_by_insertion = metadata.sort_values(DATE_INSERTED, ascending=False)

    total_number_of_genomes = len(genomes_data)
    validation_size = round(total_number_of_genomes * proportion)

    validation_genome_ids = genomes_sorted_by_insertion.index[:validation_size]
    train_genome_ids = genomes_sorted_by_insertion.index[validation_size:]

    fold = [get_split_value(genome_id, train_genome_ids, validation_genome_ids)
            for genome_id in genomes_data.genomes]

    return PredefinedSplit(fold)


def grid_search_results_to_df(grid_search, param_name, decimals=3):

    df = pd.DataFrame(grid_search.cv_results_)
    df.index = df['param_' + param_name]
    metrics = ['mean_test_'+metric for metric in MEASURES]
    df = df[metrics]
    df = df.round(decimals)

    return df


def perform_fs_k_best(train_dataset, k_range, split=None, random_state=0, return_train_score=False):

    pipeline = Pipeline(steps=[('vectorize', CountVectorizer(lowercase=False, binary=True)),
                               ('k_best', SelectKBest(score_func=chi2)),
                               ('rf', RandomForestClassifier(random_state=random_state))])

    param_grid = {
        'k_best__k': k_range,
    }

    scoring = MEASURES
    search = GridSearchCV(pipeline, param_grid, cv=split, scoring=scoring, refit=AUROC,
                          return_train_score=return_train_score)
    search.fit(train_dataset.data, train_dataset.y)

    print(f'Best roc_auc score is: {search.best_score_}')

    return search


def perform_fs_clusters(train_dataset, X_train_dist_mat, t_range, split=None, random_state=0,
                        return_train_score=False):

    pipeline = Pipeline(steps=[('vectorize', CountVectorizer(lowercase=False, binary=True)),
                               ('k_best', SelectKBest(score_func=sklearn.feature_selection.chi2, k=450)),
                               ('cluster', SelectHierarchicalClustering(X_train_dist_mat)),
                               ('rf', RandomForestClassifier(random_state=random_state))])

    param_grid = {
        'cluster__threshold': t_range,
    }

    scoring = MEASURES
    search = GridSearchCV(pipeline, param_grid, cv=split, scoring=scoring, refit=AUROC,
                          return_train_score=return_train_score)
    search.fit(train_dataset.data, train_dataset.y)

    print(f'Best roc_auc score is: {search.best_score_}')

    return search


def phi_coef(x, y):
    """Calculates phi coefficient between features

    Parameters:
    X - feature x column
    y - feature y column

    Returns:
    corr - phi coefficient value
    """

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    corr = np.sqrt(chi2 / n)

    return corr


def perform_fs_first_step(X_train, y_train, feature_names, k=100):
    """select the k features with the highest chi-square scores
     between each feature and the target labels


    Parameters:
    X_train - dataframe represents the training genomes feature vectors
    y_train - dataframe represents the training genomes labels
    feature_names - feature namess of X_train
    k - number of features to select.

    Returns:
    X_train_fs - dataframe represents the training genomes feature vectors (reduced size feature vectors which
                consists of the k selected features)
    """

    fs = SelectKBest(score_func=sklearn.feature_selection.chi2, k=k)
    fs.fit(X_train, y_train)
    fs_selected_indexes = fs.get_support()
    X_train_fs = pd.DataFrame(X_train[:, fs_selected_indexes].toarray(),
                                  columns=feature_names[fs_selected_indexes])

    return X_train_fs


def create_corr_matrix(X_train_raw, y_train, k=450):

    vectorizer = CountVectorizer(lowercase=False, binary=True)
    X_train = vectorizer.fit_transform(X_train_raw, y_train)
    feature_names = np.array(vectorizer.get_feature_names())

    X_train_k_best = perform_fs_first_step(X_train, y_train, feature_names, k=k)

    X_train_corr_mat = X_train_k_best.corr(method=phi_coef)

    return X_train_corr_mat


def feature_corr_to_dist_matrix(feature_corr_matrix):

    feature_corr_dist_matrix = 1 - feature_corr_matrix

    feature_corr_dist_matrix_condensed = ssd.squareform(feature_corr_dist_matrix)

    return feature_corr_dist_matrix_condensed



