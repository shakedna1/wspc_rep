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

        return list(cluster_id_to_feature_idx.values())

    def fit(self, X, y):
        """
        Clusters the features (X columns) using self.dist_matrix and self.threshold, and selects a feature from each
        cluster with the highest chi2 score versus y.
        The attribute self.n_features_ represents the number of features selected (=number of clusters)
        The attribute self.selected_features_ is a list of indexes that correspond to the selected features
        """
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
    """
    Perform cross validation with various k values and a RF classifier. k represents top k features with the highest
    chi2 scores between each feature and the target labels.
    The classifier scores are computed according to MEASURES. The best parameter k is selected according to AUROC score.

    :param train_dataset: GenomesData object representing the train dataset
    :param k_range: a range of k values
    :param split: an object that represents the splits to train and validation
                  (e.g. sklearn.model_selection.PredefinedSplit). If split is None, performs 5-fold stratified cross
                  validation
    :param random_state: a random state for the RF classifier
    :param return_train_score: specifies if the GridSearchCV should also calculate classifier score on train
    :return: A fitted GridSearchCV object
    """
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


def get_fs_pipeline(X_train_dist_mat, k, threshold, random_state=0):

    pipeline = Pipeline(steps=[('vectorize', CountVectorizer(lowercase=False, binary=True)),
                               ('k_best', SelectKBest(score_func=sklearn.feature_selection.chi2, k=k)),
                               ('cluster', SelectHierarchicalClustering(X_train_dist_mat, threshold=threshold)),
                               ('rf', RandomForestClassifier(random_state=random_state))])

    return pipeline


def perform_fs_clusters(train_dataset, X_train_dist_mat, t_range, split=None, random_state=0,
                        return_train_score=False):
    """
    Perform cross validation with various t values and a RF classifier. t represents a threshold for clustering.
    If the number of features is k, a threshold of 0 will leave k features (there will be k clusters).
    The higher the t, features with greater distance will be merged to the same cluster, thus a smaller number of
    features will be selected.
    The classifier scores are computed according to MEASURES. The best parameter t is selected according to AUROC score.

    :param train_dataset: GenomesData object representing the train dataset
    :param X_train_dist_mat: A precomputed distance matrix for the train set according to split
    :param t_range: a range of t values
    :param split: an object that represents the splits to train and validation
                  (e.g. sklearn.model_selection.PredefinedSplit). If split is None, performs 5-fold stratified cross
                  validation
    :param random_state: a random state for the RF classifier
    :param return_train_score: specifies if the GridSearchCV should also calculate classifier score on train
    :return: A fitted GridSearchCV object
    """

    pipeline = get_fs_pipeline(X_train_dist_mat, k=450, threshold=1, random_state=random_state)

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
    """
    Create a correlation matrix of size kxk between each two k best features X_train_raw. The best k features are
    selected according to chi2 score between each feature and the target labels.

    :param X_train_raw: X train strings of genes
    :param y_train: train labels
    :param k: k features to select according to chi2 score
    :return: kxk correlation matrix between each two k best features X_train_raw. Matrix values are between 0-1.
    A value of 0 represents no correlation between the corresponding features, a value of 1 represents a perfect
     correlation between the corresponding features.
    """

    vectorizer = CountVectorizer(lowercase=False, binary=True)
    X_train = vectorizer.fit_transform(X_train_raw, y_train)
    feature_names = np.array(vectorizer.get_feature_names())

    X_train_k_best = perform_fs_first_step(X_train, y_train, feature_names, k=k)

    X_train_corr_mat = X_train_k_best.corr(method=phi_coef)

    return X_train_corr_mat


def feature_corr_to_dist_matrix(feature_corr_matrix):
    """Transforms the correlation matrix feature_corr_matrix to a condensed distance matrix"""

    feature_corr_dist_matrix = 1 - feature_corr_matrix

    feature_corr_dist_matrix_condensed = ssd.squareform(feature_corr_dist_matrix)

    return feature_corr_dist_matrix_condensed



