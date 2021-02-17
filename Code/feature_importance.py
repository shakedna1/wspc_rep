import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier


def feature_importance(model, top=None):
    """ Returns importances of the model features and a list of indices sorted by the importances

    Parameters:
    model - the relevant classification model
    top - number of top features to return

    Returns:
    importances - features importances
    indices - indices sorted by the importances

    """

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    indices = indices[:top]

    return importances, indices


def permutation_importance(model, X, y, top=None, n_repeats=100, random_state=0):
    """
    Returns importances of the model features and a list of indices sorted by the importances, using permutation
    importance measure

    :param model: the relevant classification model
    :param X: dataset in which features will be permuted
    :param y: labels corresponding to X
    :param top: number of top features to return
    :param n_repeats: number of times the permutation will be repeated
    :param random_state: for the permutation
    :return: importances - features importances
            indices - indices sorted by the importances
    """
    result = sklearn.inspection.permutation_importance(model, X=X, y=y, n_repeats=n_repeats, random_state=random_state)
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    indices = indices[:top]

    return importances, indices


def train_and_rank_features_one_rf_model(x, y, curr_random_state, importance_func):
    """ Trains and rank features using one random forest classifier.

    Parameters:
    x - features vectors dataframe
    y - true labels dataframe
    curr_random_state - random state for the RandomForestClassifier
    importance_func - importance function

    Returns:
    importances - importances of features
    indices - indices of top features
    """

    fs_rs_model = RandomForestClassifier(random_state=curr_random_state)
    fs_rs_model.fit(x, y)
    importances, indices = importance_func(fs_rs_model)

    return importances, indices



def find_top_features_in_multiple_runs(x, y, importance_func, n_runs=100):
    """ Calculating  MDI value over multiple random forest
        classifiers with different random seeds.

    Parameters:
    x - features vectors dataframe
    y - true labels dataframe
    importance_func - importance function
    n_runs - number of random forest to use for the calculation

    Returns:
    all_importances_for_all_features -
    """

    all_importances_for_all_features = {}

    for rand_state_num in range(n_runs):

        # if rand_state_num % 5 == 0:
        #     print(f'run={rand_state_num}')

        importances, indices = train_and_rank_features_one_rf_model(x, y, rand_state_num, importance_func)

        for feature_num in range(x.shape[1]):
            all_importances_for_all_features.setdefault(feature_num, []).append(importances[feature_num])

    return all_importances_for_all_features



def count_hp_vs_nhp(genome_ids, y):
    """ Counts and returns the number of hps and nhps genomes in a list of genomes

    Parameters:
    genome_ids - genomes ids
    y - a Series with genome ids as index, and values of 0 or 1

    Returns:
    hps - the number of hps genomes in genome_ids
    nhps - the number of nhps genomes in genome_ids
    """

    genome_labels = y.loc[genome_ids]
    hps = len(genome_labels[genome_labels == 1])
    nhps = len(genome_labels[genome_labels == 0])

    assert len(genome_labels) == hps + nhps

    return hps, nhps


def count_hp_vs_nhps_feature(x, y_df, feature):
    """ Returns the number of hps and nhps genomes that contain the specific pgfam represented by the specific feature

    Parameters:
    x - features vectors dataframe
    y_df - true labels dataframe
    feature - pgfam feature

    Returns:
    hps - the number of hps genomes that contain the specific pgfam represented by the specific feature
    nhps - the number of nhps genomes that contain the specific pgfam represented by the specific feature
    """

    genomes_with_feature = x[x.loc[:, feature] == 1]
    hps, nhps = count_hp_vs_nhp(genomes_with_feature.index, y_df)

    return hps, nhps



def split_top_features_to_classes(x, y_df, indices):
    """ Splits top features to classes (top HP features/ top NHP features)

    Parameters:
    x - features vectors dataframe
    y_df - true labels dataframe
    indices - indices of top features

    Returns:
    top_hp_feats - sorted list of top HP features
    top_nhp_feats - sorted list of top NHP features
    """

    top_hp_feats, top_nhp_feats = [], []

    for i in indices:

        feature = x.columns[i]

        hps, nhps = count_hp_vs_nhps_feature(x, y_df, feature)

        if hps >= nhps:
            top_hp_feats.append(feature)
        else:
            top_nhp_feats.append(feature)

    return top_hp_feats, top_nhp_feats


def get_top_features_in_multiple_runs(x, y, importance_func, n_runs=100):
    """ Gets the top features by calculating the average MDI value using multiple random forest
        classifiers with different random seeds.

    Parameters:
    x - features vectors dataframe
    y - true labels dataframe
    importance_func - importance function
    n_runs - number of random forest to use for the calculation

    Returns:
    indices - indices of the features sorted by mean_importance.
    mean_importance - mean importance per feature over the n_runs classifiers.
    std_importance - std of importances per features over the n_runs classifiers.
    """

    all_importances_for_all_features = find_top_features_in_multiple_runs(x=x, y=y, importance_func=importance_func,
                                                                          n_runs=n_runs)

    mean_importance = [np.mean(all_importances_for_all_features[feature_num]) for feature_num in range(x.shape[1])]
    std_importance = [np.std(all_importances_for_all_features[feature_num]) for feature_num in range(x.shape[1])]
    indices = np.argsort(mean_importance)[::-1]

    return indices, mean_importance, std_importance


def get_top_features_per_class_in_multiple_runs(x, y, importance_func, n_runs=100):
    """ Returns a dictionary of the top features per class (HP/NHP)

    Parameters:
    x - features vectors dataframe
    y_df - true labels dataframe
    importance_func - importance function
    n_runs - number of random forest to use for the calculation

    Returns:
    class_features - dictionary that represents the top HP and top NHP features, with additional information.
    """

    indices, mean_importance, std_importance = get_top_features_in_multiple_runs(x, y, importance_func, n_runs)

    top_hp_feats, top_nhp_feats = split_top_features_to_classes(x, y, indices)

    class_features = {'HP': top_hp_feats,
                      'NHP': top_nhp_feats,
                      'importances': mean_importance,
                      'std': std_importance}

    return class_features


def create_top_feats_df(class_features, x, y_df, pgfam_to_desc, top_feats=None):
    """ Creates dataframe that represents the relevant information on the top features

    Parameters:
    class_features - dictionary that represents the top HP and top NHP features, with additional information
    x - features vectors dataframe
    y_df - true labels dataframe
    pgfam_to_desc - a dict with pgfam as key and its description as value
    top_feats - number of top features to include in the resulting dataframe

    Returns:
    hps_df - dataframe that represents the top HP features
    nhps_df - dataframe that represents the top NHP features
    """

    mean_importance, std_importance = class_features['importances'], class_features['std']

    total_hps, total_nhps = count_hp_vs_nhp(y_df.index, y_df)

    top_features_data = {'HP': {}, 'NHP': {}}

    classes = ['HP', 'NHP']
    for c_class in classes:

        top_features = class_features[c_class]

        for c, feature in enumerate(top_features[:top_feats]):
            hps, nhps = count_hp_vs_nhps_feature(x, y_df, feature)

            i = x.columns.get_loc(feature)

            denominator = 1 if nhps == 0 else nhps / total_nhps
            p_ratio = round((hps / total_hps) / (denominator), 2)
            mean_feature_importance = round(mean_importance[i], 3)
            std_feature_importance = round(std_importance[i], 3)
            pgfam_desc = pgfam_to_desc.get(feature, "")

            top_features_data[c_class].setdefault('Feature', []).append(feature)
            top_features_data[c_class].setdefault('Function', []).append(pgfam_desc)
            top_features_data[c_class].setdefault('Mean Importance (SD)', []).append(
                f'{mean_feature_importance} ({std_feature_importance})')
            # top_features_data[c_class].setdefault('Mean Importance Std', []).append(std_feature_importance)
            top_features_data[c_class].setdefault('HPs', []).append(hps)
            top_features_data[c_class].setdefault('NHPs', []).append(nhps)
            top_features_data[c_class].setdefault('P-Ratio', []).append(p_ratio)

    hps_df = pd.DataFrame(top_features_data['HP'])
    nhps_df = pd.DataFrame(top_features_data['NHP'])

    return hps_df, nhps_df


