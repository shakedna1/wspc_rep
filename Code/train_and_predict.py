import util

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import seaborn as sns
import sklearn.metrics as metrics

from scipy.cluster import hierarchy
from scipy.stats import chi2_contingency
from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier



def perform_fs_first_step(X_train, y_train, k=100):
    '''select the k features with the highest chi-square scores
     between each feature and the target labels


    Parameters:
    X_train - dataframe represents the training genomes feature vectors
    y_train - dataframe represents the training genomes labels
    k - number of features to select.

    Returns:
    X_train_fs - dataframe represents the training genomes feature vectors (reduced size feature vectors which
                consists of the k selected features)
    '''

    fs = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=k)
    fs.fit(X_train, y_train)
    fs_selected_indexes = fs.get_support(indices=True)
    X_train_fs = X_train.iloc[:, fs_selected_indexes]

    return X_train_fs



def predict(X_test, model):
    '''Predicts labels of X_test with a specific model


    Parameters:
    X_test - dataframe represents the test genomes features vectors
    model - the relevant model

    Returns:
    predictions - final predictions (0 - NHP, 1- HP)
    predictions_probs - probability estimates
    '''

    predictions = model.predict(X_test)
    predictions_probs = model.predict_proba(X_test)[:, 1]

    return predictions, predictions_probs



def predict_and_print_results(X_test, y_test, model):
    '''Predicts labels of X_test with a specific model and returns
       evaluation results of the model accuracy

    Parameters:
    X_test - dataframe represents the test genomes features vectors
    y_test - dataframe represents the test genomes labels
    model - the relevant model

    Returns:
    evaluation_results -  dictionary of the sensitivity, specificity, f1_macro, aupr_auc and roc_auc scores
    '''


    predictions, predictions_probs = predict(X_test, model)
    evaluation_results = calculate_results(predictions, predictions_probs, y_test)
    print_train_results(evaluation_results)

    return evaluation_results



def train_and_predict(X_train, y_train, X_test, y_test, random_state=0):
    '''Trains a new model, predicts test genoomes and prints accuracy evaluation results

    Parameters:
    X_train - dataframe represents the train genomes feature vectors
    y_train - dataframe represents the train genomes labels
    X_test - dataframe represents the test genomes feature vectors
    y_test - dataframe represents the test genomes labels
    random_state - random state for the random forest classifier

    Returns:
    fs_rs_model - resulting model
    evaluation_results -  dictionary of the sensitivity, specificity, f1_macro, aupr_auc and roc_auc scores
    '''

    rs_model = RandomForestClassifier(random_state=random_state)
    rs_model.fit(X_train, y_train)
    X_test_fs = X_test.loc[:, X_train.columns]
    evaluation_results = predict_and_print_results(X_test_fs, y_test, rs_model)

    return rs_model, evaluation_results



def calc_confusion_matrix_indexes(predictions, y_test):
    '''calculates confusion matrix

    Parameters:
    predictions - predicted labels
    y_test - dataframe represents the train genomes true labels


    Returns:
        true_negative - list of negative genomes with negative prediction value
        true_positive -list of positive genomes with positive prediction value
        false_positive -  list of negative genomes with positive prediction value
        false_negative - list of positive genomes with negative prediction value
    '''

    true_negative = [i for i in range(len(predictions)) if (predictions[i] == y_test[i] == 0)]
    true_positive = [i for i in range(len(predictions)) if (predictions[i] == y_test[i] == 1)]
    false_positive = [i for i in range(len(predictions)) if (predictions[i] == 1 and y_test[i] == 0)]
    false_negative = [i for i in range(len(predictions)) if (predictions[i] == 0 and y_test[i] == 1)]

    return true_negative, true_positive, false_positive, false_negative



def calc_confusion_matrix(predictions, y_test):
    '''calculates confusion matrix

    Parameters:
    predictions - predicted labels
    y_test - dataframe represents the train genomes true labels

    Returns: true_negative, true_positive, false_positive and false_negative values
    '''

    true_negative, true_positive, false_positive, false_negative = calc_confusion_matrix_indexes(predictions, y_test)

    return len(true_negative), len(true_positive), len(false_positive), len(false_negative)



def calculate_results(predictions, predictions_probs, y_test):
    '''Calculates accuracy results

    Parameters:
    predictions - predictions (list of genomes predictions as 1 or 0)
    y_test - true labels
    predictions_probs - probabilities (output of predict_proba)

    Returns:
    sensitivity, specificity, f1_macro, aupr_auc and roc_auc results

    '''

    true_negative, true_positive, false_positive, false_negative = calc_confusion_matrix(predictions, y_test)
    print('false_positive: ' + str(false_positive) + ',total NHPs: ' + str(true_negative + false_positive))
    print('false_negative: ' + str(false_negative) + ',total HPs: ' + str(true_positive + false_negative))

    specificity = (true_negative) / (true_negative + false_positive)
    sensitivity = (true_positive) / (true_positive + false_negative)

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, predictions_probs)
    aupr_auc = metrics.auc(recall, precision)
    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions_probs)
    roc_auc = metrics.auc(fpr, tpr)
    bacc = (sensitivity + specificity) / 2

    evaluation_results = {'specificity': specificity,
                          'sensitivity': sensitivity,
                          'bacc': bacc,
                          'aupr_auc': aupr_auc,
                          'roc_auc': roc_auc}

    return evaluation_results



def print_train_results(evaluation_results):
    '''Print BACC, sensitivity, specificity, AUPR and AUROC results

    Parameters:
    evaluation_results - dictionary of the sensitivity, specificity, f1_macro, aupr_auc and roc_auc scores

    Returns:
    '''


    ROUND = 2
    bacc = round(evaluation_results['bacc'], ROUND)
    sensitivity = round(evaluation_results['sensitivity'], ROUND)
    specificity = round(evaluation_results['specificity'], ROUND)
    aupr_auc = round(evaluation_results['aupr_auc'], ROUND)
    roc_auc = round(evaluation_results['roc_auc'], ROUND)

    print(f'BAcc: {bacc}')
    print(f'sensitivity: {sensitivity}')
    print(f'specificity: {specificity}')
    print(f'aupr_auc: {aupr_auc}')
    print(f'roc_auc: {roc_auc}')



def dendogram(X, corr_linkage):
    '''Draws dendogram

    Parameters:
    X - dataframe represents the train genomes feature vectors
    corr_linkage - the hierarchical clustering encoded as a linkage matrix

    Returns:
    '''

    fig, ax = plt.subplots(figsize=(12, 8))

    dendro = hierarchy.dendrogram(
        corr_linkage, ax=ax, labels=X.columns.tolist(), leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    fig.tight_layout()
    plt.show()



def herarchial_clustering(X, feature_corr_matrix=None, threshold=1, draw_dendogram=False, method='average'):
    '''preforms herarchial clustering of the features in dataframe X, and returns the resulting clusters

    Parameters:
    X - dataframe represents the train genomes feature vectors
    feature_corr_matrix - matrix of the correlations between each pair of features in X
    threshold - correlation threshold
    draw_dendogram - if true: the function also plots dendogram
    method - linkage method to use for the computation of distance between two clusters

    Returns: cluster_id_to_feature_ids.values - resulting clusters (each cluster is represented
             by the features that it contains)

    '''

    feature_corr_dist_matrix = feature_corr_matrix.loc[X.columns, X.columns]
    feature_corr_dist_matrix = 1 - feature_corr_dist_matrix

    feature_corr_dist_matrix_condensed = ssd.squareform(feature_corr_dist_matrix)

    corr_linkage = hierarchy.linkage(feature_corr_dist_matrix_condensed, method=method)

    if draw_dendogram:
        dendogram(X, corr_linkage)

    cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
    cluster_id_to_feature_ids = {}
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids.setdefault(cluster_id, []).append(feature_corr_dist_matrix.columns[idx])

    return cluster_id_to_feature_ids.values()



def get_chi2_df(x, y):
    '''Creates df that represents the chi-square scores between each feature in X and the target labels.
    Parameters:
    X - dataframe represents the train genomes feature vectors
    y - dataframe represents the train genomes true labels

    Returns:
    chi2_df - df that represents the chi-square scores between each feature in X and the target labels
    '''

    chi2, p_val = feature_selection.chi2(x, y)
    chi2_df = pd.DataFrame(chi2, columns=['chi2'], index=x.columns)
    return chi2_df


#
# def add_feature_to_cluster(clusters, feature, is_add_func):
#
#
#     for cluster in clusters:
#
#         cluster_rep = cluster[0]
#
#         if is_add_func(feature, cluster_rep):
#             cluster.append(feature)
#             return
#
#     clusters.append([feature])
#
#
#
#
# def cluster_features(sorted_features, is_add_func):
#     clusters = []
#     for i, feature in enumerate(sorted_features):
#         if i % 1000 == 0:
#             print(i)
#         add_feature_to_cluster(clusters, feature, is_add_func)
#
#     return clusters





def  phi_coef(x, y):
    '''Calculates phi coefficient between features

    Parameters:
    X - feature x column
    y - feature y column

    Returns:
    corr - phi coefficient value
    '''

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    corr = np.sqrt(chi2 / n)

    return corr



def heatmap_correlated_features(x_train):
    '''Calculates pairwise correlation scores between the features in x_train

    Parameters:
    x_train - training dataframe

    Returns:
    corr - correlation matrix
    '''

    corr = x_train.corr(method=phi_coef)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(corr)
    plt.show()

    #figure = ax.get_figure()
    # figure.savefig('heatmap.png', dpi=400)

    return corr



def optimize_k(min_val, max_val, inc, X_train, y_train, X_valid, y_validation):
    '''Calculates and prints validation results using different values of k for selecting the features with the
     highest chi2 values, and selects the best k value for the first step of the feature selection process according
     to the AUROC results

    Parameters:
    min_val - minimum k value to check
    max_val - maximum k value to check
    inc - k increment size
    X_train -training vectors dataframe
    y_train - training true labels
    X_valid - validation vectors dataframe
    y_validation - validation true labels

    Returns:
    best_k - best k value for the first step of the feature selection process according
     to the AUROC results
    '''


    k_vals = {}
    for k in range(min_val, max_val + 1, inc):
        print(f'________{k}________')
        X_train_fs = perform_fs_first_step(X_train, y_train.Label, k)
        model, evaluation_results = train_and_predict(X_train_fs, y_train.Label, X_valid,
                                                                           y_validation.Label, random_state=0)
        key = f'{k}'
        k_vals[key] = round(evaluation_results['roc_auc'], 2)

    best_AUROC = max([(value, key) for key, value in k_vals.items()])[0]
    best_k = min([key for key, value in k_vals.items() if value==best_AUROC])
    print('\n')
    print(f'*****Best k value by AUROC: {best_k} ********')

    best_k = int(best_k)
    return best_k




def select_feature_from_cluster(chi2_df, curr_cluster):
    '''Selects feature with the highest the highest chi2 value from a cluster

    Parameters:
    curr_cluster - cluster of features
    chi2_df - dataframe that represents the chi2 value between features and the target labels

    Returns:
    selected_feature -  index of the selected feature with the highest chi2 value
    '''

    curr_cluster_ids = [pgfam.csb_id for pgfam in curr_cluster]
    selected_feature =  chi2_df.loc[curr_cluster_ids, 'chi2'].idxmax()

    return selected_feature




def select_features(clusters, x_train, y_train, select_feature_from_cluster_func):
    '''Selects feature for each cluster

    Parameters:
    clusters - clusters of features
    x_train -  training vectors dataframe
    y_train -  training true labels
    select_feature_from_cluster_func - function for selecting feature from cluster

    Returns:
    x_train_selected_features -  x_train, after selecting the final set of features
    '''

    chi2_df = get_chi2_df(x_train, y_train)
    selected_features = [select_feature_from_cluster_func(chi2_df, cluster)
                         for cluster in clusters]
    print('selected_features len: ' + str(len(selected_features)))

    x_train_selected_features = x_train.loc[:, selected_features]

    return x_train_selected_features



def optimize_t(min_val, max_val, inc, X_train_fs, y_train, feature_corr_matrix_train, X_valid, y_validation):
    '''Calculates and prints validation results using different values of t(=threshold) for clustering correlated features in the
    second step of the feature selection process, and selects the best t value according to the AUROC results

    Parameters:
    min_val - minimum t value to check
    max_val - maximum t value to check
    inc -  t increment size
    X_train_fs -training vectors dataframe after the first step of the feature selection process
    feature_corr_matrix_train - features correlation matrix
    y_train - training true labels
    X_valid - validation vectors dataframe
    y_validation - validation true labels

    Returns:
    best_t - best t value for the second step of the feature selection process according
     to the AUROC results
    '''


    t_vals = {}

    for t in np.arange(min_val, max_val, inc):
        print(f'---------------------{round(t, 2)}------------------')

        clusters = herarchial_clustering(X_train_fs, feature_corr_matrix=feature_corr_matrix_train,
                                                           threshold=t)
        X_train_selected = select_features(clusters, X_train_fs, y_train.Label,
                                                             select_feature_from_cluster_by_chi2)

        model, evaluation_results_tvals = train_and_predict(X_train_selected, y_train.Label,
                                                                                 X_valid, y_validation.Label)

        key = f'{round(t, 2)}'
        t_vals[key] = round(evaluation_results_tvals['roc_auc'], 2)

    best_AUROC = max([(value, key) for key, value in t_vals.items()])[0]
    best_t = max([key for key, value in t_vals.items() if value==best_AUROC])
    print('\n')
    print(f'*****Best t value by AUROC: {best_t} **********')

    best_t = float(best_t)
    return(best_t)



def perform_fs_second_step(X_train_valid_fs, feature_corr_matrix, t, y_train_validation):
    ''' Removes highly correlated and redundant features from the features set by clustering correlated
        features and selecting a representative from each cluster.

    Parameters:
    X_train_valid_fs - dataframe represents the training genomes feature vectors after selecting the k features in the
                       first step of the feature selection process
    feature_corr_matrix - features correlation matrix
    t - clustering threshold
    y_train_validation - dataframe represents the training genomes labels

    Returns:
    X_train_valid_clust - Dataframe that represents the training genomes feature vectors after removing
                          highly correlated features.

    '''

    clusters = herarchial_clustering(X_train_valid_fs, feature_corr_matrix=feature_corr_matrix, threshold=t)
    X_train_valid_clust = select_features(clusters, X_train_valid_fs, y_train_validation.Label,
                                                                 select_feature_from_cluster_by_chi2)

    return X_train_valid_clust

#
# def is_high_corr(feature_corr_matrix, feature1, feature2, threshold=0.9):
#     return feature_corr_matrix.loc[feature1, feature2] >= threshold



def select_feature_from_cluster_by_chi2(chi2_df, curr_cluster):
    return chi2_df.loc[curr_cluster, 'chi2'].idxmax()




############# Predict pathogenicity labels with existing model#########

def predict_with_existing_model(classifier, classifiers_features, input_genumes_file, output_files_str):
    ''' Predicts and prints the pathogenicity label of genomes with existing classification model.

    Parameters:
    classifier - existing classification model
    classifiers_features - dioctionary represents the feature set of the classifier
    input_genumes_file - file of input genomes (each genomes is represented as a sequence of PATRIC Protein Global
                        Families (PGFams) identifiers)
    output_files_str - string for   output files naming

    Returns:
    predictions - predictions (list of genomes predictions as 1 or 0)
    predictions_probs - probabilities (output of predict_proba)

    '''
    with open(classifiers_features) as f:
        pgfams_dict = json.load(f)

    output_vec_file = output_files_str + '_vecs.json'
    util.create_features_vec(pgfams_dict, [input_genumes_file], output_vec_file)
    X_genomes = util.read_pgfams_data_json(output_vec_file, classifiers_features)
    predictions, predictions_probs = predict(X_genomes, classifier)

    pathogenicity_preds = []
    for i in predictions:
        if i == 0:
            pathogenicity_preds.append('NHP')
        else:
            pathogenicity_preds.append('HP')

    print('\n' + '***** Predictions ******')
    for idx, genome_name in enumerate(X_genomes.index):
        print(f'Genome ID: {genome_name}:, prediction: {pathogenicity_preds[idx]}')

    return predictions, predictions_probs

