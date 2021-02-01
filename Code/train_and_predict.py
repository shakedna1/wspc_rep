import util

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from scipy.cluster import hierarchy
from scipy.stats import chi2_contingency
from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline



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



def train_and_predict(X_train, y_train, X_test, y_test, features, random_state=0):
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

    rs_model = Pipeline(steps=[('vectorize', CountVectorizer(lowercase=False, binary=True, vocabulary=features)),
                               ('rf', RandomForestClassifier(random_state=random_state))])

    rs_model.fit(X_train, y_train)

    evaluation_results = predict_and_print_results(X_test, y_test, rs_model)

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


############# Predict pathogenicity labels with existing model#########
# TODO: Modify function
def predict_with_existing_model(classifier, classifiers_features, input_genumes_file, output_files_str):
    ''' Predicts the pathogenicity label of genomes with existing classification model,
     and returns dataframe of the results

    Parameters:
    classifier - existing classification model
    classifiers_features - dioctionary represents the feature set of the classifier
    input_genumes_file - file of input genomes (each genomes is represented as a sequence of PATRIC Protein Global
                        Families (PGFams) identifiers)
    output_files_str - string for   output files naming

    Returns:
    results df - dataframe that contains the predictions and the predicted probabilities for the input genomes
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

    d = {'Genome ID': X_genomes.index ,'prediction': pathogenicity_preds, 'predictions_prob': predictions_probs}
    output_df = pd.DataFrame(data=d)
    output_df = output_df.set_index('Genome ID')

    return output_df

