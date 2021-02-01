import json
import pickle


def load_model(model_path):
    '''Loads existing model

    Parameters:
    model_path - path to the model file

    Returns: loaded model
    '''

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def save_model(model_filename, model, X_train):
    '''Saves model to pkl file, and the final feature set of the model to json file'

    Parameters:
    model_filename - string used for naming the function output files
    model - model (classifier) to save
    X_train - dataframe of the training genomes with the final features vectors. Used in order to extract the final
             feature set of the model.


    Returns:
    '''

    with open(model_filename + '.pkl', 'wb') as file:
        pickle.dump(model, file)

    final_features = list(X_train.columns)
    with open(model_filename + '_features.json', 'w') as file:
        json.dump(final_features, file)
