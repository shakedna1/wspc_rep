import pickle
from pathlib import Path


def load_model(model_path):
    """Loads existing model

    Parameters:
    model_path - path to the model file

    Returns: loaded model
    """

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def save_model(path, model_filename, model):
    """Saves model to pkl file'

    Parameters:
    model_filename - string used for naming the function output files
    model - model (classifier) to save

    Returns:
    """

    out_path = Path(path)
    model_filename = model_filename + '.pkl'
    out_path = out_path / model_filename

    with open(out_path, 'wb') as file:
        pickle.dump(model, file)
