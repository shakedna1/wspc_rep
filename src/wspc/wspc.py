import pandas as pd
import pkgutil
from . import feature_selection
from . import reader


PRED = 'Prediction'
PROB = 'Probability'
TRAINED_MODEL_PATH = 'model/WSPC_model.pkl'


def predict(X, model):
    """
    Predicts pathogenicity of input genomes represented by X

    Parameters
    ----------
    X - feature vectors of the input genomes
    model - trained model for the prediction

    Returns
    ----------
    results - pd.Dataframe object with the prediction results
    """

    y_pred, y_prob = model.predict(X), model.predict_proba(X)[:, 1]

    result = pd.DataFrame(data={PRED: y_pred, PROB: y_prob}, columns=[PRED, PROB], index=X.index)

    return result


def fit(X, y, k=450, threshold=0.18):
    """
    Fits classifier to an input training genomes

    Parameters
    ----------
    X - feature vectors of the input training genomes
    y - labels of the input training genomes

    Returns
    ----------
    pipeline - pd.Dataframe object with the prediction results
    """

    pipeline = feature_selection.get_fs_pipeline(k=k, threshold=threshold)
    pipeline.fit(X, y)

    return pipeline


def read_genomes(path, y=None):
    """
    Reads all genomes information from an input directory with genome *.txt files or a merged input *.fasta file

    Parameters
    ----------
    path - path to an input directory with genome *.txt files or a merged input *.fasta file
    y - if provided, reindex genomes according to y.index

    Returns
    ----------
    pd.Series object that represents all the input genomes in the directory
    """

    genomes = reader.read_genomes(path)
    if y is not None:
        genomes = genomes.reindex(y.index)

    return genomes


def read_labels(path, X=None):
    """
    Reads all genomes labels from *.csv file

    Parameters
    ----------
    path - path to *.csv file with labels
    X - if provided, reindex labels according to X.index

    Returns
    ----------
    labels - series object with the genomes labels
    """

    labels = reader.read_labels(path)

    if X is not None:
        labels = labels.reindex(X.index)
    return labels


def load_model(model_path=None):
    """
    Loads model from path to a saved model in a *.pkl file. If not provided, saved pre-trained model will be used

    Parameters
    ----------
    model_path - path to a saved model in a *.pkl file. If not provided, saved pre-trained model will be used

    Returns
    ----------
    loaded model
    """

    if not model_path:
        model_str = pkgutil.get_data(__name__, TRAINED_MODEL_PATH)
        return reader.load_model_str(model_str)

    return reader.load_model(model_path)

