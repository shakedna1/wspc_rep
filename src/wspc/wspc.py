import pandas as pd
import pkgutil
from . import feature_selection
from . import reader

PRED = 'Prediction'
PROB = 'Probability'
TRAINED_MODEL_PATH = 'model/WSPC_model.pkl'


def predict(X, model):

    y_pred, y_prob = model.predict(X), model.predict_proba(X)[:, 1]

    result = pd.DataFrame(data={PRED: y_pred, PROB: y_prob}, columns=[PRED, PROB], index=X.index)

    return result


def fit(X, y, k=450, threshold=0.18):

    pipeline = feature_selection.get_fs_pipeline(k=k, threshold=threshold)
    pipeline.fit(X, y)

    return pipeline


def read_genomes(path):

    return reader.read_genomes(path)


def read_labels(path, X):

    labels = reader.read_labels(path)
    return labels.reindex(X.index)


def load_model(model_path=None):

    if not model_path:
        model_str = pkgutil.get_data(__name__, TRAINED_MODEL_PATH)
        return reader.load_model_str(model_str)

    return reader.load_model(model_path)

