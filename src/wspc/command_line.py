import argparse
import pickle

from os import path
from . import wspc
from datetime import datetime

PREDICT = 'predict'
FIT = 'fit'
TRAINED_MODEL_NAME = 'trained_model.pkl'
PREDICTIONS_FILE_NAME = 'predictions.csv'


def write_fitted_model(model, output_path):
    """
    Writes fitted model to pickle file

    Parameters
    ----------
    model - fitted model
    output_path - path to output directory
    """

    model_path = path.join(output_path, TRAINED_MODEL_NAME)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def write_predictions(predictions, output_path):
    """
    Writes model predictions to csv file

    Parameters
    ----------
    predictions - dataframe of the model predictions
    output_path - path to output directory
    """ 
    pred_path = path.join(output_path, PREDICTIONS_FILE_NAME)
    predictions.to_csv(pred_path)


def predict(args):
    """
    Predicts pathogenicity potentials 
    
    Parameters
    ----------
    args.i - input directory with genome *.txt files or a merged input *.fasta file
    args.model_path - path to a saved model in a *.pkl file. If not provided, saved pre-trained model will be used
    args.output - output directory, default current directory
    """

    X = wspc.read_genomes(args.i)

    model = wspc.load_model(args.model_path)

    predictions = wspc.predict(X, model)
    write_predictions(predictions, args.output)


def fit(args):
    """
    Fits new model 
    
    Parameters
    ----------
    args.i - path to input directory with genome *.txt files or a merged input *.fasta file
    args.labels_path - path to *.csv file with labels
    args.k - parameter for training - selecting k-best features using chi2
    args.t - parameter for training - clustering threshold
    args.output - output directory, default current directory
    """

    if not args.labels_path:
        raise ValueError('Please provide a path to labels using --labels_path')

    print(f"{datetime.now().strftime('%H:%M:%S')} Reading labels")
    y = wspc.read_labels(args.labels_path)
    print(f"{datetime.now().strftime('%H:%M:%S')} Read {y.shape[0]} labels")

    print(f"{datetime.now().strftime('%H:%M:%S')} Reading genomes")
    X = wspc.read_genomes(args.i, y)
    print(f"{datetime.now().strftime('%H:%M:%S')} Read {X.shape[0]} genomes")

    print(f"{datetime.now().strftime('%H:%M:%S')} Started training")
    model = wspc.fit(X, y, args.k, args.t)

    print(f"{datetime.now().strftime('%H:%M:%S')} Writing fitted model to {path.join(args.output, TRAINED_MODEL_NAME)}")
    write_fitted_model(model, args.output)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', choices=[PREDICT, FIT], default=PREDICT)
    parser.add_argument('-i', required=True, help='input directory with genome *.txt files or a merged '
                                                  'input *.fasta file')
    parser.add_argument('-o', '--output', help='output directory, default current directory', default='')
    parser.add_argument('-l', '--labels_path', help='path to *.csv file with labels')
    parser.add_argument('--model_path', help='path to a saved model in a *.pkl file. If not provided, saved pre-trained'
                                             ' model will be used')
    parser.add_argument('-k', type=int, help='parameter for training - selecting k-best features using chi2',
                        default=450)
    parser.add_argument('-t', type=float, help='parameter for training - clustering threshold', default=0.18)

    args = parser.parse_args()

    modes = {PREDICT: predict,
             FIT: fit}

    modes[args.mode](args)


if __name__ == "__main__":

    main()

