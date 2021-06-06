import argparse
import pickle
from os import path
from . import wspc


PREDICT = 'predict'
FIT = 'fit'


def write_fitted_model(model, output_path):

    model_path = path.join(output_path, 'trained_model.pkl')

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def write_predictions(predictions, output_path):

    pred_path = path.join(output_path, 'predictions.csv')
    predictions.to_csv(pred_path)


def predict(args):

    X = wspc.read_genomes(args.i)

    model = wspc.load_model(args.model_path)

    predictions = wspc.predict(X, model)
    write_predictions(predictions, args.output)


def fit(args):

    X = wspc.read_genomes(args.i)

    if not args.labels_path:
        raise ValueError('Please provide a path to labels using --labels_path')

    y = wspc.read_labels(args.labels_path, X)
    model = wspc.fit(X, y, args.k, args.t)

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
