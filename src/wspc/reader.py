import os
import pickle
import pandas as pd
from . import feature_selection

PATRIC_FILE_EXTENSION_TO_PGFAM_COL = {'.txt' : 'pgfam', '.tab' : 'pgfam_id'}
GENOME_ID = 'Genome ID'
LABEL = 'Label'
HP = 'HP'
NHP = 'NHP'


def read_merged_file(file_path):
    """
    Reads genomes merged file into pd.Series object

    Parameters
    ----------
    file_path - path to a merged input *.fasta file

    Returns
    ----------
    pd.Series object that represents the all input genomes of the merged file
    """

    genomes_order = []
    genome_to_pgfams = {}

    with open(file_path) as f:
        genome_id = ''
        for line in f:
            if line.startswith('>'):
                genome_id = line.strip()[1:]
                genomes_order.append(genome_id)
            else:
                pgfam_id = line.strip()
                genome_to_pgfams.setdefault(genome_id, []).append(pgfam_id)

    genomes_pgfams = [' '.join(genome_to_pgfams[genome]) for genome in genomes_order]

    return pd.Series(genomes_pgfams, index=genomes_order, dtype="string")


def read_genome_file(file_entry, pgfam_col):
    """
    Reads a single genome file and returns its contained pgfams

    Parameters
    ----------
    file_entry - entry to an input genome file

    Returns
    ----------
    pd.Series object that represents all the input genomes in the directory
    """

    pgfams = pd.read_csv(file_entry, usecols=[pgfam_col], sep='\t').dropna()
    pgfams = ' '.join(list(pgfams[pgfam_col]))

    return pgfams


def read_files_in_dir(dir_path):
    """
    Reads all genomes *.txt/*.tab files in a directory into pd.Series object

    Parameters
    ----------
    dir_path - a path to an input directory with genome *.txt/*.tab files

    Returns
    ----------
    pd.Series object that represents all the input genomes in the directory
    """

    genomes_ids = []
    genomes_pgfams = []

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                for extension, pgfam_col in PATRIC_FILE_EXTENSION_TO_PGFAM_COL.items():
                    if entry.name.endswith(extension):
                        genome_id = entry.name.split(extension)[0]
                        pgfams = read_genome_file(entry, pgfam_col)

                        genomes_ids.append(genome_id)
                        genomes_pgfams.append(pgfams)
                        break

    return pd.Series(genomes_pgfams, index=genomes_ids, dtype="string")


def read_genomes(path):
    """
    Reads all genomes information from an input directory with genome *.txt/*.tab files or a merged input *.fasta file

    Parameters
    ----------
    path - a path to an input directory with genome *.txt/*.tab files or a merged input *.fasta file

    Returns
    ----------
    pd.Series object that represents all the input genomes in the directory
    """

    if os.path.isdir(path):
        return read_files_in_dir(path)
    elif os.path.isfile(path):
        return read_merged_file(path)


def read_labels(path):
    """
    Reads csv file with labels from the given path

    Parameters
    ----------
    path -  path to *.csv file with labels

    Returns
    ----------
    labels - series object with the genomes labels
    """

    label_to_int = {HP: 1, NHP: 0, '1': 1, '0': 0}

    labels_df = pd.read_csv(path, dtype=str).set_index(GENOME_ID)
    labels = labels_df[LABEL].apply(lambda label: label_to_int.get(label.upper(), -1))

    return labels


def load_model(model_path):
    """
    Loads existing model from a model_path

    Parameters
    ----------
    model_path - path to the model file

    Returns
    ----------
    loaded model
    """

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_model_str(data_str):
    """
    Loads existing model from data_str

    Parameters
    ----------
    data_str - pickled representation data of the model

    Returns
    ----------
    loaded model
    """

    return pickle.loads(data_str)
