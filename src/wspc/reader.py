import os
import pickle
import pandas as pd
from . import feature_selection

PATRIC_FILE_EXTENSION = '.txt'
PGFAM_COL = 'pgfam'
GENOME_ID = 'Genome ID'
LABEL = 'Label'
HP = 'HP'
NHP = 'NHP'


def read_merged_file(file_path):

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


def read_genome_file(file_entry):

    pgfams = pd.read_csv(file_entry, usecols=[PGFAM_COL], sep='\t').dropna()
    pgfams = ' '.join(list(pgfams[PGFAM_COL]))

    return pgfams


def read_files_in_dir(dir_path):

    genomes_ids = []
    genomes_pgfams = []

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(PATRIC_FILE_EXTENSION):
                genome_id = entry.name.split(PATRIC_FILE_EXTENSION)[0]
                pgfams = read_genome_file(entry)

                genomes_ids.append(genome_id)
                genomes_pgfams.append(pgfams)

    return pd.Series(genomes_pgfams, index=genomes_ids, dtype="string")


def read_genomes(path):

    if os.path.isdir(path):
        return read_files_in_dir(path)
    elif os.path.isfile(path):
        return read_merged_file(path)


def read_labels(path):

    label_to_int = {HP: 1, NHP: 0, '1': 1, '0': 0}

    labels_df = pd.read_csv(path, dtype=str).set_index(GENOME_ID)
    labels = labels_df[LABEL].apply(lambda label: label_to_int.get(label.upper(), -1))

    return labels


def load_model(model_path):
    """Loads existing model

    Parameters:
    model_path - path to the model file

    Returns: loaded model
    """

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_model_str(data_str):

    return pickle.loads(data_str)
