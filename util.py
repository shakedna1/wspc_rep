import json
import pandas as pd
import pickle



def create_pgfams_dict(input_files):
    '''Creates dictionary of all pgfams in the input files

    Parameters:
    input_file_path - training genomes pgfams file

    Returns:
    pgFam_dict - dictionary with k=pgfam_id, v=genomes contain the relevant pgfam

    '''

    pgFam_dict = {}
    ind = 0
    for input_file_path in input_files:
        with open(input_file_path, 'r') as input_file:
            for line in input_file:
                if line.startswith('>'):
                    genome_id = line.split('>')[1].split('\n')[0]
                else:
                    pgfam_id = line.split('\n')[0]
                    if pgfam_id not in pgFam_dict.keys():
                        names = {genome_id}
                        pgFam_dict[pgfam_id] = (ind, names)
                        ind += 1
                    else:
                        names = pgFam_dict[pgfam_id][1]
                        old_ind = pgFam_dict[pgfam_id][0]
                        names.add(genome_id)
                        pgFam_dict[pgfam_id] = (old_ind, names)
        input_file.close()

    return pgFam_dict



def write_dict(pgFam_dict, output_name, min_num=10):
    '''Writes pgfams dictionary information to a json file.
    For better space and time efficiency, pgFam_dict and the resulting json file will contain only
    pgfams which are contained in a number >= min_num of genome. (Since pgfams that contained in a very small
    number of genomes have low chi-square values, these pgfams will be removed at the first step of the
    feature selection proceess. Hence, in order to save time and space, these features can be ignored when creating
    the initial features vecotrs. The value of min_num should be selected carfully according to the size of the dataset.
    The deafult value, min_num=10 fits the WSPC training set.)

    Parameters:
    pgFam_dict - training genomes file
    output_name - name of the output file
    min_num - only pgfams which are contained in a number >= min_num of genomes will be included in the json file.

    Returns:
    pgFam_dict - final pgfams dictionary of pgfams which are contained in >=min_num genomes
    '''

    print('Creating features dictionary...')

    pgFam_dict_f = {}
    new_ind = 1
    for k, v in pgFam_dict.items():
        genomes_list = v[1]
        if len(genomes_list) >= min_num:
            pgFam_dict_f[k] = (new_ind, list(set(genomes_list)))
            new_ind += 1

    inds = [pgFam_dict_f[i][0] for i in pgFam_dict_f.keys()]
    genomes_lists = [pgFam_dict_f[i][1] for i in pgFam_dict_f.keys()]
    all_list = [inds, list(pgFam_dict_f.keys()), genomes_lists]
    with open(output_name + '.json', 'w') as json_file:
        json.dump(all_list, json_file)

    print('Creating features dictionary - Done')

    return pgFam_dict_f



def create_features_vec(pgFam_vec, input_files, output_file):
    '''Creates feature vectors for the genomes in the input files, according to pgFam_dict

    Parameters:
    pgFam_dict - pgfams dictionary
    input_files - array of all genomes files (train, test etc)
    output_file = output file name

    Returns:
    '''

    print('Creating features vectors..')


    features_vec_dict = {}
    num_finished = -1
    for input_file_path in input_files:
        with open(input_file_path, 'r') as input_file:
            genome_pgfams = []
            for line in input_file:
                if line.startswith('>'):
                    for pgfam_id in genome_pgfams:
                        i = pgFam_vec.index(pgfam_id)
                        features_vec_dict[genome_id][i] = 1

                    genome_id = line.split('>')[1].split('\n')[0]
                    features_vec_dict[genome_id] = [0] * len(pgFam_vec)
                    genome_pgfams = []
                else:
                    pgfam_id = line.split('\n')[0]
                    if pgfam_id in pgFam_vec:
                        genome_pgfams.append(pgfam_id)
            for pgfam_id in genome_pgfams:
                i = pgFam_vec.index(pgfam_id)
                features_vec_dict[genome_id][i] = 1
        input_file.close()

    vecs = [features_vec_dict[i] for i in features_vec_dict.keys()]
    features_vecs_data = [list(features_vec_dict.keys()), vecs]
    with open(output_file, 'w') as new_file:
        json.dump(features_vecs_data, new_file)

    print('Creating features vectors - Done')

    return features_vecs_data



def read_pgfams_data_json(features_json, order_json=None):
    '''Reads json with feature vectors information in order creates dataframe
    of the genomes and their feature vectors. The features order will be according to the
    order that order_json specify.

    Parameters:
    features_json - json file that contain the features vectors information

    Returns:
    '''

    with open(features_json) as f:
        genomes_names, fet_vecs = json.load(f)
    with open(order_json) as f:
            pgfams_order = json.load(f)

    X = pd.DataFrame(data=fet_vecs, columns=pgfams_order, index=genomes_names)
    X.index = X.index.astype(dtype='str')

    return X



def create_x_from_pgfams_data(features_vecs_data, pgfams_dict):
    '''Creates X dataframe of feature vectors according to features_vecs_data. This dataframe is used for
     training and evaluation (the dataset is splitted later for training genomes and validation genomes as part of
     the training pipline).

    Parameters:
    features_vecs_data - features vectors information
    pgfams_dict - pgfams dictionary

    returns: X - the resulting dataframe
    '''

    genomes_names = features_vecs_data[0]
    fet_vecs = features_vecs_data[1]
    X = pd.DataFrame(data=fet_vecs, columns=pgfams_dict, index=genomes_names)
    X.index = X.index.astype(dtype='str')

    return X



def create_x_df(train_files , test_files, dict_f_name, fet_vecs_f_name, min_num=10):
    '''Creates dictionary of pgfams in train files, and than creates dataframe of features vectors for each genome in
    each files in train_files + test_files. This dataframe is used for training and evaluation (the dataset is splitted
    later for training genomes and validation genomes as part of the training pipline).

   Parameters:
   train_files - files of train genomes
   test_files - files of test/validation genomes
   dict_f_name - name for the pgfams dictionary file
   fet_vecs_f_name - name for the feature vectors file
   min_num - only pgfams which are contained in a number >= min_num of genomes will be included in the json file
    and the final pgfams dictionary (used for better time and space efficiency)

    Returns:
    '''

    pgFam_dict = create_pgfams_dict(train_files)
    print(f'total pgfams: {len(pgFam_dict)}')

    f_pgFam_dict = write_dict(pgFam_dict, dict_f_name, min_num=min_num)
    print(f'total pgfams in final dict: {len(f_pgFam_dict)}')

    input_files = train_files + test_files
    all_list = create_features_vec(list(f_pgFam_dict.keys()), input_files, fet_vecs_f_name)
    X = create_x_from_pgfams_data(all_list, f_pgFam_dict)

    return X



def load_model(model_path):
    '''Loads existing model

    Parameters:
    model_path - path to the model file

    Returns: loaded model
    '''

    with open(model_path, 'rb') as f:
        return pickle.load(f)



def create_labels_df(csv_file_name):
    '''Creates dataframe indicating the pathogenicity label of each genome in the file 'csv_file_name'

    Parameters:
    csv_file_name - path to the genomes information file


    Returns:
    labels_df - dataframe indicating the pathogenicity label of each genome in the file 'csv_file_name'
    '''

    df = pd.read_csv(csv_file_name, dtype=str).set_index('Genome ID')
    labels_df = pd.DataFrame(df['Label'].apply(lambda label: 1 if label == 'HP' else 0),
                             index = df.index, columns = ['Label'])

    return labels_df



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



def pgfams_order_in_vec(df, order):
    '''Sorts the dataframe column (features) according to specific order

    Parameters:
    df - dataframe to sort
    order - order for sorting

    Returns:
    sorted_df - sorted df
    '''

    with open(order) as f:
         pgfams_order = json.load(f)
    sorted_df = df.reindex(columns=pgfams_order)

    return sorted_df