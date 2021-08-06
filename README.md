# WSPC

## Installation and dependencies

WSPC package can be installed via one of the following options:

1. Using pip install (via the command line):
```buildoutcfg
pip install wspc
```
2. Using conda (on conda enviroment):
```buildoutcfg
conda install -c zivukelsongroup wspc
```

Dependencies:

- Python >=3.6
- Packages: pandas, numpy, scikit-learn, scipy

## Command Line

> In windows: make sure that the python "Scripts\\" directory is added to PATH,
>so that the package can be executed as a command

### Usage:

```buildoutcfg
usage: wspc [-h] [-m {predict,fit}] -i I [-o OUTPUT] [-l LABELS_PATH] [--model_path MODEL_PATH] [-k K] [-t T]

optional arguments:
  -h, --help            show this help message and exit
  -m {predict,fit}, --mode {predict,fit}
  -i I                  input directory with genome *.txt files or a merged input *.fasta file
  -o OUTPUT, --output OUTPUT
                        output directory, default current directory
  -l LABELS_PATH, --labels_path LABELS_PATH
                        path to *.csv file with labels
  --model_path MODEL_PATH
                        path to a saved model in a *.pkl file. If not provided, saved pre-trained model will be used
  -k K                  parameter for training - selecting k-best features using chi2
  -t T                  parameter for training - clustering threshold
```

### Predict:

You can predict the pathogenicity potentials of group of genomes using a saved model in a *.pkl file.
If a path is not provided, saved pre-trained model will be used.
The WSPC pre-trained model can be found in https://github.com/shakedna1/wspc_rep/blob/main/src/wspc/model/WSPC_model.pkl.

```buildoutcfg
wspc -m predict -i path_to_input_genomes
```


### Train:

Train a new model using the fit command.

You can train a new model using the same k (selecting k-best features using chi2)
and t (clustering threshold) values of WSPC (450 and 0.18 respectively) or using a
different values of your choice.

```buildoutcfg
wspc -m fit -i path_to_input_genomes -l path_to_labels -k 450 -t 0.18
```

## Reconstruction of Training and Prediction on the dataset from the paper

1. Download and extract the WSPC dataset (WSPC train set & WSPC test set) from https://github.com/shakedna1/wspc_rep/raw/main/Data/train_test_datasets.zip
    In Ubuntu:
    ```buildoutcfg
       wget https://github.com/shakedna1/wspc_rep/raw/main/Data/train_test_datasets.zip
       unzip train_test_datasets.zip
    ```

2. Train:
    ```buildoutcfg
        wspc -m fit -i train_genomes.fasta -l train_genomes_info.csv -k 450 -t 0.18
    ```
   The file trained_model.pkl will be saved in the same directory (or in the directory provided through
    the -o argument)

3. Test:
    ```buildoutcfg
       wspc -m predict -i test_genomes.fasta --model_path trained_model.pkl
    ```
   The file predictions.csv will contain the predictions

## Running WSPC as a python module

Below are a detailed running examples of WSPC as a python module:

### 1. Train a new model and predict genomes pathogenicity using the new model:



#### Imports:
```
import wspc
```

#### Train a new model:
```
X_train = wspc.read_genomes(path_to_genomes)
y = wspc.read_labels(path_to_labels, X_train)

model = wspc.fit(X_train, y, k=450, t=0.18)
```

#### Predict pathogenicity:
```
X_test = wspc.read_genomes(path_to_genomes)
predictions = wspc.predict(X_test, model)
```

### 2. Predict genomes pathogenicity using an exiting model:

#### Imports:
```
import wspc
```

#### Load a pre-trained model:


model_path - path to a saved model in a *.pkl file. If not provided, saved pre-trained model will be used
```
model = wspc.load_model(model_path)
```

#### Predict pathogenicity:
```
X_test = wspc.read_genomes(path_to_genomes)
predictions = wspc.predict(X_test, model)
```

## WSPC input:

WSPC can handle two types of input:
1. Input directory with genome *.txt files:


2. Merged input *.fasta file:
For the merged file exact format, look at the file: https://github.com/shakedna1/wspc_rep/blob/changes_30_7/Data/train_genomes.fasta.
Example of the fasta file content:
```
>1346.123
PGF_10048015
PGF_00062045
PGF_00409415
PGF_00766022
PGF_02011026
X
X
X
PGF_07480521
PGF_01162199
PGF_03475877
PGF_00876106
PGF_06473395
PGF_06429692
PGF_00007012
PGF_04788810
```
* 1346.123 - genome name, the lines bellow represents the genome sequence of pgfams annotations. X represents a missing/un-annotated gene.



In order to predict the pathogenicity of a list of genomes,

### Obtain PATRIC Global Protein Families (PGFams) annotations for new sequenced genome:

PATRIC Provides Global Protein Families (PGFams) annotations service for new genomes.
In order to generate PGFams annotations file for a new sequenced genome:

1. Use PATRIC's Genome Annotation Service: https://patricbrc.org/app/Annotation.
For detailed instructions, Follow the instructions under the PATRIC genome annotations service documentation:
https://docs.patricbrc.org/user_guides/services/genome_annotation_service.html

2. Download the resulting "Taxonomy name + label".txt file (click on view, then download. The "Taxonomy name + label" is the genome name).

3. Extract pgfam annotations using the column "pgfam".


