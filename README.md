# WSPC

Installing the package:
```buildoutcfg
pip install wspc
```

## Dependencies
- Python >=3.6
- Packages: pandas, numpy, scikit-learn, scipy

## Command Line

> In windows: make sure that the python "Scripts\\" directory is added to PATH, 
>so that the package can be executed as a command 

Usage:

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

Predict:

```buildoutcfg
wspc -m predict -i path_to_input_genomes
```


Train:

```buildoutcfg
wspc -m fit -i path_to_input_genomes -l path_to_labels -k 450 -t 0.18
```

