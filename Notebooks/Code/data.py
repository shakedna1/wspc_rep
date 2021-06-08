import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DATE_INSERTED = 'Date Inserted'
GENOME_ID = 'Genome ID'
LABEL = 'Label'
HP = 'HP'
NHP = 'NHP'


class FastaReader:

    @staticmethod
    def read(file_path):
        genome_to_pgfams = {}

        with open(file_path) as f:
            for line in f:
                if line.startswith('>'):
                    genome_id = line.strip()[1:]

                else:
                    pgfam_id = line.strip()
                    genome_to_pgfams.setdefault(genome_id, []).append(pgfam_id)

        return genome_to_pgfams


class MetadataReader:

    @staticmethod
    def read(file_path):

        metadata = pd.read_csv(file_path, dtype=str).set_index(GENOME_ID)
        if DATE_INSERTED in metadata.columns:
            metadata[DATE_INSERTED] = pd.to_datetime(metadata[DATE_INSERTED])

        return metadata


class GenomesData:
    """
    Read and store genomes data and metadata

    Parameters
    ----------
    genomes_path : path to a fasta file containing gene family ids of each genome
    metadata_path : path to a csv file with the metadata associated to each genome and its pathogenicity label

    Attributes
    ----------
    genome_to_genes : a dictionary with genome id as key, and a list of gene family ids as value
    metadata: a DataFrame where the index is the genome ids

    genomes: metadata.index
    data: a Series of strings, where each string is a sequence of gene family ids in the corresponding genome, ordered
    according to metadata.index
    y: a Series of binary labels ordered according to metadata.index

    """

    label_to_int = {HP: 1, NHP: 0}

    def __init__(self, genomes_path, metadata_path):

        self.genome_to_genes = FastaReader.read(genomes_path)
        self.metadata = MetadataReader.read(metadata_path)

        self._y = self.convert_labels()

        assert all(self.genomes.isin(self.genome_to_genes.keys()))

    @property
    def genomes(self):
        """Returns a series of genome ids"""
        return self.metadata.index

    @property
    def data(self):
        """Returns a series of strings, each string is a sequence of genes separated by space, corresponding to the
        order of self.genomes"""
        return self.convert_genomes_to_strings()

    @property
    def y(self):
        """Returns a series of binary labels of self.genomes"""
        return self._y

    def convert_genomes_to_strings(self):
        """Returns a series of strings, each string is a sequence of genes separated by space (genome)"""
        return pd.Series([' '.join(self.genome_to_genes[genome]) for genome in self.genomes],
                         dtype="string", index=self.genomes)

    def convert_labels(self):
        """Converts the genome labels to binary labels according to GenomesData.label_to_int"""
        return self.metadata[LABEL].apply(lambda label: GenomesData.label_to_int.get(label, -1))

    def vectorize_data(self, vectorizer=CountVectorizer(lowercase=False, binary=True)):
        """
        Converts the data strings into vectors using vectorizer.
        :param vectorizer: a vectorizer used for data transformation
        :return: DataFrame where rows are vectors of self.genomes and columns are vectorizer.get_feature_names()
        """
        encoding = vectorizer.fit_transform(self.data)
        return pd.DataFrame(encoding.toarray(), columns=vectorizer.get_feature_names(), index=self.genomes)

    def __len__(self):
        return len(self.genomes)


