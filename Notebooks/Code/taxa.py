from ete3 import NCBITaxa

ncbi = NCBITaxa()

"""Consts"""
UNCLASSIFIED = 'unclassified'
SPECIES = 'species'
GENUS = 'genus'


def get_ranks(taxid):
    """
    Returns a dictionary of rank : list of values, possible ranks are: species, genus, phylum etc.
    :param taxid: NCBI tax ID
    :return: A dictionary of rank : list of values
    """
    lineage = ncbi.get_lineage(taxid)

    id_to_rank = ncbi.get_rank(lineage)

    id_to_name = ncbi.get_taxid_translator(lineage)

    ranks = {}
    for id_num, rank in id_to_rank.items():
        ranks.setdefault(rank, []).append(id_to_name[id_num])

    return ranks


def get_tax(genome, tax_type):
    """
    Returns a list of names that belong to tax_type
    :param genome: genome ID, e.g. 28450.1894. 28450 is the tax id
    :param tax_type: the requested tax_type, e.g. species
    :return: a list of names that belong to tax_type
    """
    taxa_id = int(genome.split('.')[0])
    return get_ranks(taxa_id).get(tax_type, [])


def get_first_tax_desc(genome, tax_type):
    """
    Returns the first name in the list that belongs to tax_type
    :param genome: genome ID, e.g. 28450.1894. 28450 is the tax id
    :param tax_type: the requested tax_type, e.g. species
    :return: the first name in the list that belongs to tax_type if exists, else '-'
    """
    ranks = get_tax(genome, tax_type)

    if ranks:
        return ranks[0]

    return '-'


def get_unclassified_species(genome):
    """Return unclassified species name if species is unclassified, else None"""

    no_rank_tax = get_tax(genome, 'no rank')

    unclassified_term = None
    for term in no_rank_tax:
        if UNCLASSIFIED in term:
            unclassified_term = term

    return unclassified_term


def calc_tax_broadness(genome_ids, tax_type):
    """
    Calculates number of different names belonging to tax_type
    :param genome_ids: a list of genome ids
    :param tax_type: tax type, e.g. genus
    :return: number of different names belonging to tax_type
    """

    tax_names = set(get_first_tax_desc(genome, tax_type) for genome in genome_ids)

    return len(tax_names)


def get_genomes_tax(genomes, tax):
    return set(get_first_tax_desc(genome, tax) for genome in genomes)


def get_genomes_species(genomes):
    return set(get_first_tax_desc(genome, SPECIES) for genome in genomes if not get_unclassified_species(genome))


def get_novel_species(train_genomes, test_genomes):

    train_species = get_genomes_species(train_genomes)
    test_species = get_genomes_species(test_genomes)

    return test_species - train_species


def get_genomes_with_tax(genomes, tax_names, tax):

    return [genome for genome in genomes if get_first_tax_desc(genome, tax) in tax_names]

