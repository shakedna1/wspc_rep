from ete3 import NCBITaxa

ncbi = NCBITaxa()

"""Consts"""
UNCLASSIFIED = 'unclassified'
LABEL = 'Label'
GENOME_ID = 'Genome ID'
GENOME_NAME = 'Genome Name'
SPECIES = 'species'
HP = 'HP'
NHP = 'NHP'
RATIO = 'Ratio'
MAJORITY_LABEL = 'Majority Label'


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
