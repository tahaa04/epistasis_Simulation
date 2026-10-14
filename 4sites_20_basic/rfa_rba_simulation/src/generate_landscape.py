from itertools import product, combinations
import numpy as np
import pandas as pd

from src.config import (
    STATES,
    BASE_SEQUENCE,
    VARIABLE_POSITIONS,
    RANDOM_SEED,
    ORDER_SCALES,
    INTERCEPT,
)


def zero_mean_array(array):
    """
    Convert an arbitrary k-dimensional array into a pure RFA k-th-order
    effect array by making the mean along every axis equal to zero.

    For a 2-D pairwise array, this makes every row and every column sum to zero.
    For higher orders, every 1-D slice along every axis sums to zero.
    """
    array = array.astype(float).copy()

    for axis in range(array.ndim):
        array -= array.mean(axis=axis, keepdims=True)

    return array


def generate_true_rfa_architecture():
    """
    Generate a reproducible full RFA architecture containing orders 1 to 4.

    Returns
    -------
    architecture : dict
        Keys are tuples of site indices, e.g. (0,), (0, 2), (0, 1, 3).
        Values are zero-mean numpy arrays of shape (4,)*order.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    q = len(STATES)
    n_sites = len(VARIABLE_POSITIONS)

    architecture = {}

    for order in range(1, n_sites + 1):
        for site_combo in combinations(range(n_sites), order):
            raw = rng.normal(
                loc=0.0,
                scale=ORDER_SCALES[order],
                size=(q,) * order,
            )
            architecture[site_combo] = zero_mean_array(raw)

    return architecture


def phenotype_from_architecture(genotype, architecture):
    """
    Compute phenotype for one 4-state genotype using the full RFA architecture.
    """
    state_to_index = {state: i for i, state in enumerate(STATES)}

    y = INTERCEPT

    for site_combo, effect_array in architecture.items():
        indices = tuple(state_to_index[genotype[site]] for site in site_combo)
        y += effect_array[indices]

    return float(y)


def make_full_sequence(variable_genotype):
    """
    Insert the four varying amino acids into the fixed 20-aa background sequence.
    VARIABLE_POSITIONS are 1-based for readability.
    """
    sequence = list(BASE_SEQUENCE)

    for position, amino_acid in zip(VARIABLE_POSITIONS, variable_genotype):
        sequence[position - 1] = amino_acid

    return "".join(sequence)


def generate_complete_landscape():
    """
    Generate all 4^4 = 256 genotypes and their exact synthetic phenotypes.
    """
    architecture = generate_true_rfa_architecture()
    rows = []

    for genotype in product(STATES, repeat=len(VARIABLE_POSITIONS)):
        phenotype = phenotype_from_architecture(genotype, architecture)

        rows.append({
            "genotype": "".join(genotype),
            "site1": genotype[0],
            "site2": genotype[1],
            "site3": genotype[2],
            "site4": genotype[3],
            "full_sequence": make_full_sequence(genotype),
            "phenotype": phenotype,
        })

    landscape = pd.DataFrame(rows)
    return landscape, architecture
