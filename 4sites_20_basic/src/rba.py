from itertools import combinations, product
import pandas as pd

from src.config import STATES, REFERENCE_STATE


def phenotype_lookup(df):
    return dict(zip(df["genotype"], df["phenotype"]))


def make_reference_genotype(n_sites=4):
    return [REFERENCE_STATE] * n_sites


def calculate_true_rba_coefficients(df):
    """
    Calculate exact classical RBA coefficients from the complete landscape.

    Reference genotype: AAAA.

    Only non-reference states are stored because any RBA coefficient involving
    the reference state is zero by definition.

    Uses the standard inclusion-exclusion / finite-difference formula.
    """
    n_sites = 4
    non_reference_states = [s for s in STATES if s != REFERENCE_STATE]
    y = phenotype_lookup(df)

    reference = make_reference_genotype(n_sites)
    reference_key = "".join(reference)
    intercept = y[reference_key]

    effect_lookup = {}

    rows = [{
        "order": 0,
        "sites": "intercept",
        "states": "",
        "coefficient": intercept,
    }]

    for order in range(1, n_sites + 1):
        for sites in combinations(range(n_sites), order):
            for states in product(non_reference_states, repeat=order):

                # Start with phenotype of genotype carrying all requested mutations.
                genotype = reference.copy()
                for site, state in zip(sites, states):
                    genotype[site] = state

                effect = y["".join(genotype)]

                # Subtract intercept and every already-known lower-order RBA effect
                # contained in this mutant genotype.
                effect -= intercept

                for lower_order in range(1, order):
                    for local_subset in combinations(range(order), lower_order):
                        subset_sites = tuple(sites[i] for i in local_subset)
                        subset_states = tuple(states[i] for i in local_subset)
                        effect -= effect_lookup[(subset_sites, subset_states)]

                effect_lookup[(sites, states)] = effect

                rows.append({
                    "order": order,
                    "sites": ",".join(str(site + 1) for site in sites),
                    "states": ",".join(states),
                    "coefficient": effect,
                })

    coefficients_df = pd.DataFrame(rows)
    return coefficients_df, effect_lookup


def predict_from_rba_row(row, coefficients_df):
    """
    Reconstruct one phenotype from the full stored RBA coefficients.
    """
    genotype = tuple(row[f"site{i}"] for i in range(1, 5))

    intercept = coefficients_df.loc[
        coefficients_df["order"] == 0, "coefficient"
    ].iloc[0]

    prediction = intercept

    for _, coef_row in coefficients_df[coefficients_df["order"] > 0].iterrows():
        sites = tuple(int(x) - 1 for x in coef_row["sites"].split(","))
        states = tuple(coef_row["states"].split(","))

        if all(genotype[site] == state for site, state in zip(sites, states)):
            prediction += coef_row["coefficient"]

    return prediction
