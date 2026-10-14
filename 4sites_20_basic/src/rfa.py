from itertools import combinations, product
import numpy as np
import pandas as pd

from src.config import STATES


def conditional_mean(df, sites, states):
    """
    Mean phenotype among all complete-landscape genotypes having the requested
    states at the requested sites.

    sites use zero-based indices: 0, 1, 2, 3.
    """
    mask = np.ones(len(df), dtype=bool)

    for site, state in zip(sites, states):
        mask &= df[f"site{site + 1}"].eq(state).to_numpy()

    return df.loc[mask, "phenotype"].mean()


def calculate_true_rfa_coefficients(df):
    """
    Calculate the exact RFA decomposition directly from the complete phenotype table.

    Definition:
      e0 = global mean phenotype

    For a site/state combination S:
      e_S = mean(y | S fixed) - contributions of all lower-order subsets.

    Because the dataset is complete and balanced, these are the exact reference-free
    coefficients.

    Returns
    -------
    coefficients_df : pandas.DataFrame
        One row per expanded RFA coefficient.
    effect_lookup : dict
        Maps (site_tuple, state_tuple) -> effect value.
    """
    n_sites = 4
    q_states = list(STATES)

    e0 = df["phenotype"].mean()
    effect_lookup = {}

    rows = [{
        "order": 0,
        "sites": "intercept",
        "states": "",
        "coefficient": e0,
    }]

    for order in range(1, n_sites + 1):
        for sites in combinations(range(n_sites), order):
            for states in product(q_states, repeat=order):

                effect = conditional_mean(df, sites, states) - e0

                # Subtract every lower-order effect belonging to a proper,
                # non-empty subset of these sites.
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


def check_rfa_zero_mean(effect_lookup, tolerance=1e-10):
    """
    Verify the RFA zero-mean condition for every included effect array.

    For every site-combination, averaging over the states of any one participating
    site while holding the other states fixed should give zero.
    """
    problems = []

    grouped_site_sets = sorted({sites for sites, _ in effect_lookup})

    for sites in grouped_site_sets:
        order = len(sites)

        for varied_axis in range(order):
            other_axes = [i for i in range(order) if i != varied_axis]

            for fixed_states in product(STATES, repeat=len(other_axes)):
                total = 0.0

                for varied_state in STATES:
                    state_list = [None] * order
                    fixed_iter = iter(fixed_states)

                    for axis in range(order):
                        if axis == varied_axis:
                            state_list[axis] = varied_state
                        else:
                            state_list[axis] = next(fixed_iter)

                    total += effect_lookup[(sites, tuple(state_list))]

                if abs(total) > tolerance:
                    problems.append((sites, varied_axis, fixed_states, total))

    return problems


def predict_from_rfa_row(row, coefficients_df):
    """
    Reconstruct one phenotype from the full stored RFA coefficients.
    Used only as a validation check.
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
