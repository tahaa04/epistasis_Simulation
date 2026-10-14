from itertools import combinations, product
import numpy as np
import pandas as pd

from src.config import STATES, REFERENCE_STATE


NON_REFERENCE_RBA_STATES = tuple(
    state for state in STATES if state != REFERENCE_STATE
)

# For RFA sum/effect coding we use the first q-1 states as independent states.
# The final state is reconstructed from the zero-mean constraint.
RFA_BASIS_STATES = STATES[:-1]
RFA_DEPENDENT_STATE = STATES[-1]


def rfa_contrast_value(actual_state, basis_state):
    """
    Sum-to-zero/effect coding for q=4 states.

    With STATES = (A, V, L, G), the three independent contrast columns are:
      A -> [ 1,  0,  0]
      V -> [ 0,  1,  0]
      L -> [ 0,  0,  1]
      G -> [-1, -1, -1]

    This guarantees that the expanded RFA effects across A,V,L,G sum to zero.
    """
    if actual_state == basis_state:
        return 1.0
    if actual_state == RFA_DEPENDENT_STATE:
        return -1.0
    return 0.0


def make_rfa_columns(max_order, n_sites=4):
    """
    Independent RFA regression columns up to max_order.

    Each column is represented as:
      (sites_tuple, basis_states_tuple)

    Example:
      ((0, 2), ("A", "V"))
    """
    columns = []

    for order in range(1, max_order + 1):
        for sites in combinations(range(n_sites), order):
            for basis_states in product(RFA_BASIS_STATES, repeat=order):
                columns.append((sites, basis_states))

    return columns


def build_rfa_design_matrix(df, max_order):
    """
    Build a constrained/sum-to-zero RFA design matrix.

    The intercept is the first column.
    """
    columns = make_rfa_columns(max_order)
    X = np.ones((len(df), 1 + len(columns)), dtype=float)

    for row_index, row in df.iterrows():
        genotype = tuple(row[f"site{i}"] for i in range(1, 5))

        for col_index, (sites, basis_states) in enumerate(columns, start=1):
            value = 1.0

            for site, basis_state in zip(sites, basis_states):
                value *= rfa_contrast_value(
                    genotype[site],
                    basis_state,
                )

            X[row_index, col_index] = value

    return X, columns


def expand_rfa_coefficients(beta, columns, max_order):
    """
    Convert the independent sum-coded regression coefficients into the same
    expanded RFA representation used by rfa_true_coefficients.csv.

    Thus the estimated and true RFA coefficients can be merged directly.
    """
    independent = {
        column: beta[i + 1]
        for i, column in enumerate(columns)
    }

    rows = [{
        "order": 0,
        "sites": "intercept",
        "states": "",
        "estimated_coefficient": beta[0],
    }]

    for order in range(1, max_order + 1):
        for sites in combinations(range(4), order):
            for actual_states in product(STATES, repeat=order):

                effect = 0.0

                for basis_states in product(RFA_BASIS_STATES, repeat=order):
                    multiplier = 1.0

                    for actual_state, basis_state in zip(
                        actual_states,
                        basis_states,
                    ):
                        multiplier *= rfa_contrast_value(
                            actual_state,
                            basis_state,
                        )

                    effect += (
                        independent[(sites, basis_states)]
                        * multiplier
                    )

                rows.append({
                    "order": order,
                    "sites": ",".join(str(site + 1) for site in sites),
                    "states": ",".join(actual_states),
                    "estimated_coefficient": effect,
                })

    return pd.DataFrame(rows)


def make_rba_columns(max_order, n_sites=4):
    """
    Treatment/reference-coded RBA columns.

    Reference state A receives no column. V, L and G are explicit mutation
    indicators, with their interaction products added through max_order.
    """
    columns = []

    for order in range(1, max_order + 1):
        for sites in combinations(range(n_sites), order):
            for states in product(NON_REFERENCE_RBA_STATES, repeat=order):
                columns.append((sites, states))

    return columns


def build_rba_design_matrix(df, max_order):
    """
    Build the ordinary treatment-coded RBA design matrix.
    """
    columns = make_rba_columns(max_order)
    X = np.ones((len(df), 1 + len(columns)), dtype=float)

    for row_index, row in df.iterrows():
        genotype = tuple(row[f"site{i}"] for i in range(1, 5))

        for col_index, (sites, states) in enumerate(columns, start=1):
            X[row_index, col_index] = float(
                all(
                    genotype[site] == state
                    for site, state in zip(sites, states)
                )
            )

    return X, columns


def rba_coefficients_to_dataframe(beta, columns):
    """
    Put fitted RBA coefficients in the same representation as
    rba_true_coefficients.csv.
    """
    rows = [{
        "order": 0,
        "sites": "intercept",
        "states": "",
        "estimated_coefficient": beta[0],
    }]

    for coefficient, (sites, states) in zip(beta[1:], columns):
        rows.append({
            "order": len(sites),
            "sites": ",".join(str(site + 1) for site in sites),
            "states": ",".join(states),
            "estimated_coefficient": coefficient,
        })

    return pd.DataFrame(rows)


def fit_ordinary_least_squares(X, y):
    """
    Fit ordinary least squares using NumPy.

    Returns the coefficient vector, matrix rank, and number of model parameters.
    """
    beta, residuals, rank, singular_values = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    return beta, rank, X.shape[1]


def fit_rfa(df, max_order):
    X, columns = build_rfa_design_matrix(df, max_order)
    y = df["phenotype"].to_numpy(dtype=float)

    beta, rank, n_parameters = fit_ordinary_least_squares(X, y)

    estimates = expand_rfa_coefficients(
        beta,
        columns,
        max_order,
    )

    return estimates, rank, n_parameters


def fit_rba(df, max_order):
    X, columns = build_rba_design_matrix(df, max_order)
    y = df["phenotype"].to_numpy(dtype=float)

    beta, rank, n_parameters = fit_ordinary_least_squares(X, y)

    estimates = rba_coefficients_to_dataframe(
        beta,
        columns,
    )

    return estimates, rank, n_parameters


def normalize_key_columns(df):
    """
    CSV reading turns the blank intercept state into NaN.
    Normalize it so true and estimated tables merge reliably.
    """
    df = df.copy()
    df["sites"] = df["sites"].fillna("").astype(str)
    df["states"] = df["states"].fillna("").astype(str)
    return df


def compare_with_truth(estimates, true_coefficients, max_order):
    """
    Merge estimated coefficients with the corresponding true coefficients.

    Only orders present in the fitted model (0..max_order) are compared.
    """
    truth = true_coefficients[
        true_coefficients["order"] <= max_order
    ].copy()

    truth = truth.rename(
        columns={"coefficient": "true_coefficient"}
    )

    estimates = normalize_key_columns(estimates)
    truth = normalize_key_columns(truth)

    comparison = truth.merge(
        estimates,
        on=["order", "sites", "states"],
        how="inner",
        validate="one_to_one",
    )

    comparison["error"] = (
        comparison["estimated_coefficient"]
        - comparison["true_coefficient"]
    )
    comparison["absolute_error"] = comparison["error"].abs()

    return comparison


def coefficient_metrics(comparison):
    """
    Simple coefficient-recovery metrics.

    Metrics are calculated both across all fitted coefficients and later
    separately by order in run_stage2.py.
    """
    true = comparison["true_coefficient"].to_numpy(dtype=float)
    estimated = comparison["estimated_coefficient"].to_numpy(dtype=float)

    error = estimated - true

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    if np.std(true) > 0 and np.std(estimated) > 0:
        pearson_r = np.corrcoef(true, estimated)[0, 1]
    else:
        pearson_r = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
    }
