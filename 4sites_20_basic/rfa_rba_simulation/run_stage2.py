from pathlib import Path
import pandas as pd

from src.sampling import make_nested_missing_datasets
from src.regression import (
    fit_rfa,
    fit_rba,
    compare_with_truth,
    coefficient_metrics,
)


ROOT = Path(__file__).resolve().parent

FULL_DATA_PATH = ROOT / "data" / "full" / "complete_landscape.csv"
RFA_TRUE_PATH = (
    ROOT / "data" / "true_coefficients" / "rfa_true_coefficients.csv"
)
RBA_TRUE_PATH = (
    ROOT / "data" / "true_coefficients" / "rba_true_coefficients.csv"
)

MISSING_DIR = ROOT / "data" / "missing"
COEFFICIENT_DIR = ROOT / "results" / "coefficient_comparisons"
METRICS_DIR = ROOT / "results" / "metrics"

MISSING_DIR.mkdir(parents=True, exist_ok=True)
COEFFICIENT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def save_missing_datasets(complete):
    datasets = make_nested_missing_datasets(
        complete,
        missing_fractions=(0.15, 0.25),
        random_seed=6,
    )

    for fraction, info in datasets.items():
        label = f"{int(round(fraction * 100))}pct"

        observed_path = MISSING_DIR / f"observed_{label}.csv"
        removed_path = MISSING_DIR / f"removed_{label}.csv"

        info["observed"].to_csv(observed_path, index=False)
        info["removed"].to_csv(removed_path, index=False)

        print(
            f"{label}: removed {info['n_removed']} / {len(complete)} "
            f"genotypes ({100 * info['actual_fraction']:.2f}%). "
            f"{len(info['observed'])} remain."
        )

    return datasets


def metrics_by_order(comparison):
    rows = []

    # Overall, including intercept.
    overall = coefficient_metrics(comparison)
    rows.append({
        "coefficient_order": "all",
        **overall,
        "n_coefficients": len(comparison),
    })

    # Per genetic order. Intercept is reported as order 0.
    for order in sorted(comparison["order"].unique()):
        subset = comparison[comparison["order"] == order]
        metrics = coefficient_metrics(subset)

        rows.append({
            "coefficient_order": str(order),
            **metrics,
            "n_coefficients": len(subset),
        })

    return rows


def main():
    complete = pd.read_csv(FULL_DATA_PATH)
    rfa_true = pd.read_csv(RFA_TRUE_PATH)
    rba_true = pd.read_csv(RBA_TRUE_PATH)

    datasets = save_missing_datasets(complete)

    metric_rows = []

    for fraction, info in datasets.items():
        observed = info["observed"]
        missing_label = f"{int(round(fraction * 100))}pct"

        # Orders 1, 2 and 3 are identifiable with both current datasets:
        # order 1 = 13 independent parameters
        # order 2 = 67 independent parameters
        # order 3 = 175 independent parameters
        #
        # A complete fourth-order model has 256 independent parameters,
        # but there are fewer than 256 observations after removing data.
        # Therefore unregularized fourth-order OLS would not have a unique
        # coefficient solution and is intentionally not fitted here.
        for max_order in (1, 2, 3):

            for method in ("RFA", "RBA"):
                if method == "RFA":
                    estimates, rank, n_parameters = fit_rfa(
                        observed,
                        max_order,
                    )
                    comparison = compare_with_truth(
                        estimates,
                        rfa_true,
                        max_order,
                    )
                else:
                    estimates, rank, n_parameters = fit_rba(
                        observed,
                        max_order,
                    )
                    comparison = compare_with_truth(
                        estimates,
                        rba_true,
                        max_order,
                    )

                comparison.insert(0, "method", method)
                comparison.insert(1, "missing_fraction", fraction)
                comparison.insert(2, "fit_max_order", max_order)

                out_path = (
                    COEFFICIENT_DIR
                    / f"{method.lower()}_{missing_label}_order{max_order}.csv"
                )
                comparison.to_csv(out_path, index=False)

                if rank != n_parameters:
                    raise RuntimeError(
                        f"{method} order {max_order}, {missing_label}: "
                        f"design matrix is rank deficient "
                        f"(rank {rank}, parameters {n_parameters})."
                    )

                for row in metrics_by_order(comparison):
                    metric_rows.append({
                        "method": method,
                        "missing_fraction": fraction,
                        "fit_max_order": max_order,
                        "n_observations": len(observed),
                        "n_independent_parameters": n_parameters,
                        "design_matrix_rank": rank,
                        **row,
                    })

                print(
                    f"Fitted {method} order {max_order} to {missing_label}: "
                    f"{len(observed)} observations, "
                    f"{n_parameters} independent parameters."
                )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = METRICS_DIR / "coefficient_recovery_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print()
    print(f"Saved coefficient comparisons to: {COEFFICIENT_DIR}")
    print(f"Saved summary metrics to: {metrics_path}")
    print()
    print(
        "Note: full fourth-order OLS is not fitted because after removing "
        "data there are fewer observations than the 256 independent "
        "parameters. A regularized/sparse full-model experiment should be "
        "handled separately."
    )


if __name__ == "__main__":
    main()
