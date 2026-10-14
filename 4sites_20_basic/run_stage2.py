from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
PLOTS_DIR = ROOT / "results" / "plots"

MISSING_DIR.mkdir(parents=True, exist_ok=True)
COEFFICIENT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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


def coefficient_similarity_metrics(comparison):
    """
    Calculate the main coefficient-recovery metrics.

    The intercept is excluded from the main similarity score because RFA and RBA
    intercepts have different interpretations.

    Returns:
      coefficient_r2 : numerical agreement with ground truth (1 is perfect)
      pearson_r      : agreement in coefficient pattern/ranking (1 is perfect)
      rmse           : root mean squared error
      mae            : mean absolute error
      nrmse          : RMSE normalized by SD of true coefficients (0 is perfect)
    """
    df = comparison[comparison["order"] > 0].copy()

    true = df["true_coefficient"].to_numpy(dtype=float)
    estimated = df["estimated_coefficient"].to_numpy(dtype=float)

    error = estimated - true

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    true_sd = np.std(true)
    nrmse = rmse / true_sd if true_sd > 0 else np.nan

    if np.std(true) > 0 and np.std(estimated) > 0:
        pearson_r = np.corrcoef(true, estimated)[0, 1]
    else:
        pearson_r = np.nan

    ss_res = np.sum((true - estimated) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    coefficient_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "coefficient_r2": coefficient_r2,
        "pearson_r": pearson_r,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "n_coefficients": len(df),
    }


def metrics_by_order(comparison):
    """
    Calculate recovery metrics overall (excluding intercept) and separately
    for each coefficient order represented in the fitted model.
    """
    rows = []

    overall = coefficient_similarity_metrics(comparison)
    rows.append({
        "coefficient_scope": "all_non_intercept",
        **overall,
    })

    for order in sorted(
        comparison.loc[comparison["order"] > 0, "order"].unique()
    ):
        subset = comparison[comparison["order"] == order].copy()

        true = subset["true_coefficient"].to_numpy(dtype=float)
        estimated = subset["estimated_coefficient"].to_numpy(dtype=float)
        error = estimated - true

        rmse = np.sqrt(np.mean(error ** 2))
        mae = np.mean(np.abs(error))

        true_sd = np.std(true)
        nrmse = rmse / true_sd if true_sd > 0 else np.nan

        if np.std(true) > 0 and np.std(estimated) > 0:
            pearson_r = np.corrcoef(true, estimated)[0, 1]
        else:
            pearson_r = np.nan

        ss_res = np.sum((true - estimated) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        coefficient_r2 = (
            1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        )

        rows.append({
            "coefficient_scope": f"order_{int(order)}",
            "coefficient_r2": coefficient_r2,
            "pearson_r": pearson_r,
            "rmse": rmse,
            "mae": mae,
            "nrmse": nrmse,
            "n_coefficients": len(subset),
        })

    return rows


def plot_overall_metric(metrics, metric_name, ylabel, title, filename):
    """
    Plot a similarity metric across fitted model orders.

    Styling requested:
      RFA = green
      RBA = blue
      15% missing = solid line
      25% missing = dotted line
    """
    overall = metrics[
        metrics["coefficient_scope"] == "all_non_intercept"
    ].copy()

    colors = {
        "RFA": "green",
        "RBA": "blue",
    }

    line_styles = {
        0.15: "-",
        0.25: ":",
    }

    plt.figure(figsize=(8, 5))

    for method in ("RFA", "RBA"):
        for missing_fraction in (0.15, 0.25):
            subset = overall[
                (overall["method"] == method)
                & (np.isclose(
                    overall["missing_fraction"],
                    missing_fraction,
                ))
            ].sort_values("fit_max_order")

            label = (
                f"{method} - "
                f"{int(round(missing_fraction * 100))}% missing"
            )

            plt.plot(
                subset["fit_max_order"],
                subset[metric_name],
                marker="o",
                color=colors[method],
                linestyle=line_styles[missing_fraction],
                linewidth=2,
                label=label,
            )

    plt.xlabel("Fitted maximum model order")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([1, 2, 3])
    plt.legend()
    plt.tight_layout()

    output_path = PLOTS_DIR / filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def make_plots(metrics):
    """
    Create the three main overall coefficient-recovery plots.
    """
    plot_paths = []

    plot_paths.append(
        plot_overall_metric(
            metrics=metrics,
            metric_name="coefficient_r2",
            ylabel="Coefficient R² vs ground truth",
            title="Coefficient recovery accuracy",
            filename="overall_coefficient_r2.png",
        )
    )

    plot_paths.append(
        plot_overall_metric(
            metrics=metrics,
            metric_name="pearson_r",
            ylabel="Pearson r vs ground truth",
            title="Coefficient pattern recovery",
            filename="overall_pearson_r.png",
        )
    )

    plot_paths.append(
        plot_overall_metric(
            metrics=metrics,
            metric_name="nrmse",
            ylabel="Normalized RMSE",
            title="Coefficient error relative to ground truth",
            filename="overall_nrmse.png",
        )
    )

    return plot_paths


def main():
    complete = pd.read_csv(FULL_DATA_PATH)
    rfa_true = pd.read_csv(RFA_TRUE_PATH)
    rba_true = pd.read_csv(RBA_TRUE_PATH)

    datasets = save_missing_datasets(complete)

    metric_rows = []

    for fraction, info in datasets.items():
        observed = info["observed"]
        missing_label = f"{int(round(fraction * 100))}pct"

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

                output_path = (
                    COEFFICIENT_DIR
                    / f"{method.lower()}_{missing_label}_order{max_order}.csv"
                )
                comparison.to_csv(output_path, index=False)

                if rank != n_parameters:
                    raise RuntimeError(
                        f"{method} order {max_order}, {missing_label}: "
                        f"design matrix is rank deficient "
                        f"(rank {rank}, parameters {n_parameters})."
                    )

                for metric_row in metrics_by_order(comparison):
                    metric_rows.append({
                        "method": method,
                        "missing_fraction": fraction,
                        "fit_max_order": max_order,
                        "n_observations": len(observed),
                        "n_independent_parameters": n_parameters,
                        "design_matrix_rank": rank,
                        **metric_row,
                    })

                print(
                    f"Fitted {method} order {max_order} to {missing_label}: "
                    f"{len(observed)} observations, "
                    f"{n_parameters} independent parameters."
                )

    metrics = pd.DataFrame(metric_rows)

    metrics_path = (
        METRICS_DIR / "coefficient_similarity_summary.csv"
    )
    metrics.to_csv(metrics_path, index=False)

    plot_paths = make_plots(metrics)

    print()
    print(f"Saved coefficient comparisons to: {COEFFICIENT_DIR}")
    print(f"Saved similarity metrics to: {metrics_path}")
    print("Saved plots:")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
