from pathlib import Path
import numpy as np

from src.generate_landscape import generate_complete_landscape
from src.rfa import (
    calculate_true_rfa_coefficients,
    check_rfa_zero_mean,
    predict_from_rfa_row,
)
from src.rba import (
    calculate_true_rba_coefficients,
    predict_from_rba_row,
)


ROOT = Path(__file__).resolve().parent
FULL_DATA_DIR = ROOT / "data" / "full"
TRUE_COEF_DIR = ROOT / "data" / "true_coefficients"

FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRUE_COEF_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ------------------------------------------------------------
    # 1. Generate complete 4^4 = 256 genotype landscape.
    # ------------------------------------------------------------
    landscape, generated_rfa_architecture = generate_complete_landscape()

    landscape_path = FULL_DATA_DIR / "complete_landscape.csv"
    landscape.to_csv(landscape_path, index=False)

    print(f"Generated {len(landscape)} genotypes.")
    print(f"Saved complete landscape to: {landscape_path}")

    # ------------------------------------------------------------
    # 2. Calculate exact RFA coefficients FROM THE COMPLETE DATA.
    # ------------------------------------------------------------
    rfa_df, rfa_lookup = calculate_true_rfa_coefficients(landscape)

    rfa_path = TRUE_COEF_DIR / "rfa_true_coefficients.csv"
    rfa_df.to_csv(rfa_path, index=False)

    zero_mean_problems = check_rfa_zero_mean(rfa_lookup)

    if zero_mean_problems:
        raise RuntimeError(
            f"RFA zero-mean check failed for {len(zero_mean_problems)} slices."
        )

    print("RFA zero-mean checks passed.")
    print(f"Saved true RFA coefficients to: {rfa_path}")

    # ------------------------------------------------------------
    # 3. Calculate exact classical RBA coefficients using AAAA
    #    as the reference genotype.
    # ------------------------------------------------------------
    rba_df, rba_lookup = calculate_true_rba_coefficients(landscape)

    rba_path = TRUE_COEF_DIR / "rba_true_coefficients.csv"
    rba_df.to_csv(rba_path, index=False)

    print(f"Saved true RBA coefficients to: {rba_path}")

    # ------------------------------------------------------------
    # 4. Validation: reconstruct all 256 phenotype values from
    #    each full decomposition.
    # ------------------------------------------------------------
    rfa_predictions = landscape.apply(
        lambda row: predict_from_rfa_row(row, rfa_df), axis=1
    ).to_numpy()

    rba_predictions = landscape.apply(
        lambda row: predict_from_rba_row(row, rba_df), axis=1
    ).to_numpy()

    observed = landscape["phenotype"].to_numpy()

    rfa_max_error = np.max(np.abs(observed - rfa_predictions))
    rba_max_error = np.max(np.abs(observed - rba_predictions))

    print()
    print("Validation")
    print("----------")
    print(f"RFA maximum reconstruction error: {rfa_max_error:.3e}")
    print(f"RBA maximum reconstruction error: {rba_max_error:.3e}")

    if rfa_max_error > 1e-9:
        raise RuntimeError("Full RFA decomposition does not reconstruct the data.")

    if rba_max_error > 1e-9:
        raise RuntimeError("Full RBA decomposition does not reconstruct the data.")

    print("Both full decompositions reconstruct the complete landscape exactly.")

    print()
    print("Coefficient counts")
    print("------------------")
    print(rfa_df.groupby("order").size().rename("RFA expanded terms"))
    print()
    print(rba_df.groupby("order").size().rename("RBA independent terms"))


if __name__ == "__main__":
    main()
