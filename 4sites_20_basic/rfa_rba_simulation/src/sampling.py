import numpy as np
import pandas as pd


def make_nested_missing_datasets(
    complete_df,
    missing_fractions=(0.15, 0.25),
    random_seed=123,
):
    """
    Randomly remove genotypes from the complete landscape.

    A single random permutation is used, so the genotypes missing from the
    15% dataset are also missing from the 25% dataset. This makes the two
    missingness levels easier to compare.

    Because 15% of 256 = 38.4, the number removed is rounded to the nearest
    whole genotype (38 genotypes, i.e. 14.84%).
    """
    rng = np.random.default_rng(random_seed)
    shuffled_indices = rng.permutation(complete_df.index.to_numpy())

    datasets = {}

    for fraction in sorted(missing_fractions):
        n_remove = int(round(len(complete_df) * fraction))
        removed_indices = shuffled_indices[:n_remove]

        observed = complete_df.drop(index=removed_indices).copy()
        removed = complete_df.loc[removed_indices].copy()

        observed = observed.sort_index().reset_index(drop=True)
        removed = removed.sort_index().reset_index(drop=True)

        datasets[fraction] = {
            "observed": observed,
            "removed": removed,
            "n_removed": n_remove,
            "actual_fraction": n_remove / len(complete_df),
        }

    return datasets
