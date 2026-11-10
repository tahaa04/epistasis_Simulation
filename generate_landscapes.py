"""Generate complete simulated genotype-phenotype landscapes and their exact
RFA and RBA coefficients.

For each architecture in config.ARCHITECTURES and each realization:
  1. draw generating terms in the architecture's framework (RFA or RBA);
  2. sum them into phenotypes for all 256 genotypes;
  3. standardize phenotypes to mean 0, variance 1 (population variance);
  4. compute exact (non-regression) RFA and RBA coefficients from the
     standardized landscape (RBA reference = AAAA);
  5. verify reconstruction, constraints, and the intended architecture;
  6. save everything under data/generated_landscapes/<architecture>/realization_XX/.

Run:  python generate_landscapes.py
"""

import json
from math import comb

import numpy as np
import pandas as pd

import config
import epistasis as ep

Q = config.N_STATES
N = config.N_SITES


# ---------------------------------------------------------------------------
# Drawing generating terms
# ---------------------------------------------------------------------------

def empty_terms():
    """All-zero terms for every site combination (intercept included)."""
    return {sites: np.zeros((Q,) * len(sites)) for sites in ep.site_subsets(N)}


def project_zero_sum(array):
    """Orthogonal projection onto arrays whose every 1-D section averages to 0."""
    for axis in range(array.ndim):
        array = array - array.mean(axis=axis, keepdims=True)
    return array


def mutant_mask(sites):
    """Boolean mask of RBA entries where every site carries a non-reference state."""
    return ep.rba_mutant_mask(config.REFERENCE)(sites, np.zeros((Q,) * len(sites)))


def rescale_order(terms, order, target_rms):
    """Scale all terms of one order so the RMS of their nonzero entries equals target_rms."""
    subsets = ep.site_subsets(N, order)
    values = np.concatenate([terms[s][terms[s] != 0] for s in subsets])
    factor = target_rms / np.sqrt(np.mean(values ** 2))
    for s in subsets:
        terms[s] = terms[s] * factor


def draw_rfa_dense(rng, order_scales):
    terms = empty_terms()
    for k, scale in order_scales.items():
        for sites in ep.site_subsets(N, k):
            terms[sites] = project_zero_sum(rng.normal(size=(Q,) * k))
        rescale_order(terms, k, scale)
    return terms


def draw_rba_dense(rng, order_scales):
    terms = empty_terms()
    for k, scale in order_scales.items():
        for sites in ep.site_subsets(N, k):
            terms[sites] = rng.normal(size=(Q,) * k) * mutant_mask(sites)
        rescale_order(terms, k, scale)
    return terms


def draw_rba_sparse(rng, order_scales, nonzero_fraction):
    """Exactly max(1, round(f * n_k)) nonzero mutant terms at each order k."""
    terms = empty_terms()
    for k, scale in order_scales.items():
        positions = [(sites, states)
                     for sites in ep.site_subsets(N, k)
                     for states in zip(*np.nonzero(mutant_mask(sites)))]
        n_nonzero = max(1, int(round(nonzero_fraction * len(positions))))
        for idx in rng.choice(len(positions), size=n_nonzero, replace=False):
            sites, states = positions[idx]
            terms[sites][states] = rng.normal()
        rescale_order(terms, k, scale)
    return terms


def draw_rfa_sparse(rng, order_scales, nonzero_fraction):
    """Sum of a few elementary zero-sum contrasts at each order (see config.py)."""
    terms = empty_terms()
    for k, scale in order_scales.items():
        subsets = ep.site_subsets(N, k)
        n_terms = comb(N, k) * Q ** k
        n_contrasts = max(1, int(round(nonzero_fraction * n_terms / 2 ** k)))
        for _ in range(n_contrasts):
            sites = subsets[rng.integers(len(subsets))]
            contrast = np.array(rng.normal())
            for _site in sites:
                a, b = rng.choice(Q, size=2, replace=False)
                vector = np.zeros(Q)
                vector[a], vector[b] = 1.0, -1.0
                contrast = np.multiply.outer(contrast, vector)
            terms[sites] = terms[sites] + contrast
        rescale_order(terms, k, scale)
    return terms


def draw_rfa_variance_controlled(rng, variance_targets):
    """Dense zero-sum terms, each order scaled to an exact variance contribution."""
    terms = draw_rfa_dense(rng, {k: 1.0 for k in variance_targets})
    current = ep.rfa_variance_by_order(terms, Q)
    for k, target in variance_targets.items():
        factor = np.sqrt(target / current[k])
        for sites in ep.site_subsets(N, k):
            terms[sites] = terms[sites] * factor
    return terms


def draw_generating_terms(rng, arch):
    framework, kind = arch["framework"], arch["kind"]
    if kind == "variance":
        return draw_rfa_variance_controlled(rng, arch["variance_targets"])
    if kind == "dense":
        draw = draw_rfa_dense if framework == "RFA" else draw_rba_dense
        return draw(rng, arch["order_scales"])
    if kind == "sparse":
        draw = draw_rfa_sparse if framework == "RFA" else draw_rba_sparse
        return draw(rng, arch["order_scales"], arch["nonzero_fraction"])
    raise ValueError(f"Unknown architecture kind: {kind}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def framework_values_by_order(terms, framework):
    """Per-order arrays of the free coefficients of a framework.

    RFA: all q**k entries per site combination. RBA: only all-mutant entries.
    """
    mask = ep.rba_mutant_mask(config.REFERENCE) if framework == "RBA" else None
    values = ep.terms_by_order(terms, mask)
    values.pop(0, None)                                   # drop the intercept
    return values


def verify_realization(arch, Y, generating, raw_mean, raw_sd, rfa, rba):
    """Run all checks. Returns {check_name: {"value": float, "passed": bool}}."""
    tol = config.TOLERANCE
    checks = {}

    def record(name, value, passed=None):
        value = float(value)
        checks[name] = {"value": value, "passed": bool(value < tol if passed is None else passed)}

    # 1. Standardization
    record("standardized_mean_abs", abs(Y.mean()))
    record("standardized_variance_minus_1_abs", abs(Y.var() - 1.0))

    # 2. Exact coefficients reconstruct the phenotypes
    record("rfa_reconstruction_max_abs_error",
           np.max(np.abs(ep.phenotypes_from_terms(rfa, N, Q) - Y)))
    record("rba_reconstruction_max_abs_error",
           np.max(np.abs(ep.phenotypes_from_terms(rba, N, Q) - Y)))

    # 3. Framework constraints
    record("rfa_zero_sum_max_violation", ep.rfa_zero_sum_violation(rfa))
    record("rfa_intercept_equals_mean_abs", abs(float(rfa[()]) - Y.mean()))
    record("rba_reference_terms_max_abs", ep.rba_reference_violation(rba, config.REFERENCE))
    record("rba_intercept_equals_reference_phenotype_abs",
           abs(float(rba[()]) - Y[config.REFERENCE]))
    variance = ep.rfa_variance_by_order(rfa, Q)
    record("rfa_variance_partition_sum_error", abs(sum(variance.values()) - Y.var()))

    # 4. Intended architecture: the exact coefficients in the generating framework
    #    equal the generating terms after the same standardization.
    computed = rfa if arch["framework"] == "RFA" else rba
    expected = {s: t / raw_sd for s, t in generating.items()}
    expected[()] = (generating[()] - raw_mean) / raw_sd
    record("generating_terms_recovered_max_abs_error",
           max(np.max(np.abs(computed[s] - expected[s])) for s in computed))

    # 5. Architecture-specific properties, measured on the computed coefficients
    values = framework_values_by_order(computed, arch["framework"])
    gen_values = framework_values_by_order(generating, arch["framework"])
    nonzero = {k: int(np.sum(np.abs(v) > tol)) for k, v in values.items()}
    gen_nonzero = {k: int(np.sum(v != 0)) for k, v in gen_values.items()}
    record("nonzero_pattern_matches_generating",
           0.0 if nonzero == gen_nonzero else 1.0, passed=nonzero == gen_nonzero)

    if arch["kind"] in ("dense", "sparse"):
        scales = arch["order_scales"]
        absent = [k for k in values if k not in scales]
        record("absent_orders_max_abs",
               max((np.max(np.abs(values[k])) for k in absent), default=0.0))
        if arch["kind"] == "dense":
            full = all(nonzero[k] == values[k].size for k in scales)
            record("present_orders_fully_dense", 0.0 if full else 1.0, passed=full)
        # RMS of nonzero terms per order relative to the first order present
        rms = {k: np.sqrt(np.mean(values[k][np.abs(values[k]) > tol] ** 2)) for k in scales}
        k0 = min(scales)
        record("order_rms_ratio_max_error",
               max(abs(rms[k] / rms[k0] - scales[k] / scales[k0]) for k in scales))
    if arch["kind"] == "variance":
        fractions = {k: v / Y.var() for k, v in variance.items()}
        record("variance_fraction_max_error",
               max(abs(fractions[k] - t) for k, t in arch["variance_targets"].items()))
    return checks


# ---------------------------------------------------------------------------
# One realization
# ---------------------------------------------------------------------------

def generate_realization(arch, arch_number, realization):
    seed = [config.BASE_SEED, arch_number, realization]
    rng = np.random.default_rng(seed)

    generating = draw_generating_terms(rng, arch)
    Y_raw = ep.phenotypes_from_terms(generating, N, Q)
    raw_mean, raw_sd = float(Y_raw.mean()), float(Y_raw.std())     # population SD
    Y = (Y_raw - raw_mean) / raw_sd

    rfa = ep.rfa_coefficients(Y)
    rba = ep.rba_coefficients(Y, config.REFERENCE)
    checks = verify_realization(arch, Y, generating, raw_mean, raw_sd, rfa, rba)

    failed = [name for name, c in checks.items() if not c["passed"]]
    if failed:
        raise RuntimeError(f"{arch['name']} realization {realization} failed: {failed}")

    variance = ep.rfa_variance_by_order(rfa, Q)
    rfa_values = framework_values_by_order(rfa, "RFA")
    rba_values = framework_values_by_order(rba, "RBA")
    tol = config.TOLERANCE
    summary = {
        "rfa_variance_fraction_by_order": {k: variance[k] / Y.var() for k in sorted(variance)},
        "rfa_nonzero_terms_by_order": {k: int(np.sum(np.abs(v) > tol)) for k, v in rfa_values.items()},
        "rfa_total_terms_by_order": {k: int(v.size) for k, v in rfa_values.items()},
        "rfa_rms_by_order": {k: float(np.sqrt(np.mean(v ** 2))) for k, v in rfa_values.items()},
        "rba_nonzero_terms_by_order": {k: int(np.sum(np.abs(v) > tol)) for k, v in rba_values.items()},
        "rba_total_terms_by_order": {k: int(v.size) for k, v in rba_values.items()},
        "rba_rms_by_order": {k: float(np.sqrt(np.mean(v ** 2))) for k, v in rba_values.items()},
    }
    return dict(seed=seed, Y=Y, Y_raw=Y_raw, raw_mean=raw_mean, raw_sd=raw_sd,
                rfa=rfa, rba=rba, checks=checks, summary=summary)


def save_realization(arch, realization, result):
    out_dir = config.OUTPUT_DIR / arch["name"] / f"realization_{realization:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = ep.landscape_to_table(result["Y"], config.STATES, config.REFERENCE)
    table["phenotype_unstandardized"] = result["Y_raw"].ravel()
    table.to_csv(out_dir / "genotype_phenotype.csv", index=False)

    ep.terms_to_table(result["rfa"], config.STATES, "RFA").to_csv(
        out_dir / "rfa_coefficients.csv", index=False)
    ep.terms_to_table(result["rba"], config.STATES, "RBA", config.REFERENCE).to_csv(
        out_dir / "rba_coefficients.csv", index=False)

    def str_keys(d):
        return {str(k): v for k, v in d.items()}

    arch_params = {k: (str_keys(v) if isinstance(v, dict) else v) for k, v in arch.items()}
    parameters = {
        "architecture": arch_params,
        "realization": realization,
        "seed": {"numpy_default_rng_seed": result["seed"],
                 "meaning": "[BASE_SEED, architecture_number, realization]"},
        "genotype_space": {"states": list(config.STATES), "n_sites": N,
                           "n_genotypes": Q ** N,
                           "rba_reference": config.REFERENCE_GENOTYPE,
                           "genotype_order": "lexicographic in state order A,V,L,G; site 1 slowest"},
        "standardization": {"raw_mean": result["raw_mean"],
                            "raw_sd_population": result["raw_sd"],
                            "formula": "phenotype = (phenotype_unstandardized - raw_mean) / raw_sd_population"},
        "realized": {name: str_keys(d) for name, d in result["summary"].items()},
        "verification": {"tolerance": config.TOLERANCE, "checks": result["checks"],
                         "all_passed": all(c["passed"] for c in result["checks"].values())},
        "files": {"genotype_phenotype.csv": "all 256 genotypes, standardized phenotype",
                  "rfa_coefficients.csv": "exact RFA terms, all 625 (zero-sum constrained)",
                  "rba_coefficients.csv": "exact RBA terms relative to AAAA, 256 mutant terms "
                                          "(terms involving a reference state are 0 by definition)"},
    }
    with open(out_dir / "parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ep.check_two_site_example()
    print("Hand-worked 2-site RFA/RBA example: OK")

    rows = []
    for arch_number, arch in enumerate(config.ARCHITECTURES, start=1):
        for realization in range(config.N_REALIZATIONS):
            result = generate_realization(arch, arch_number, realization)
            save_realization(arch, realization, result)

            s, c = result["summary"], result["checks"]
            row = {"architecture": arch["name"], "group": arch["group"],
                   "framework": arch["framework"], "kind": arch["kind"],
                   "realization": realization, "seed": str(result["seed"])}
            for k in range(1, N + 1):
                row[f"rfa_var_frac_order{k}"] = s["rfa_variance_fraction_by_order"].get(k, 0.0)
            for k in range(1, N + 1):
                row[f"rfa_nonzero_order{k}"] = s["rfa_nonzero_terms_by_order"].get(k, 0)
            for k in range(1, N + 1):
                row[f"rba_nonzero_order{k}"] = s["rba_nonzero_terms_by_order"].get(k, 0)
            row["max_reconstruction_error"] = max(
                c["rfa_reconstruction_max_abs_error"]["value"],
                c["rba_reconstruction_max_abs_error"]["value"])
            row["rfa_zero_sum_max_violation"] = c["rfa_zero_sum_max_violation"]["value"]
            row["generating_terms_recovered_max_abs_error"] = \
                c["generating_terms_recovered_max_abs_error"]["value"]
            row["all_checks_passed"] = all(v["passed"] for v in c.values())
            rows.append(row)
        print(f"  {arch['name']:<30s} {config.N_REALIZATIONS} realizations: all checks passed")

    summary = pd.DataFrame(rows)
    summary.to_csv(config.OUTPUT_DIR / "summary.csv", index=False)

    # Architecture-level overview (mean over realizations)
    pd.set_option("display.width", 200)
    cols = [f"rfa_var_frac_order{k}" for k in range(1, N + 1)]
    overview = summary.groupby("architecture")[cols].mean().round(3)
    overview.columns = [f"RFA var. ord{k}" for k in range(1, N + 1)]
    nz = summary.groupby("architecture")[
        [f"rfa_nonzero_order{k}" for k in range(1, N + 1)]
        + [f"rba_nonzero_order{k}" for k in range(1, N + 1)]].mean().round(1)
    nz.columns = [f"RFA nz{k}" for k in range(1, N + 1)] + [f"RBA nz{k}" for k in range(1, N + 1)]
    print("\nMean RFA variance fraction per order, and mean number of nonzero terms per order")
    print("(RFA totals per order: 16/96/256/256; RBA totals: 12/54/108/81)\n")
    print(pd.concat([overview, nz], axis=1).to_string())
    print(f"\nWorst reconstruction error: {summary['max_reconstruction_error'].max():.2e}; "
          f"worst zero-sum violation: {summary['rfa_zero_sum_max_violation'].max():.2e}")
    print(f"Saved {len(summary)} realizations to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
