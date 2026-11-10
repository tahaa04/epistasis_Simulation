"""Shared mathematical functions for reference-free (RFA) and reference-based (RBA)
analysis of complete combinatorial genotype-phenotype landscapes.

Primary reference: Park, Metzger & Thornton (2024) Nat. Commun. 15:7953, and its
Supplementary Information (sections 2.3-2.7).

Representation
--------------
A complete landscape over n sites with q states per site is stored as a NumPy
array ``Y`` of shape (q,)*n; ``Y[g1, ..., gn]`` is the phenotype of genotype
(g1, ..., gn), where each g_i is a state index (0..q-1).

A decomposition (RFA or RBA) is stored as a dict ``terms`` that maps a tuple of
site indices (sorted, 0-based) to an array of shape (q,)*k, where k is the
number of sites in the tuple (the order of the term):

    terms[()]      -> 0-d array, the intercept (e0 or lambda0)
    terms[(i,)]    -> shape (q,),   first-order terms at site i
    terms[(i, j)]  -> shape (q, q), pairwise terms at sites i < j
    ...

Both frameworks are special cases of the "generalized linear decomposition"
(SI Eq. 2): the phenotype of a genotype is the sum of the terms for every
state-combination it contains,

    y(g) = e0 + sum_i e_i(g_i) + sum_{i<j} e_ij(g_i, g_j) + ...

They differ only in the constraints placed on the terms:
  * RFA: zero-mean property -- every 1-D section of every term array averages
    to zero (SI section 2.4).
  * RBA: every term involving the reference state at any of its sites is zero
    (SI section 2.5). The q-1 remaining entries per site are the effects of
    mutations away from the reference.
"""

from itertools import combinations, product

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Genotype space helpers
# ---------------------------------------------------------------------------

def all_genotypes(n_sites, n_states):
    """Return all genotypes as an int array of shape (q**n, n).

    Rows follow NumPy C-order of the landscape array, i.e. ``Y.ravel()`` lines
    up row by row with this array (the last site varies fastest).
    """
    return np.array(list(product(range(n_states), repeat=n_sites)), dtype=int)


def site_subsets(n_sites, order=None):
    """All sorted site tuples of a given order, or of every order 0..n if None."""
    if order is not None:
        return list(combinations(range(n_sites), order))
    subsets = []
    for k in range(n_sites + 1):
        subsets.extend(combinations(range(n_sites), k))
    return subsets


def expand_term(term, sub_sites, sup_sites):
    """Reshape a term defined on ``sub_sites`` so it broadcasts over ``sup_sites``.

    ``sub_sites`` must be a subset of ``sup_sites`` (both sorted). The returned
    array has one axis per site in ``sup_sites``: length q for sites in
    ``sub_sites`` and length 1 (broadcast) for the others.
    """
    term = np.asarray(term)
    q = term.shape[0] if term.ndim > 0 else 1
    shape = [q if site in sub_sites else 1 for site in sup_sites]
    return term.reshape(shape)


# ---------------------------------------------------------------------------
# Phenotypes from terms (works for RFA and RBA alike)
# ---------------------------------------------------------------------------

def phenotypes_from_terms(terms, n_sites, n_states):
    """Sum all terms into a complete landscape array of shape (q,)*n."""
    all_sites = tuple(range(n_sites))
    Y = np.zeros((n_states,) * n_sites)
    for sites, term in terms.items():
        Y = Y + expand_term(term, sites, all_sites)
    return Y


# ---------------------------------------------------------------------------
# Exact coefficients from a complete landscape (no regression)
# ---------------------------------------------------------------------------

def rfa_coefficients(Y):
    """Exact reference-free effects of a complete landscape (SI Eq. 1).

    For each site combination S (processed in order of increasing size),

        e_S(s) = <y | genotypes sharing states s at sites S>
                 - sum over proper subsets T of S of e_T(s restricted to T).

    The first quantity is the mean phenotype of the subspace, obtained by
    averaging Y over all sites not in S.
    """
    n_sites = Y.ndim
    terms = {}
    for sites in site_subsets(n_sites):
        other_axes = tuple(i for i in range(n_sites) if i not in sites)
        subspace_mean = Y.mean(axis=other_axes)          # shape (q,)*k
        terms[sites] = subspace_mean - _sum_of_lower_order(terms, sites)
    return terms


def rba_coefficients(Y, reference):
    """Exact reference-based effects of a complete landscape.

    ``reference`` is the tuple of reference state indices (e.g. AAAA -> (0,0,0,0)).
    For each site combination S (in order of increasing size),

        lambda_S(s) = y(genotype with states s at S and reference elsewhere)
                      - sum over proper subsets T of S of lambda_T(s restricted to T).

    lambda_0 is the reference phenotype, first-order terms are single-mutant
    effects, second-order terms are deviations of double mutants from additivity,
    and so on. Entries where any site in S carries its reference state come out
    exactly zero, which is the RBA constraint.
    """
    n_sites = Y.ndim
    terms = {}
    for sites in site_subsets(n_sites):
        index = tuple(slice(None) if i in sites else reference[i]
                      for i in range(n_sites))
        context_phenotype = Y[index]                       # shape (q,)*k
        terms[sites] = context_phenotype - _sum_of_lower_order(terms, sites)
    return terms


def _sum_of_lower_order(terms, sites):
    """Sum of all already-computed terms on proper subsets of ``sites``."""
    total = np.zeros((1,) * len(sites))
    for k in range(len(sites)):
        for sub in combinations(sites, k):
            total = total + expand_term(terms[sub], sub, sites)
    return total


# ---------------------------------------------------------------------------
# Constraint checks and summaries
# ---------------------------------------------------------------------------

def rfa_zero_sum_violation(terms):
    """Largest |mean| of any 1-D section of any non-intercept RFA term array.

    The zero-mean property (SI section 2.4) requires this to be 0 (to rounding).
    """
    worst = 0.0
    for sites, term in terms.items():
        for axis in range(len(sites)):
            worst = max(worst, float(np.max(np.abs(term.mean(axis=axis)))))
    return worst


def rba_reference_violation(terms, reference):
    """Largest |value| of any RBA term entry that involves a reference state."""
    worst = 0.0
    for sites, term in terms.items():
        for axis, site in enumerate(sites):
            ref_section = np.take(term, reference[site], axis=axis)
            worst = max(worst, float(np.max(np.abs(ref_section))))
    return worst


def rfa_variance_by_order(terms, n_states):
    """Phenotypic variance attributable to each order (SI section 2.7).

    Var(y|G) = sum over non-intercept effects e of e**2 / q**order(e).
    Returns a dict {order: variance}; values sum to the population variance of Y.
    """
    variance = {}
    for sites, term in terms.items():
        k = len(sites)
        if k == 0:
            continue
        variance[k] = variance.get(k, 0.0) + float(np.sum(term ** 2)) / n_states ** k
    return variance


def terms_by_order(terms, mask=None):
    """Concatenate term values per order into flat arrays: {order: values}.

    ``mask`` (optional) is a function (sites, term) -> boolean array selecting
    which entries to keep, e.g. to drop RBA entries involving the reference.
    """
    out = {}
    for sites, term in terms.items():
        k = len(sites)
        values = term[mask(sites, term)] if mask is not None else np.ravel(term)
        out.setdefault(k, []).append(np.ravel(values))
    return {k: np.concatenate(v) for k, v in sorted(out.items())}


def rba_mutant_mask(reference):
    """Mask function selecting RBA entries in which every site carries a mutant state."""
    def mask(sites, term):
        keep = np.ones(np.shape(term), dtype=bool)
        for axis, site in enumerate(sites):
            index = [slice(None)] * len(sites)
            index[axis] = reference[site]
            keep[tuple(index)] = False
        return keep
    return mask


# ---------------------------------------------------------------------------
# Tables for saving
# ---------------------------------------------------------------------------

def term_label(sites, states, state_names):
    """Human-readable term label, e.g. '1A' or '1A:3V' (sites are 1-based)."""
    if len(sites) == 0:
        return "intercept"
    return ":".join(f"{site + 1}{state_names[s]}" for site, s in zip(sites, states))


def terms_to_table(terms, state_names, framework, reference=None):
    """Long-format DataFrame of coefficients.

    RFA: all q**k entries of every term (625 rows for 4 sites x 4 states), plus
    each term's variance contribution e**2 / q**k.
    RBA: only entries in which every site carries a non-reference state
    (256 rows for 4 sites x 4 states); all other RBA entries are zero by
    definition.
    """
    q = len(state_names)
    rows = []
    for sites, term in sorted(terms.items(), key=lambda kv: (len(kv[0]), kv[0])):
        k = len(sites)
        for states in product(range(q), repeat=k):
            if framework == "RBA" and any(s == reference[site]
                                          for site, s in zip(sites, states)):
                continue
            value = float(term[states]) if k > 0 else float(term)
            row = {
                "term": term_label(sites, states, state_names),
                "order": k,
                "sites": ",".join(str(site + 1) for site in sites),
                "states": ",".join(state_names[s] for s in states),
                "coefficient": value,
            }
            if framework == "RFA":
                row["variance_contribution"] = value ** 2 / q ** k if k > 0 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def landscape_to_table(Y, state_names, reference):
    """Long-format genotype-phenotype table (one row per genotype)."""
    n_sites, q = Y.ndim, Y.shape[0]
    genotypes = all_genotypes(n_sites, q)
    table = pd.DataFrame({
        "genotype": ["".join(state_names[s] for s in g) for g in genotypes],
    })
    for i in range(n_sites):
        table[f"site_{i + 1}"] = [state_names[s] for s in genotypes[:, i]]
    table["n_mutations"] = (genotypes != np.array(reference)).sum(axis=1)
    table["phenotype"] = Y.ravel()
    return table


# ---------------------------------------------------------------------------
# Hand-worked sanity example
# ---------------------------------------------------------------------------

def check_two_site_example():
    """Check both formalisms on a 2-site, 2-state landscape solvable by hand.

    Phenotypes: y(00)=0, y(01)=0, y(10)=0, y(11)=1 (an AND-like interaction).
      RFA: e0 = 1/4; e1 = e2 = (-1/4, +1/4);
           e12 = [[+1/4, -1/4], [-1/4, +1/4]].
      RBA (reference 00): lambda0 = 0; lambda1(1) = lambda2(1) = 0;
           lambda12(1,1) = 1; all reference-involving entries 0.
    Raises AssertionError if anything is off.
    """
    Y = np.array([[0.0, 0.0], [0.0, 1.0]])
    rfa = rfa_coefficients(Y)
    assert np.isclose(rfa[()], 0.25)
    assert np.allclose(rfa[(0,)], [-0.25, 0.25])
    assert np.allclose(rfa[(1,)], [-0.25, 0.25])
    assert np.allclose(rfa[(0, 1)], [[0.25, -0.25], [-0.25, 0.25]])

    rba = rba_coefficients(Y, reference=(0, 0))
    assert np.isclose(rba[()], 0.0)
    assert np.allclose(rba[(0,)], [0.0, 0.0])
    assert np.allclose(rba[(1,)], [0.0, 0.0])
    assert np.allclose(rba[(0, 1)], [[0.0, 0.0], [0.0, 1.0]])

    assert np.allclose(phenotypes_from_terms(rfa, 2, 2), Y)
    assert np.allclose(phenotypes_from_terms(rba, 2, 2), Y)
    assert np.isclose(sum(rfa_variance_by_order(rfa, 2).values()), Y.var())
    return True
