"""Configuration for simulated genotype-phenotype landscapes.

Genotype space: 4 sites x 4 states (A, V, L, G) = 256 genotypes.

How coefficients are drawn
--------------------------
Every architecture is defined by generating coefficients ("terms") in ONE
framework (RFA or RBA); phenotypes are the sum of those terms. The intercept is
set to 0 because every landscape is standardized afterwards (mean 0, variance 1),
which removes any constant offset.

``order_scales`` {order: s_k} sets the size of the effects at each order:
  * Draws are Normal(0, 1).
  * RFA: each site-combination's q**k array is projected onto the zero-mean
    subspace (subtracting the mean along every axis), so the zero-sum constraints
    hold exactly.
  * RBA: only entries where every site carries a mutant (non-reference) state
    are drawn; entries involving the reference state are 0 by definition.
  * All drawn (nonzero) terms of order k are then rescaled together so that
    their root-mean-square equals s_k EXACTLY. This makes "decreasing" and
    "similar-sized" true properties of every realization, not just on average.
  * Orders absent from ``order_scales`` have all terms equal to 0.

Sparse architectures (``kind="sparse"``), target nonzero fraction f = 0.10 of the
non-intercept terms at each order:
  * RBA: exactly round(f * n_k) mutant terms of order k are nonzero
    (at least 1), where n_k = C(4,k) * 3**k; the rest are 0.
  * RFA: the zero-sum constraint forbids a single isolated nonzero term, so the
    sparsest building block is an "elementary contrast": pick one site
    combination and two distinct states (a_i, b_i) at each of its k sites, and set
        e(s) = c * prod_i ([s_i == a_i] - [s_i == b_i]),
    which is nonzero at only 2**k of the 4**k entries yet sums to zero along every
    axis. Order k receives max(1, round(f * n_k / 2**k)) contrasts placed at
    random (n_k = C(4,k) * 4**k), with c ~ Normal(0, 1). Contrasts that overlap
    simply add. The realized nonzero fraction is recorded per realization.

Variance-controlled architectures (``kind="variance"``, RFA only):
  dense zero-sum terms are drawn at every order (scale 1), then each order is
  multiplied by sqrt(target_k / V_k), where V_k = sum(e**2) / 4**k is its variance
  contribution (Park et al. SI section 2.7). Total variance is then exactly 1.

Reproducibility: realization r of architecture number a uses
``np.random.default_rng([BASE_SEED, a, r])`` (a NumPy SeedSequence), giving
independent, reproducible streams.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Genotype space
# ---------------------------------------------------------------------------
STATES = ("A", "V", "L", "G")
N_SITES = 4
N_STATES = len(STATES)
REFERENCE_GENOTYPE = "AAAA"                                  # RBA reference
REFERENCE = tuple(STATES.index(s) for s in REFERENCE_GENOTYPE)

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
N_REALIZATIONS = 10
BASE_SEED = 20240911
SPARSE_NONZERO_FRACTION = 0.10
TOLERANCE = 1e-10            # numerical tolerance for all verification checks

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "generated_landscapes"

# Effect-size profiles (root-mean-square of the terms at each order)
DECREASING_SCALES = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125}     # halves per order
SIMILAR_SCALES = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

# ---------------------------------------------------------------------------
# The 16 architectures
# ---------------------------------------------------------------------------
ARCHITECTURES = [
    # ---- Group A: effect distributions -------------------------------------
    dict(name="A01_rfa_decreasing", group="A", framework="RFA", kind="dense",
         order_scales=DECREASING_SCALES,
         description="RFA terms at all orders; RMS halves with each order (1, 0.5, 0.25, 0.125)."),
    dict(name="A02_rba_decreasing", group="A", framework="RBA", kind="dense",
         order_scales=DECREASING_SCALES,
         description="RBA terms at all orders; RMS halves with each order (1, 0.5, 0.25, 0.125)."),
    dict(name="A03_rfa_similar", group="A", framework="RFA", kind="dense",
         order_scales=SIMILAR_SCALES,
         description="RFA terms at all orders with equal RMS (1) at every order."),
    dict(name="A04_rba_similar", group="A", framework="RBA", kind="dense",
         order_scales=SIMILAR_SCALES,
         description="RBA terms at all orders with equal RMS (1) at every order."),
    dict(name="A05_rfa_sparse", group="A", framework="RFA", kind="sparse",
         order_scales=SIMILAR_SCALES, nonzero_fraction=SPARSE_NONZERO_FRACTION,
         description="Sparse RFA: few elementary zero-sum contrasts per order (~10% nonzero terms), RMS 1 among nonzero terms."),
    dict(name="A06_rba_sparse", group="A", framework="RBA", kind="sparse",
         order_scales=SIMILAR_SCALES, nonzero_fraction=SPARSE_NONZERO_FRACTION,
         description="Sparse RBA: 10% of mutant terms per order nonzero, RMS 1 among nonzero terms."),

    # ---- Group B: order-specific effects -----------------------------------
    *[dict(name=f"B{idx:02d}_rfa_order{k}_only", group="B", framework="RFA", kind="dense",
           order_scales={k: 1.0},
           description=f"RFA terms only at order {k} (RMS 1); all other orders zero.")
      for idx, k in zip(range(7, 11), range(1, 5))],
    *[dict(name=f"B{idx:02d}_rba_order{k}_only", group="B", framework="RBA", kind="dense",
           order_scales={k: 1.0},
           description=f"RBA terms only at order {k} (RMS 1); all other orders zero.")
      for idx, k in zip(range(11, 15), range(1, 5))],

    # ---- Group C: variance-controlled landscapes ---------------------------
    dict(name="C15_rfa_variance_equal", group="C", framework="RFA", kind="variance",
         variance_targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},
         description="RFA with equal variance contributions from orders 1-4 (25% each)."),
    dict(name="C16_rfa_variance_increasing", group="C", framework="RFA", kind="variance",
         variance_targets={1: 0.10, 2: 0.15, 3: 0.30, 4: 0.45},
         description="RFA with variance contributions 10/15/30/45% from orders 1-4."),
]
