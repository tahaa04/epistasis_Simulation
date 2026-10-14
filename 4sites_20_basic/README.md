# RFA vs RBA simulation — stage 1

This stage does three things:

1. Creates a complete combinatorial landscape with 4 variable sites and 4 amino-acid states per site:
   4^4 = 256 genotypes.
2. Generates one synthetic phenotype for every genotype from a known full RFA architecture containing
   first-, second-, third-, and fourth-order effects.
3. Calculates and stores the exact ("true") coefficients from the complete phenotype table for:
   - reference-free analysis (RFA)
   - reference-based analysis (RBA), using A as the reference state at every variable site.

No genotypes are removed yet. No sigmoid, measurement noise, or regularization is used yet.

## Folder structure

rfa_rba_simulation/
├── README.md
├── requirements.txt
├── run_stage1.py
├── src/
│   ├── config.py
│   ├── generate_landscape.py
│   ├── rfa.py
│   └── rba.py
├── data/
│   ├── full/
│   │   └── complete_landscape.csv
│   └── true_coefficients/
│       ├── rfa_true_coefficients.csv
│       └── rba_true_coefficients.csv
└── results/

## Run

From the project folder:

```bash
pip install -r requirements.txt
python run_stage1.py
```

## Important representations

### RFA
RFA coefficients are stored for every state-combination at each included site-combination.
For example, for sites (1, 2), all 4 × 4 = 16 state pairs are stored.
They satisfy the RFA zero-mean constraints.

### RBA
The reference genotype is AAAA at the four variable sites.
Any coefficient involving the reference state A is zero by definition, so only non-reference
states V, L, and G need to be stored. For example, a pair of sites has 3 × 3 = 9 independent
pairwise RBA coefficients.

The full RFA and full RBA decompositions reconstruct the same 256 phenotype values exactly.


## Stage 2: incomplete datasets and regression

`run_stage2.py` now:

1. Randomly removes about 15% and exactly 25% of the 256 genotypes.
2. Uses one fixed random permutation, so the 15%-missing genotypes are a subset
   of the 25%-missing genotypes.
3. Fits first-, second-, and third-order:
   - RFA ordinary least-squares regressions using sum-to-zero/effect coding.
   - RBA ordinary least-squares regressions using treatment/reference coding
     with A as the reference state.
4. Converts fitted RFA coefficients back to the same expanded representation as
   `rfa_true_coefficients.csv`.
5. Stores a direct true-vs-estimated coefficient table for every method,
   missingness level, and fitted maximum order.
6. Stores RMSE, MAE, and Pearson correlation coefficient-recovery metrics.

A full fourth-order unregularized fit is intentionally not performed after
removing data. A fourth-order model has 256 independent parameters, but the
15%- and 25%-missing datasets have only 218 and 192 observations. Its ordinary
least-squares coefficients therefore would not be uniquely identifiable.
