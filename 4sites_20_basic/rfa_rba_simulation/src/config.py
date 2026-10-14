# Four amino-acid states allowed at each of the four varying sites.
STATES = ("A", "V", "L", "G")

# RBA reference state at every varying site.
REFERENCE_STATE = "A"

# A 20-aa background sequence. Only four positions vary.
BASE_SEQUENCE = "MKTAYIAKQRQISFVKSHFS"

# 1-based positions in the 20-aa sequence that will vary.
VARIABLE_POSITIONS = (4, 8, 12, 16)

# Random seed for reproducibility.
RANDOM_SEED = 22

# Standard deviation used when generating pure RFA effects at each order.
# Higher-order effects are made progressively smaller, but all four orders are present.
ORDER_SCALES = {
    1: 1.00,
    2: 0.60,
    3: 0.30,
    4: 0.15,
}

INTERCEPT = 10.0
