"""Central configuration for the OptionMC extension.

Every number a reader might want to change lives here, so no experiment script
hard-codes a market assumption. Paths are absolute so scripts run from any
working directory.
"""
from pathlib import Path

# --- directories ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- reproducibility -----------------------------------------------------
SEED = 42                      # every experiment seeds from this
TRADING_DAYS_PER_YEAR = 252    # scope section 13

# --- market data (scope sections 3 and 6) --------------------------------
TICKER = "SPY"                 # American-style ETF options; NOT SPX/XSP
HISTORY_PERIOD = "5y"
SPY_HISTORY_CSV = DATA_DIR / "spy_history.csv"
SPY_OPTION_CSV = DATA_DIR / "spy_option_snapshot.csv"
RISK_FREE_CSV = DATA_DIR / "risk_free_dgs3mo.csv"

EXPIRY_MIN_DAYS = 60           # scope: expiry roughly 60-90 days out
EXPIRY_MAX_DAYS = 90
STRIKE_MIN_MONEYNESS = 0.95    # scope: strike near 95%-100% of S0
STRIKE_MAX_MONEYNESS = 1.00

FRED_SERIES = "DGS3MO"         # 3-Month Treasury Constant Maturity Rate
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

# --- LSMC defaults (scope section 8) -------------------------------------
LSMC_N_PATHS = 10_000
LSMC_N_STEPS = 50
LSMC_DEGREE = 2                # first implementation: 1, S, S^2
LSMC_ANTITHETIC = True

# --- binomial benchmark (scope section 9) --------------------------------
BINOMIAL_N_STEPS = 2_000

# --- numerical experiments (scope section 11) ----------------------------
CONVERGENCE_PATHS = [1_000, 5_000, 10_000, 25_000, 50_000]
DISCRETIZATION_STEPS = [10, 25, 50, 100]
REGRESSION_DEGREES = [1, 2, 3]

# Every configuration is run this many times with independent seeds. One run
# reports one draw from the error distribution, which is not enough to answer
# "does a higher polynomial degree actually help?" -- the answer would flip
# with the seed.
N_REPLICATIONS = 30

# --- pricing grid and interpolation (scope section 15) -------------------
GRID_MIN_MONEYNESS = 0.60      # 0.60*S0 ... 1.40*S0
GRID_MAX_MONEYNESS = 1.40
GRID_N_POINTS = 65
INTERPOLATION_CHECK_POINTS = 15   # scope: 10-20 random check points

# The grid is priced once and read 50,000 times, so it is worth more paths
# than a single valuation would need. Common random numbers make this cheap:
# the paths are simulated once and rescaled for every grid point.
GRID_N_PATHS = 200_000
GRID_N_STEPS = 50

# The grid uses a cubic basis rather than the quadratic one used elsewhere.
# Measured, not assumed: against a Bermudan tree on the same exercise grid the
# worst-node bias is 0.238 (degree 1), 0.104 (degree 2), 0.054 (degree 3),
# 0.062 (degree 4). Degree 2 leaves a visible negative bias across the
# in-the-money nodes, where a quadratic cannot follow the continuation value
# over so wide a spot range; degree 4 starts fitting noise and flips the bias
# positive. Degree 3 is the turning point.
GRID_DEGREE = 3

INTERPOLATION_METHOD = "pchip"

# --- portfolio and risk (scope sections 3, 12, 14, 17) -------------------
SHARES = 100                   # Portfolio A: 100 SPY shares
CONTRACT_MULTIPLIER = 100      # 1 option contract == 100 shares
N_RISK_SCENARIOS = 50_000
RISK_HORIZON_DAYS = 10         # 10 trading days
CONFIDENCE_LEVELS = [0.95, 0.99]

# --- Longstaff-Schwartz (2001) Table 1 benchmark parameters --------------
# Used to validate our LSMC and binomial code against published values
# before any SPY data is involved. These are the paper's parameters, not
# invented numbers.
LS_BENCHMARK = {
    "K": 40.0,
    "r": 0.06,
    "q": 0.0,
    "exercise_points_per_year": 50,
}

# --- cross-section validation (advanced phase 2) -------------------------
CROSS_SECTION_CSV = DATA_DIR / "spy_option_cross_section.csv"
CROSS_SECTION_MIN_MONEYNESS = 0.90
CROSS_SECTION_MAX_MONEYNESS = 1.05
CROSS_SECTION_MAX_SPREAD_PERCENT = 10.0

# SPY lists strikes a dollar apart near the money. Calibrating on one and
# testing on its neighbour would ask the smile to interpolate across a single
# dollar, which any curve does almost exactly -- the test would pass without
# testing anything. Thinning to this spacing puts a real gap between adjacent
# calibration points. Reported, not silent: the count before and after is
# printed and saved.
CROSS_SECTION_STRIKE_SPACING = 5.0

# Spacings compared in the calibration-density study, in dollars.
CROSS_SECTION_SPACING_STUDY = [2.0, 5.0, 10.0, 20.0, 40.0]

CROSS_SECTION_LSMC_PATHS = 20_000
CROSS_SECTION_IV_TREE_STEPS = 500
