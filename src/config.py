from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CORE20: tuple[str, ...] = (
    "market_equity", "be_me", "ret_12_1", "ret_1_0", "ret_6_1",
    "ret_60_12", "ope_be", "gp_at", "at_gr1", "inv_gr1",
    "netis_at", "oaccruals_at", "ivol_capm_21d", "beta_60m", "rmax5_21d",
    "turnover_126d", "dolvol_126d", "ami_126d", "debt_me", "niq_at",
)

# Frozen before running the feature-breadth experiment. Every block contains
# five market/trading, five valuation/financing, five quality/profitability,
# and five growth/investment/intangible characteristics.
MIXED_BLOCK_A: tuple[str, ...] = (
    "ret_3_1", "resff3_6_1", "prc_highprc_252d", "rvol_21d", "bidaskhl_21d",
    "at_me", "ni_me", "ocf_me", "netdebt_me", "eqnetis_at",
    "ocf_at", "op_at", "f_score", "at_turnover", "qmj_prof",
    "capx_gr1", "sale_gr1", "emp_gr1", "rd_me", "cash_at",
)
MIXED_BLOCK_B: tuple[str, ...] = (
    "ret_9_1", "resff3_12_1", "seas_1_1na", "ivol_capm_252d", "zero_trades_21d",
    "fcf_me", "sale_me", "ebitda_mev", "div12m_me", "dbnetis_at",
    "ni_be", "niq_be", "op_atl1", "ebit_sale", "z_score",
    "capx_gr2", "sale_gr3", "saleq_gr1", "rd_sale", "tangibility",
)
MIXED_BLOCK_C: tuple[str, ...] = (
    "ret_12_7", "seas_2_5na", "ivol_ff3_21d", "beta_dimson_21d", "zero_trades_126d",
    "ebit_bev", "bev_mev", "eqnpo_12m", "eqnpo_me", "debt_gr3",
    "ope_bel1", "o_score", "cop_at", "gp_atl1", "ni_inc8q",
    "capx_gr3", "ppeinv_gr1a", "ncoa_gr1a", "rd5_at", "age",
)
MIXED_BLOCK_D: tuple[str, ...] = (
    "seas_6_10na", "betadown_252d", "iskew_capm_21d", "coskew_21d", "dolvol_var_126d",
    "eqpo_me", "chcsho_12m", "fnl_gr1a", "capex_abn", "ival_me",
    "cop_atl1", "dgp_dsale", "mispricing_perf", "opex_at", "qmj",
    "noa_gr1a", "lnoa_gr1a", "kz_index", "aliq_at", "aliq_mat",
)

FEATURES_20: tuple[str, ...] = CORE20
FEATURES_40: tuple[str, ...] = FEATURES_20 + MIXED_BLOCK_A
FEATURES_60: tuple[str, ...] = FEATURES_40 + MIXED_BLOCK_B
FEATURES_80: tuple[str, ...] = FEATURES_60 + MIXED_BLOCK_C
FEATURES_100: tuple[str, ...] = FEATURES_80 + MIXED_BLOCK_D

FEATURES_40_LAG1: tuple[str, ...] = tuple(
    f"{name}_feature40_lag1" for name in FEATURES_40
)
FEATURES_40_LAG1_AVAILABLE: str = "features40_lag1_available"
FEATURES_40_WITH_LAG1: tuple[str, ...] = (
    FEATURES_40 + FEATURES_40_LAG1 + (FEATURES_40_LAG1_AVAILABLE,)
)
FEATURES_40_LAG2: tuple[str, ...] = tuple(
    f"{name}_feature40_lag2" for name in FEATURES_40
)
FEATURES_40_LAG2_AVAILABLE: str = "features40_lag2_available"
FEATURES_40_VELOCITY: tuple[str, ...] = tuple(
    f"{name}_feature40_velocity1" for name in FEATURES_40
)
FEATURES_40_WITH_LAG2: tuple[str, ...] = (
    FEATURES_40
    + FEATURES_40_LAG1
    + FEATURES_40_LAG2
    + (FEATURES_40_LAG1_AVAILABLE, FEATURES_40_LAG2_AVAILABLE)
)
FEATURES_40_DYNAMIC: tuple[str, ...] = (
    FEATURES_40
    + FEATURES_40_LAG1
    + FEATURES_40_VELOCITY
    + (FEATURES_40_LAG1_AVAILABLE,)
)

for expected_size, feature_set in zip(
    (20, 40, 60, 80, 100),
    (FEATURES_20, FEATURES_40, FEATURES_60, FEATURES_80, FEATURES_100),
):
    if len(feature_set) != expected_size or len(set(feature_set)) != expected_size:
        raise ValueError(f"FEATURES_{expected_size} must contain unique characteristics.")

CORE20_LAG1: tuple[str, ...] = tuple(f"{name}_lag1" for name in CORE20)
CORE20_LAG2: tuple[str, ...] = tuple(f"{name}_lag2" for name in CORE20)
CORE20_VELOCITY: tuple[str, ...] = tuple(f"{name}_velocity1" for name in CORE20)
LAG1_AVAILABLE: str = "core20_lag1_available"
LAG2_AVAILABLE: str = "core20_lag2_available"

CORE20_LAG1_FEATURES: tuple[str, ...] = CORE20 + CORE20_LAG1 + (LAG1_AVAILABLE,)
CORE20_DYNAMIC_FEATURES: tuple[str, ...] = (
    CORE20 + CORE20_LAG1 + CORE20_VELOCITY + (LAG1_AVAILABLE,)
)
CORE20_LAG2_FEATURES: tuple[str, ...] = (
    CORE20 + CORE20_LAG1 + CORE20_LAG2 + (LAG1_AVAILABLE, LAG2_AVAILABLE)
)

if len(CORE20) != 20 or len(set(CORE20)) != 20:
    raise ValueError("CORE20 must contain 20 unique characteristics.")


@dataclass(frozen=True)
class UniverseConfig:
    country: str = "USA"
    start_year: int = 1980
    end_year: int = 2024
    allowed_size_groups: tuple[str, ...] = ("micro", "small", "large", "mega")
    security_id_col: str = "permno"


@dataclass(frozen=True)
class WindowConfig:
    train_years: int = 15
    validation_years: int = 4
    test_years: int = 1
    refit_month: int = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    method: str = "monthly_rank"
    missing_feature_fill: float = 0.0


@dataclass(frozen=True)
class PortfolioConfig:
    n_groups: int = 10
    weighting: str = "equal"
    newey_west_lags: int = 6
    transaction_cost_bps: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    project_dir: Path
    data_path: Path
    output_dir: Path
    selected_models: tuple[str, ...]
    feature_set_id: str = "CORE20_RANKED"
    target_col: str = "ret_exc_lead1m"
    seed: int = 42
    use_gpu: bool = True
    pipeline_version: str = "models_pipeline_v1"
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_id

    def validate(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id cannot be empty.")
        if self.windows.test_years != 1:
            raise ValueError("The current annual runner requires test_years=1.")
        if self.preprocessing.method != "monthly_rank":
            raise ValueError("The initial benchmark feature set uses monthly_rank preprocessing.")
        if self.target_col in CORE20:
            raise ValueError("The target cannot also be a feature.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("project_dir", "data_path", "output_dir"):
            data[key] = str(data[key])
        return data
