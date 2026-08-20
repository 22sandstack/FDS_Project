from __future__ import annotations
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import write_csv_atomic, write_parquet_atomic
from .config import FEATURES_40, FEATURES_40_LAG1, FEATURES_40_LAG1_AVAILABLE, FEATURES_40_VELOCITY, ExperimentConfig
from .evaluation import fractional_quantile_membership, has_cross_sectional_signal, newey_west_tstat, performance_stats
LGBM_COMPONENT_ID = 'LGBM_40'
DEEPSET_COMPONENT_ID = 'DEEPSET_40_DYNAMIC'
DEFAULT_NW_LAGS = 6
DEFAULT_PERMUTATION_REPEATS = 3
DEFAULT_PERMUTATION_SEED = 42

def chosen_output_dir(config: ExperimentConfig, chosen_model_id: str) -> Path:
    path = config.run_dir / 'chosen_model_analysis' / chosen_model_id
    path.mkdir(parents=True, exist_ok=True)
    return path

def _prediction_path(config: ExperimentConfig, model_id: str) -> Path:
    return config.run_dir / 'predictions' / f'{model_id}.parquet'

def _rank_ic_path(config: ExperimentConfig, model_id: str) -> Path:
    return config.run_dir / 'diagnostics' / f'{model_id}_monthly_rank_ic.parquet'

def _portfolio_path(config: ExperimentConfig, model_id: str) -> Path:
    return config.run_dir / 'portfolios' / f'{model_id}_long_short.parquet'

def _load_required(path: Path, columns: Sequence[str] | None=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Required artifact is missing: {path}')
    frame = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    if 'eom' in frame.columns:
        frame['eom'] = pd.to_datetime(frame['eom'])
    return frame

def _normal_two_sided_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))

def _holm_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = pd.to_numeric(p_values, errors='coerce').dropna()
    if valid.empty:
        return adjusted
    ordered = valid.sort_values()
    m = len(ordered)
    running = 0.0
    for rank, (index, p_value) in enumerate(ordered.items(), start=1):
        candidate = min(1.0, (m - rank + 1) * float(p_value))
        running = max(running, candidate)
        adjusted.loc[index] = running
    return adjusted

def _align_monthly_series(chosen: pd.DataFrame, component: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    left = chosen[['eom', value_column]].rename(columns={value_column: 'chosen_value'})
    right = component[['eom', value_column]].rename(columns={value_column: 'component_value'})
    aligned = left.merge(right, on='eom', how='inner', validate='one_to_one')
    aligned['difference'] = aligned['chosen_value'] - aligned['component_value']
    return aligned

def compare_model_pairs(
    config: ExperimentConfig,
    pairs: Sequence[tuple[str, str]],
    *,
    metrics: Sequence[str] = ('rank_ic', 'long_short_return'),
    newey_west_lags: int = DEFAULT_NW_LAGS,
) -> pd.DataFrame:
    """Calculate paired monthly tests for a supplied list of model contrasts."""
    if not pairs:
        raise ValueError('At least one model pair is required.')
    allowed_metrics = {
        'rank_ic': ('rank_ic', _rank_ic_path),
        'long_short_return': ('long_short_ret', _portfolio_path),
    }
    unknown_metrics = set(metrics) - set(allowed_metrics)
    if unknown_metrics:
        raise ValueError(f'Unknown comparison metrics: {sorted(unknown_metrics)}')
    if not metrics:
        raise ValueError('At least one comparison metric is required.')

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for model_a, model_b in pairs:
        model_a = str(model_a).strip()
        model_b = str(model_b).strip()
        if not model_a or not model_b:
            raise ValueError('Model names in each pair must be non-empty.')
        if model_a == model_b:
            raise ValueError(f'A model cannot be compared with itself: {model_a}')
        pair = (model_a, model_b)
        if pair in seen:
            continue
        seen.add(pair)

        for metric in metrics:
            value_column, path_builder = allowed_metrics[metric]
            model_a_data = _load_required(
                path_builder(config, model_a), columns=['eom', value_column]
            )
            model_b_data = _load_required(
                path_builder(config, model_b), columns=['eom', value_column]
            )
            aligned = _align_monthly_series(
                model_a_data, model_b_data, value_column=value_column
            )
            valid = aligned[['chosen_value', 'component_value']].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            difference = valid['chosen_value'] - valid['component_value']
            if difference.empty:
                raise ValueError(
                    f'No aligned finite observations for {model_a} and {model_b} ({metric}).'
                )
            model_a_mean = float(valid['chosen_value'].mean())
            model_b_mean = float(valid['component_value'].mean())
            mean_difference = float(difference.mean())
            t_stat = newey_west_tstat(difference, newey_west_lags)
            p_value = _normal_two_sided_pvalue(t_stat)
            rows.append({
                'model_a': model_a,
                'model_b': model_b,
                'metric': metric,
                'n_months': int(len(difference)),
                'model_a_mean': model_a_mean,
                'model_b_mean': model_b_mean,
                'mean_difference_a_minus_b': mean_difference,
                'newey_west_t_stat': t_stat,
                'p_value': p_value,
            })
    return pd.DataFrame(rows)

def compare_prespecified_families(
    config: ExperimentConfig,
    chosen_model_id: str,
    *,
    newey_west_lags: int = DEFAULT_NW_LAGS,
) -> pd.DataFrame:
    """Run the four frozen paired-test families and adjust within outcome."""
    families: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
        'F1_ENSEMBLE_COMPONENTS': (
            'Does the fixed ensemble improve on either component?',
            (
                (chosen_model_id, LGBM_COMPONENT_ID),
                (chosen_model_id, DEEPSET_COMPONENT_ID),
            ),
        ),
        'F2_CLOUD_INFORMATION_DESIGN': (
            'What is the value of cloud context relative to matched neural and temporal alternatives?',
            (
                ('DEEPSET_40', 'MLP_40'),
                ('DEEPSET_40_LAG1', 'MLP_40_LAG1'),
                ('DEEPSET_40', 'MLP_40_LAG1'),
            ),
        ),
        'F3_BENCHMARK_PERFORMANCE': (
            'Do the static and dynamic cloud models improve on the selected LightGBM benchmark?',
            (
                ('DEEPSET_40', 'LGBM_40'),
                ('DEEPSET_40_DYNAMIC', 'LGBM_40'),
            ),
        ),
        'F4_TEMPORAL_DEVELOPMENT': (
            'What is the incremental value of lagged and explicitly dynamic inputs?',
            (
                ('MLP_40_LAG1', 'MLP_40'),
                ('DEEPSET_40_LAG1', 'DEEPSET_40'),
                ('DEEPSET_40_DYNAMIC', 'DEEPSET_40'),
                ('DEEPSET_40_DYNAMIC', 'DEEPSET_40_LAG1'),
            ),
        ),
    }
    parts: list[pd.DataFrame] = []
    for family_order, (family, (hypothesis, pairs)) in enumerate(families.items(), start=1):
        part = compare_model_pairs(
            config,
            pairs,
            newey_west_lags=newey_west_lags,
        )
        part.insert(0, 'family_order', family_order)
        part.insert(1, 'family', family)
        part.insert(2, 'family_hypothesis', hypothesis)
        part.insert(3, 'comparison_order', np.repeat(np.arange(1, len(pairs) + 1), 2))
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result['holm_adjusted_p_value'] = result.groupby(
        ['family', 'metric'], group_keys=False
    )['p_value'].transform(_holm_adjust)
    result['holm_reject_5pct'] = result['holm_adjusted_p_value'].lt(0.05)

    def formal_interpretation(row: pd.Series) -> str:
        direction = (
            f"{row.model_a} is higher than {row.model_b}"
            if row.mean_difference_a_minus_b > 0
            else f"{row.model_b} is higher than {row.model_a}"
            if row.mean_difference_a_minus_b < 0
            else f"{row.model_a} and {row.model_b} have equal sample means"
        )
        evidence = (
            'The difference remains significant after Holm adjustment within this family and outcome.'
            if row.holm_reject_5pct
            else 'The difference is not significant after Holm adjustment within this family and outcome.'
        )
        return f"{direction} (A-B = {row.mean_difference_a_minus_b:+.4f}). {evidence}"

    result['formal_interpretation'] = result.apply(formal_interpretation, axis=1)
    result = result.sort_values(
        ['family_order', 'comparison_order', 'metric']
    ).reset_index(drop=True)
    write_csv_atomic(
        result,
        chosen_output_dir(config, chosen_model_id)
        / 'prespecified_paired_comparisons.csv',
    )
    return result

def build_volatility_regimes(config: ExperimentConfig) -> pd.DataFrame:
    columns = ['eom', 'excntry', 'id', 'size_grp', 'me', 'ret_exc']
    data = pd.read_parquet(config.data_path, columns=columns)
    data['eom'] = pd.to_datetime(data['eom'])
    size = data['size_grp'].astype('string').str.strip().str.lower()
    data = data.loc[data['eom'].dt.year.between(config.universe.start_year, config.universe.end_year) & data['excntry'].eq(config.universe.country) & data['id'].notna() & size.isin(config.universe.allowed_size_groups)].copy()
    data['size_grp'] = size.loc[data.index]
    data['me'] = pd.to_numeric(data['me'], errors='coerce')
    data['ret_exc'] = pd.to_numeric(data['ret_exc'], errors='coerce')
    data = data.sort_values(['id', 'eom'])
    previous_eom = data.groupby('id')['eom'].shift(1)
    exact_previous_month = previous_eom.eq(data['eom'] - pd.offsets.MonthEnd(1))
    data['beginning_me'] = data.groupby('id')['me'].shift(1).where(exact_previous_month)
    usable = data.loc[np.isfinite(data['ret_exc']) & np.isfinite(data['beginning_me']) & data['beginning_me'].gt(0)].copy()
    usable['weighted_return'] = usable['ret_exc'] * usable['beginning_me']
    market = usable.groupby('eom', as_index=False).agg(weighted_return=('weighted_return', 'sum'), market_weight=('beginning_me', 'sum')).sort_values('eom')
    market['market_ret_exc'] = market['weighted_return'] / market['market_weight']
    market['trailing_12m_market_vol'] = market['market_ret_exc'].rolling(12, min_periods=12).std(ddof=1)
    market['past_expanding_median_vol'] = market['trailing_12m_market_vol'].expanding(min_periods=60).median().shift(1)
    market['regime'] = np.where(market['trailing_12m_market_vol'] > market['past_expanding_median_vol'], 'HIGH_VOL', 'LOW_VOL')
    market.loc[market['past_expanding_median_vol'].isna(), 'regime'] = pd.NA
    return market[['eom', 'market_ret_exc', 'trailing_12m_market_vol', 'past_expanding_median_vol', 'regime']]

def analyze_regime_stability(config: ExperimentConfig, chosen_model_id: str, regimes: pd.DataFrame | None=None, *, newey_west_lags: int=DEFAULT_NW_LAGS) -> pd.DataFrame:
    if regimes is None:
        regimes = build_volatility_regimes(config)
    rank_ic = _load_required(_rank_ic_path(config, chosen_model_id), columns=['eom', 'rank_ic'])
    portfolio = _load_required(_portfolio_path(config, chosen_model_id), columns=['eom', 'long_short_ret'])
    monthly = rank_ic.merge(portfolio, on='eom', how='inner', validate='one_to_one').merge(regimes, on='eom', how='left', validate='one_to_one').dropna(subset=['regime'])
    rows: list[dict] = []
    for regime, group in monthly.groupby('regime', sort=True):
        valid_ic = group['rank_ic'].replace([np.inf, -np.inf], np.nan).dropna()
        stats = performance_stats(group['long_short_ret'], newey_west_lags)
        rows.append({'regime': regime, 'n_months': int(group['eom'].nunique()), 'mean_rank_ic': float(valid_ic.mean()) if not valid_ic.empty else np.nan, 'rank_ic_newey_west_t_stat': newey_west_tstat(valid_ic, newey_west_lags), 'annualized_return': stats['annualized_return'], 'sharpe_ratio': stats['sharpe_ratio']})
    result = pd.DataFrame(rows)
    output = chosen_output_dir(config, chosen_model_id)
    write_parquet_atomic(monthly, output / 'monthly_regime_results.parquet')
    result.to_csv(output / 'regime_stability_summary.csv', index=False)
    return result

def characteristic_families() -> dict[str, tuple[str, ...]]:
    return {base: (base, FEATURES_40_LAG1[index], FEATURES_40_VELOCITY[index]) for index, base in enumerate(FEATURES_40)}

def mean_monthly_rank_ic(data: pd.DataFrame, *, prediction_column: str='y_pred', target_column: str='y_true', minimum_stocks: int=20) -> float:
    values: list[float] = []
    for _, month in data.groupby('eom', sort=True):
        valid = month[[prediction_column, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < minimum_stocks or valid[prediction_column].nunique() < 2:
            continue
        rank_ic = valid[prediction_column].rank(method='average').corr(valid[target_column].rank(method='average'))
        if np.isfinite(rank_ic):
            values.append(float(rank_ic))
    return float(np.mean(values)) if values else np.nan

def _permute_within_month(data: pd.DataFrame, columns: Sequence[str], rng: np.random.Generator, *, strata: Sequence[str]=()) -> pd.DataFrame:
    result = data.copy()
    group_columns = ['eom', *strata]
    for _, index in result.groupby(group_columns, sort=False, dropna=False).groups.items():
        positions = np.asarray(list(index))
        if len(positions) < 2:
            continue
        shuffled = rng.permutation(len(positions))
        values = result.loc[positions, list(columns)].to_numpy(copy=True)
        result.loc[positions, list(columns)] = values[shuffled]
    return result

def permutation_importance(data: pd.DataFrame, *, predict_fn: Callable[[pd.DataFrame], np.ndarray | pd.Series], feature_groups: Mapping[str, Sequence[str]] | None=None, regimes: pd.DataFrame | None=None, n_repeats: int=DEFAULT_PERMUTATION_REPEATS, seed: int=DEFAULT_PERMUTATION_SEED, strata: Sequence[str]=(), checkpoint_path: Path | None=None) -> pd.DataFrame:
    if n_repeats < 1:
        raise ValueError('n_repeats must be at least 1.')
    required = {'eom', 'y_true'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Permutation data is missing required columns: {sorted(missing)}')
    feature_groups = dict(characteristic_families() if feature_groups is None else feature_groups)
    missing_features = {feature for columns in feature_groups.values() for feature in columns if feature not in data.columns}
    if missing_features:
        raise ValueError(f'Permutation data is missing features: {sorted(missing_features)}')
    working = data.copy()
    working['eom'] = pd.to_datetime(working['eom'])
    if regimes is not None:
        regime_frame = regimes[['eom', 'regime']].copy()
        regime_frame['eom'] = pd.to_datetime(regime_frame['eom'])
        working = working.merge(regime_frame, on='eom', how='left', validate='many_to_one')
    baseline_prediction = np.asarray(predict_fn(working.copy()), dtype=float)
    if len(baseline_prediction) != len(working):
        raise ValueError('predict_fn returned the wrong number of predictions.')
    working['y_pred'] = baseline_prediction
    subsets: dict[str, pd.Index] = {'ALL': working.index}
    if regimes is not None:
        for regime in ('HIGH_VOL', 'LOW_VOL'):
            subsets[regime] = working.index[working['regime'].eq(regime)]
    baseline_scores = {label: mean_monthly_rank_ic(working.loc[index]) for label, index in subsets.items()}
    rows: list[dict] = []
    completed: set[str] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = pd.read_csv(checkpoint_path)
        if 'characteristic' in saved.columns:
            rows = saved.to_dict('records')
            completed = set(saved['characteristic'].astype(str))
    base_columns = [column for column in working.columns if column != 'y_pred']
    for family_number, (characteristic, columns) in enumerate(feature_groups.items()):
        if characteristic in completed:
            continue
        repeat_scores: dict[str, list[float]] = {label: [] for label in subsets}
        family_rng = np.random.default_rng(seed + family_number * 10_000)
        for _ in range(n_repeats):
            permuted = _permute_within_month(working[base_columns], columns, family_rng, strata=strata)
            prediction = np.asarray(predict_fn(permuted), dtype=float)
            if len(prediction) != len(permuted):
                raise ValueError('predict_fn returned the wrong number of predictions.')
            permuted['y_pred'] = prediction
            for label, index in subsets.items():
                repeat_scores[label].append(mean_monthly_rank_ic(permuted.loc[index]))
        row = {'characteristic': characteristic}
        for label in subsets:
            score_values = np.asarray(repeat_scores[label], dtype=float)
            mean_permuted = float(np.nanmean(score_values)) if np.isfinite(score_values).any() else np.nan
            baseline = baseline_scores[label]
            importance = baseline - mean_permuted if np.isfinite(baseline) and np.isfinite(mean_permuted) else np.nan
            prefix = label.lower()
            row[f'{prefix}_baseline_rank_ic'] = baseline
            row[f'{prefix}_permuted_rank_ic'] = mean_permuted
            row[f'{prefix}_importance'] = importance
        rows.append(row)
        if checkpoint_path is not None:
            write_csv_atomic(pd.DataFrame(rows), checkpoint_path)
    result = pd.DataFrame(rows)
    for label in subsets:
        column = f'{label.lower()}_importance'
        result[f'{label.lower()}_rank'] = result[column].rank(ascending=False, method='min')
    return result.sort_values(['all_rank', 'characteristic'], na_position='last').reset_index(drop=True)

def analyze_characteristic_stability(config: ExperimentConfig, chosen_model_id: str, *, data: pd.DataFrame, predict_fn: Callable[[pd.DataFrame], np.ndarray | pd.Series], regimes: pd.DataFrame | None=None, n_repeats: int=DEFAULT_PERMUTATION_REPEATS, seed: int=DEFAULT_PERMUTATION_SEED) -> pd.DataFrame:
    if regimes is None:
        regimes = build_volatility_regimes(config)
    output = chosen_output_dir(config, chosen_model_id)
    path = output / 'deepset_permutation_importance.csv'
    result = permutation_importance(data, predict_fn=predict_fn, regimes=regimes, n_repeats=n_repeats, seed=seed, strata=(FEATURES_40_LAG1_AVAILABLE,), checkpoint_path=path)
    write_csv_atomic(result, path)
    return result

def summarize_lgbm_shap(shap_values: pd.DataFrame) -> pd.DataFrame:
    """Pool annual OOS TreeSHAP sums using their observation counts."""
    required = {'regime', 'feature', 'sum_abs_shap', 'n_observations'}
    missing = required - set(shap_values.columns)
    if missing:
        raise ValueError(f'SHAP data is missing required columns: {sorted(missing)}')
    result = shap_values.groupby(['regime', 'feature'], as_index=False).agg(
        sum_abs_shap=('sum_abs_shap', 'sum'),
        n_observations=('n_observations', 'sum'),
    )
    result['mean_abs_shap'] = result['sum_abs_shap'] / result['n_observations']
    result['importance_share'] = result.groupby('regime')['mean_abs_shap'].transform(lambda values: values / values.sum())
    result['rank'] = result.groupby('regime')['mean_abs_shap'].rank(ascending=False, method='min')
    return result.sort_values(['regime', 'rank', 'feature']).reset_index(drop=True)

def _form_weighted_decile_portfolio(data: pd.DataFrame, *, n_groups: int=10, weight_column: str | None=None) -> pd.DataFrame:
    rows: list[dict] = []
    for eom, month in data.groupby('eom', sort=True):
        forecast = month.loc[np.isfinite(pd.to_numeric(month['y_pred'], errors='coerce'))].copy()
        if forecast.empty or not has_cross_sectional_signal(forecast):
            rows.append({'eom': eom, 'long_short_ret': 0.0})
            continue
        assigned = fractional_quantile_membership(forecast, n_groups)
        realized = pd.to_numeric(assigned['y_true'], errors='coerce')
        finite_return = np.isfinite(realized)
        if weight_column is None:
            base_weight = pd.Series(1.0, index=assigned.index)
        else:
            base_weight = pd.to_numeric(assigned[weight_column], errors='coerce').where(lambda x: np.isfinite(x) & x.gt(0), 0.0)
        bottom_weight = (assigned['membership_1'] * base_weight).where(finite_return, 0.0)
        top_weight = (assigned[f'membership_{n_groups}'] * base_weight).where(finite_return, 0.0)

        def side_return(weights: pd.Series) -> float:
            mass = float(weights.sum())
            if mass <= 0:
                return np.nan
            return float(np.dot(weights, realized.fillna(0.0)) / mass)
        bottom_return = side_return(bottom_weight)
        top_return = side_return(top_weight)
        long_short = top_return - bottom_return if np.isfinite(top_return) and np.isfinite(bottom_return) else np.nan
        rows.append({'eom': eom, 'long_short_ret': long_short})
    return pd.DataFrame(rows)

def analyze_portfolio_robustness(config: ExperimentConfig, chosen_model_id: str, *, newey_west_lags: int=DEFAULT_NW_LAGS) -> pd.DataFrame:
    predictions = _load_required(_prediction_path(config, chosen_model_id))
    required = {'eom', 'y_true', 'y_pred', 'me', 'size_grp'}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f'Chosen-model predictions are missing portfolio columns: {sorted(missing)}')
    size = predictions['size_grp'].astype('string').str.strip().str.lower()
    predictions = predictions.copy()
    predictions['size_grp'] = size
    predictions['me'] = pd.to_numeric(predictions['me'], errors='coerce')
    variants = {'BASELINE_EQUAL_WEIGHT': _form_weighted_decile_portfolio(predictions), 'EX_MICRO_EQUAL_WEIGHT': _form_weighted_decile_portfolio(predictions.loc[~predictions['size_grp'].eq('micro')]), 'VALUE_WEIGHTED': _form_weighted_decile_portfolio(predictions, weight_column='me')}
    monthly_parts: list[pd.DataFrame] = []
    rows: list[dict] = []
    for strategy, monthly in variants.items():
        monthly = monthly.copy()
        monthly['strategy'] = strategy
        monthly_parts.append(monthly)
        stats = performance_stats(monthly['long_short_ret'], newey_west_lags)
        rows.append({'strategy': strategy, 'n_months': int(monthly['long_short_ret'].notna().sum()), **stats})
    monthly_result = pd.concat(monthly_parts, ignore_index=True)
    result = pd.DataFrame(rows)
    output = chosen_output_dir(config, chosen_model_id)
    write_parquet_atomic(monthly_result, output / 'portfolio_robustness_monthly.parquet')
    result.to_csv(output / 'portfolio_robustness_summary.csv', index=False)
    return result
