
# XGBoost and LightGBM Benchmark

import gc
import json
import time
import pickle
import warnings
from pathlib import Path

import pyarrow as pa
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
import lightgbm as lgb
import optuna
import torch
import matplotlib.pyplot as plt

from benchmark_common import (
    portfolio_metrics, capped_softmax_weights,
    mean_split_legs, long_only_leg,
    weight_l1_turnover, drift_weights,
    apply_overlay_and_costs, predict_at_dates,
    rank_correlation_oos
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


cuda_available = torch.cuda.is_available()
cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None

print(f'cuda available (torch): {cuda_available}')
if cuda_available:
    print(f'cuda device: {cuda_device_name}')


def _empty_cuda_cache():
    if cuda_available and torch is not None:
        torch.cuda.empty_cache()


def _gpu_cleanup():
    gc.collect()
    _empty_cuda_cache()


def _probe(probe_fn, label):
    try:
        probe_x = np.random.randn(200, 10).astype(np.float32)
        probe_y = np.random.randn(200).astype(np.float32)
        probe_fn(probe_x, probe_y)
        return True
    except Exception as exc:
        print(f'{label}: gpu probe failed ({type(exc).__name__}: {exc})')
        return False


xgb_use_cuda = False
lgb_use_gpu = False
if cuda_available:
    xgb_use_cuda = _probe(
        lambda x, y: xgb.XGBRegressor(n_estimators = 5, tree_method = 'hist', device = 'cuda', verbosity = 0).fit(x, y), 'xgboost',
    )
    lgb_use_gpu = _probe(
        lambda x, y: lgb.LGBMRegressor(n_estimators = 5, device = 'gpu', verbose = -1).fit(x, y), 'lightgbm',
    )

xgb_device_params = {'tree_method': 'hist', 'device': 'cuda'} if xgb_use_cuda else {'tree_method': 'hist'}
lgb_device_params = {'device': 'gpu'} if lgb_use_gpu else {}



data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/tree_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

train_end = pd.Timestamp('2015-12-31')
val_end = pd.Timestamp('2020-12-31')

ret_col_1m = 'ret_exc_lead1m'
ret_col = 'ret_exc_lead6m'

rebalance_freq = 6
horizon_months = 6
tc_bps = 25
min_stocks = 30
min_leg_stocks = 10
ret_clip_low = -1.0
ret_clip_high = 1.0

target_vol = 0.10
# vol_lookback_months is retained for reference and for any future monthly
# resolution diagnostic. the period resolution overlay used below for the
# headline metrics only has six month rebalance period returns available,
# so its lookback is set directly as vol_lookback_periods rather than being
# silently derived from vol_lookback_months, since the two are different
# estimators over different sample sizes and should not be conflated
vol_lookback_months = 36
vol_lookback_periods = 6
max_leverage_long_only = 3.0
max_leverage_long_short = 3.0
max_position_weight = 0.05

n_trials_xgb = 50
n_trials_lgb = 50
optuna_seed = 24
n_hpo_months = 36



schema = pq.read_schema(data_path)

non_feature = {
    # identifiers
    'id', 'gvkey', 'iid', 'isin', 'cusip', 'permno', 'permco',
    # dates, country, currency, size grouping
    'eom', 'date', 'excntry', 'curcd', 'size_grp',
    # the prediction target column at the one month horizon, retained here so
    # the cumulative six month target can be constructed below
    ret_col_1m,
    # industry classification codes encoded as float
    'sic', 'naics', 'gics', 'ff49',
    # exchange and share classification codes
    'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd',
    # filter and quality indicators, all encoded as float
    'obs_main', 'exch_main', 'primary_sec', 'common', 'bidask',
    'source_crsp',
    # return calculation metadata
    'adjfct', 'fx', 'ret_lag_dif',
    # raw same period returns, redundant with ret_1_0 short term reversal characteristic
    'ret', 'ret_exc', 'ret_local',
    # level forms of characteristics, redundant with the ranked characteristics
    'me', 'me_company', 'prc', 'prc_local', 'prc_high', 'prc_low',
    'dolvol', 'shares', 'tvol',
}

feature_cols = [
    c for c in schema.names
    if c not in non_feature
    and pa.types.is_floating(schema.field(c).type)
    and '_lag' not in c
]

print(f'feature columns selected: {len(feature_cols)}')

needed = list(dict.fromkeys(
    [c for c in ['id', 'eom', 'excntry', ret_col_1m] + feature_cols
     if c in schema.names]
))

table = pq.read_table(data_path, columns = needed)
cast_fields = []
for field in table.schema:
    if field.name in feature_cols and pa.types.is_float64(field.type):
        cast_fields.append(field.with_type(pa.float32()))
    else:
        cast_fields.append(field)
table = table.cast(pa.schema(cast_fields))

df = table.to_pandas()
del table
gc.collect()
df['eom'] = pd.to_datetime(df['eom'])

df[ret_col_1m] = df[ret_col_1m].clip(lower = ret_clip_low, upper = ret_clip_high)

print(f'loaded: {df.shape[0]:,} rows, {len(feature_cols)} characteristic columns')
print(f'date range: {df["eom"].min().date()} to {df["eom"].max().date()}')
print(f'countries: {df["excntry"].nunique()}')



# Construct the cumulative six month forward target. For each firm and
# month we form the product of one plus the next six one month forward
# returns, minus one. We require all six constituent observations to be
# present. Firms whose forward block contains a gap, namely a delisting or
# a missing return, are dropped from that month's cross section.

df = df.sort_values(['id', 'eom']).reset_index(drop = True)

# group by firm and shift the one month forward return backward by k months
# for k in 0.5, then compound. Shifting in this direction aligns
# ret_exc_lead1m at month t+k onto row t, the return realised
# between t+k and t+k+1, which is exactly the kth component of the six
# month forward block starting at t.

shifted = []
for k in range(horizon_months):
    s = df.groupby('id', sort = False)[ret_col_1m].shift(-k)
    shifted.append(s.to_numpy(dtype = np.float64))

shifted = np.stack(shifted, axis = 1)
valid_block = np.isfinite(shifted).all(axis = 1)

cum = np.where(
    valid_block,
    np.prod(1.0 + shifted, axis = 1) - 1.0,
    np.nan,
)
df[ret_col] = cum.astype(np.float32)

# clip the cumulative target to the same band as the underlying one month
# returns to avoid extreme outliers driving the loss. the band is wider
# than for one month returns because six month compounded returns have
# fatter tails
df[ret_col] = df[ret_col].clip(lower = ret_clip_low * 2.0, upper = ret_clip_high * 2.0)

retained = int(np.isfinite(cum).sum())
print('cumulative six month target constructed')
print(f'retained rows with valid six month forward block: {retained:,} of {len(df):,}')
print(f'retention: {100.0 * retained / len(df):.2f}%')

# drop the one month forward target from feature pool consideration. it
# was retained only to construct the six month target
del shifted
gc.collect()



# Per month preprocessing. For every cross section we rank normalise each
# characteristic to the unit interval, centre by subtracting 0.5 so that
# the cross sectional mean is approximately zero, and then impute the
# remaining missing values to zero. The imputation follows the benchmark
# methodology, under which the cross sectional median maps to 0.5 before
# centering and to zero after centering, so that imputed values do not
# affect the mean of any feature within the cross section.

sorted_eoms = sorted(df['eom'].unique())
all_months = {}
n_feat = len(feature_cols)

for eom in sorted_eoms:
    month = df[df['eom'] == eom].copy()
    month = month[month[ret_col].notna()]
    if len(month) < min_stocks:
        continue

    ids = month['id'].to_numpy()
    r = month[ret_col].to_numpy().astype(np.float64)

    x = np.zeros((len(month), n_feat), dtype = np.float32)
    for j, col in enumerate(feature_cols):
        if col not in month.columns:
            continue
        vals = month[col].astype(np.float64).to_numpy()
        valid = np.isfinite(vals)
        if valid.sum() > 1:
            ranks = pd.Series(vals[valid]).rank(pct = True).to_numpy(dtype = np.float32)
            x[valid, j] = ranks - 0.5

    all_months[eom] = {'ids': ids, 'r': r, 'x': x}

sorted_dates = sorted(all_months.keys())
print(f'processed: {len(sorted_dates)} months')
print(f'avg firms/month: {np.mean([len(m["ids"]) for m in all_months.values()]):.0f}')


train_dates = [d for d in sorted_dates if d <= train_end]
val_dates = [d for d in sorted_dates if train_end < d <= val_end]
test_dates = [d for d in sorted_dates if d > val_end]

print(f'train: {len(train_dates)} months')
print(f'val: {len(val_dates)} months')
print(f'test: {len(test_dates)} months')

x_train = np.vstack([all_months[d]['x'] for d in train_dates])
y_train = np.concatenate([all_months[d]['r'] for d in train_dates])
print(f'x_train: {x_train.shape}')

hpo_dates = train_dates[-n_hpo_months:]
x_hpo = np.vstack([all_months[d]['x'] for d in hpo_dates])
y_hpo = np.concatenate([all_months[d]['r'] for d in hpo_dates])
print(f'x_hpo: {x_hpo.shape}')

trainval_dates = train_dates + val_dates
x_trainval = np.vstack([all_months[d]['x'] for d in trainval_dates])
y_trainval = np.concatenate([all_months[d]['r'] for d in trainval_dates])
print(f'x_trainval: {x_trainval.shape}')


def run_mean_split_simulation(model, month_dates):
    ls_period_rets, ls_period_dates = [], []
    ls_tc_history = []
    lo_period_rets, lo_period_dates = [], []
    lo_tc_history = []

    # state for drift-based L1 turnover accounting per leg
    prev_long_ids = None
    prev_long_w = None
    prev_long_realised = None
    prev_short_ids = None
    prev_short_w = None
    prev_short_realised = None
    prev_lo_ids = None
    prev_lo_w = None
    prev_lo_realised = None

    ls_holdings, lo_holdings = [], []
    rb_counter = -1

    for pos, eom in enumerate(month_dates):
        if pos % rebalance_freq != 0:
            continue
        if eom not in all_months:
            continue
        m = all_months[eom]
        ids = m['ids']
        r = m['r']
        x = m['x']

        n_firms = len(ids)
        if n_firms < min_stocks:
            continue

        pred = model.predict(x)
        valid_pred = np.isfinite(pred)
        if valid_pred.sum() < min_stocks:
            continue

        valid_ret = np.isfinite(r)

        legs = mean_split_legs(pred, valid_pred, valid_ret, min_leg_stocks)
        lo_idx = long_only_leg(valid_pred, valid_ret, min_leg_stocks)
        if legs is None or lo_idx is None:
            continue
        mean_score, long_idx, short_idx = legs
        rb_counter += 1

        long_firm_ids = ids[long_idx]
        short_firm_ids = ids[short_idx]

        # long_idx and short_idx are already restricted to firms with a
        # valid prediction and a valid realised return, so the capped
        # softmax weights below sum to one over exactly the firms actually
        # held, with no separate renormalisation step needed that could
        # otherwise reintroduce a position above max_position_weight
        long_w = capped_softmax_weights(pred[long_idx] - mean_score, max_position_weight)
        short_w = capped_softmax_weights(mean_score - pred[short_idx], max_position_weight)

        long_ids_list = long_firm_ids.tolist()
        short_ids_list = short_firm_ids.tolist()
        long_realised = {int(ids[fi]): float(r[fi]) for fi in long_idx}
        short_realised = {int(ids[fi]): float(r[fi]) for fi in short_idx}
        long_ret = float(np.dot(long_w, r[long_idx]))
        short_ret = float(np.dot(short_w, r[short_idx]))
        ls_ret = long_ret - short_ret

        # drift previous leg weights and compute L1 turnover against them
        d_long_ids, d_long_w = drift_weights(prev_long_ids, prev_long_w, prev_long_realised)
        d_short_ids, d_short_w = drift_weights(prev_short_ids, prev_short_w, prev_short_realised)
        lt = weight_l1_turnover(d_long_ids, d_long_w, long_ids_list, long_w)
        st = weight_l1_turnover(d_short_ids, d_short_w, short_ids_list, short_w)
        ls_flat_tc = (lt + st) * tc_bps / 10000.0

        ls_period_rets.append(ls_ret)
        ls_period_dates.append(eom)
        ls_tc_history.append(ls_flat_tc)
        prev_long_ids = long_ids_list
        prev_long_w = long_w
        prev_long_realised = long_realised
        prev_short_ids = short_ids_list
        prev_short_w = short_w
        prev_short_realised = short_realised

        # long only leg construction, restricted to firms with a valid
        # prediction and a valid realised return
        lo_firm_ids = ids[lo_idx]
        lo_w = capped_softmax_weights(pred[lo_idx], max_position_weight)
        lo_ids_list = lo_firm_ids.tolist()
        lo_realised = {int(ids[fi]): float(r[fi]) for fi in lo_idx}
        lo_ret = float(np.dot(lo_w, r[lo_idx]))

        d_lo_ids, d_lo_w = drift_weights(prev_lo_ids, prev_lo_w, prev_lo_realised)
        lo_turn = weight_l1_turnover(d_lo_ids, d_lo_w, lo_ids_list, lo_w)
        lo_flat_tc = lo_turn * tc_bps / 10000.0

        lo_period_rets.append(lo_ret)
        lo_period_dates.append(eom)
        lo_tc_history.append(lo_flat_tc)
        prev_lo_ids = lo_ids_list
        prev_lo_w = lo_w
        prev_lo_realised = lo_realised

        # record holdings
        for i, fi in enumerate(long_idx):
            ls_holdings.append({
                'rebalance_index': rb_counter, 'eom': eom, 'leg': 'long',
                'id': int(ids[fi]), 'weight': float(long_w[i]),
                'realised_return': float(r[fi]),
            })
        for i, fi in enumerate(short_idx):
            ls_holdings.append({
                'rebalance_index': rb_counter, 'eom': eom, 'leg': 'short',
                'id': int(ids[fi]), 'weight': float(-short_w[i]),
                'realised_return': float(r[fi]),
            })
        for i, fi in enumerate(lo_idx):
            lo_holdings.append({
                'rebalance_index': rb_counter, 'eom': eom, 'leg': 'long',
                'id': int(ids[fi]), 'weight': float(lo_w[i]),
                'realised_return': float(r[fi]),
            })

    return {
        'long_short': {
            'returns': np.array(ls_period_rets),
            'tc': np.array(ls_tc_history),
            'dates': ls_period_dates,
            'holdings_df': pd.DataFrame(ls_holdings),
        },
        'long_only': {
            'returns': np.array(lo_period_rets),
            'tc': np.array(lo_tc_history),
            'dates': lo_period_dates,
            'holdings_df': pd.DataFrame(lo_holdings),
        },
    }


## Hyperparameter search

periods_per_year = 12.0 / rebalance_freq
n_vol_periods = vol_lookback_periods


def _trial_oom(exc):
    s = str(exc).lower()
    return 'out of memory' in s or 'cudaerrormemoryallocation' in s


def _eval_hpo_sharpe(model):
    sim = run_mean_split_simulation(model, val_dates)
    ls = sim['long_short']
    lo = sim['long_only']
    if len(ls['returns']) == 0:
        return -999.0, -999.0
    ls_scaled, _, _ = apply_overlay_and_costs(
        ls['returns'], ls['tc'], target_vol, n_vol_periods, periods_per_year, max_leverage_long_short,
    )
    lo_scaled, _, _ = apply_overlay_and_costs(
        lo['returns'], lo['tc'], target_vol, n_vol_periods, periods_per_year, max_leverage_long_only,
    )
    ls_sharpe = portfolio_metrics(ls_scaled, periods_per_year).get('sharpe', -999.0)
    lo_sharpe = portfolio_metrics(lo_scaled, periods_per_year).get('sharpe', -999.0)
    # a degenerate near zero volatility validation window now returns nan
    # rather than an inflated sharpe. nan is correct for final reporting but
    # must not reach optuna directly, since optuna marks a nan objective as
    # a failed trial rather than a low scored one, silently discarding the
    # trial's information from the search
    if not np.isfinite(ls_sharpe):
        ls_sharpe = -999.0
    if not np.isfinite(lo_sharpe):
        lo_sharpe = -999.0
    return float(ls_sharpe), float(lo_sharpe)



# xgboost hyperparameter search

xgb_best_params_path = results_dir / 'xgb_best_params.json'
xgb_study_path = results_dir / 'xgb_optuna_study.pkl'
xgb_trials_path = results_dir / 'xgb_optuna_trials.csv'

if xgb_best_params_path.exists():
    with open(xgb_best_params_path) as fh:
        cached = json.load(fh)
    xgb_best = cached['best_params']
    xgb_best_value = cached['best_value']
    xgb_hpo_time = cached['hpo_time_seconds']
    if xgb_study_path.exists():
        with open(xgb_study_path, 'rb') as fh:
            xgb_study = pickle.load(fh)
    else:
        xgb_study = None
    print(f'XGBoost hyperparameters already tuned, loaded from {xgb_best_params_path.name}')
    print(f'XGBoost best val ls sharpe: {xgb_best_value:.4f}')
    print(f'XGBoost best params: {xgb_best}')
else:
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log = True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log = True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log = True),
            'random_state': optuna_seed,
            'n_jobs': -1,
            'verbosity': 0,
            **xgb_device_params,
        }
        model = xgb.XGBRegressor(**params)
        try:
            try:
                model.fit(x_hpo, y_hpo)
            except Exception as exc:
                if _trial_oom(exc):
                    del model
                    _gpu_cleanup()
                    raise optuna.exceptions.TrialPruned()
                raise
            ls_sharpe, lo_sharpe = _eval_hpo_sharpe(model)
            trial.set_user_attr('val_sharpe_long_only', lo_sharpe)
            return ls_sharpe
        finally:
            del model                                                   #type: ignore
            _gpu_cleanup()

    xgb_study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = optuna_seed),
    )
    t0 = time.time()
    xgb_study.optimize(xgb_objective, n_trials = n_trials_xgb, show_progress_bar = True)
    xgb_hpo_time = time.time() - t0
    xgb_best = xgb_study.best_params
    xgb_best_value = float(xgb_study.best_value)

    with open(xgb_best_params_path, 'w') as fh:
        json.dump({
            'construction': 'mean_split_softmax_cap_6m',
            'best_params': xgb_best,
            'best_value': xgb_best_value,
            'best_trial_number': int(xgb_study.best_trial.number),
            'best_trial_user_attrs': dict(xgb_study.best_trial.user_attrs),
            'n_trials_completed': sum(1 for t in xgb_study.trials if t.state.name == 'COMPLETE'),
            'hpo_time_seconds': float(xgb_hpo_time),
        }, fh, indent = 2, default = float)

    xgb_trials_df = xgb_study.trials_dataframe()
    xgb_trials_df.to_csv(xgb_trials_path, index = False)
    with open(xgb_study_path, 'wb') as fh:
        pickle.dump(xgb_study, fh)

    print(f'XGBoost best val ls sharpe: {xgb_best_value:.4f}')
    print(f'XGBoost best params: {xgb_best}')
    print(f'XGBoost hpo time: {xgb_hpo_time:.1f} s, {xgb_hpo_time / 60:.2f} min')


# lightgbm hyperparameter search

lgb_best_params_path = results_dir / 'lgb_best_params.json'
lgb_study_path = results_dir / 'lgb_optuna_study.pkl'
lgb_trials_path = results_dir / 'lgb_optuna_trials.csv'

if lgb_best_params_path.exists():
    with open(lgb_best_params_path) as fh:
        cached = json.load(fh)
    lgb_best = cached['best_params']
    lgb_best_value = cached['best_value']
    lgb_hpo_time = cached['hpo_time_seconds']
    if lgb_study_path.exists():
        with open(lgb_study_path, 'rb') as fh:
            lgb_study = pickle.load(fh)
    else:
        lgb_study = None
    print(f'LightGBM hyperparameters already tuned, loaded from {lgb_best_params_path.name}')
    print(f'LightGBM best val ls sharpe: {lgb_best_value:.4f}')
    print(f'LightGBM best params: {lgb_best}')
else:
    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log = True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log = True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log = True),
            'random_state': optuna_seed,
            'n_jobs': -1,
            'verbose': -1,
            **lgb_device_params,
        }
        model = lgb.LGBMRegressor(**params)
        try:
            try:
                model.fit(x_hpo, y_hpo)
            except Exception as exc:
                if _trial_oom(exc):
                    del model
                    _gpu_cleanup()
                    raise optuna.exceptions.TrialPruned()
                raise
            ls_sharpe, lo_sharpe = _eval_hpo_sharpe(model)
            trial.set_user_attr('val_sharpe_long_only', lo_sharpe)
            return ls_sharpe
        finally:
            del model                                                    #type: ignore 
            _gpu_cleanup()

    lgb_study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = optuna_seed),
    )
    t0 = time.time()
    lgb_study.optimize(lgb_objective, n_trials = n_trials_lgb, show_progress_bar = True)
    lgb_hpo_time = time.time() - t0
    lgb_best = lgb_study.best_params
    lgb_best_value = float(lgb_study.best_value)

    with open(lgb_best_params_path, 'w') as fh:
        json.dump({
            'construction': 'mean_split_softmax_cap_6m',
            'best_params': lgb_best,
            'best_value': lgb_best_value,
            'best_trial_number': int(lgb_study.best_trial.number),
            'best_trial_user_attrs': dict(lgb_study.best_trial.user_attrs),
            'n_trials_completed': sum(1 for t in lgb_study.trials if t.state.name == 'COMPLETE'),
            'hpo_time_seconds': float(lgb_hpo_time),
        }, fh, indent = 2, default = float)

    lgb_trials_df = lgb_study.trials_dataframe()
    lgb_trials_df.to_csv(lgb_trials_path, index = False)
    with open(lgb_study_path, 'wb') as fh:
        pickle.dump(lgb_study, fh)

    print(f'LightGBM best val ls sharpe: {lgb_best_value:.4f}')
    print(f'LightGBM best params: {lgb_best}')
    print(f'LightGBM hpo time: {lgb_hpo_time:.1f} s, {lgb_hpo_time / 60:.2f} min')



_gpu_cleanup()


class XGBPredictor:
    """Thin wrapper exposing a sklearn style predict, save_model, and
    feature_importances_ on top of a native xgboost Booster."""
    def __init__(self, booster):
        self.booster = booster

    def predict(self, x):
        if not isinstance(x, xgb.DMatrix):
            x = xgb.DMatrix(x)
        return self.booster.predict(x)

    def save_model(self, path):
        self.booster.save_model(path)

    @property
    def feature_importances_(self):
        score = self.booster.get_score(importance_type = 'gain')
        imp = np.zeros(len(feature_cols), dtype = np.float64)
        for k, v in score.items():
            try:
                idx = int(k.lstrip('f'))
                if 0 <= idx < len(imp):
                    imp[idx] = v
            except ValueError:
                continue
        return imp


class ChunkIter(xgb.DataIter):
    """Hand the training matrix to xgboost in chunks. xgboost stores the
    quantised form of each chunk on disk and discards the raw chunk
    after consumption, so the peak memory footprint is bounded by the
    chunk size rather than by the full matrix."""
    def __init__(self, x, y, chunk_size, cache_prefix):
        self.x = x
        self.y = y
        self.chunk_size = chunk_size
        self.n_chunks = (len(x) + chunk_size - 1) // chunk_size
        self.current = 0
        super().__init__(cache_prefix = cache_prefix)

    def next(self, input_data):
        if self.current >= self.n_chunks:
            return False
        start = self.current * self.chunk_size
        end = min(start + self.chunk_size, len(self.x))
        input_data(data = self.x[start:end], label = self.y[start:end])
        self.current += 1
        return True

    def reset(self):
        self.current = 0


# build the booster parameter dictionary. The CPU device assignment is
# deliberate and required by the external memory mode
xgb_final_params = dict(xgb_best)
xgb_final_params.update({
    'tree_method': 'hist',
    'device': 'cpu',
    'max_bin': 64,
    'objective': 'reg:squarederror',
    'seed': optuna_seed,
    'nthread': -1,
})

n_estimators = int(xgb_final_params.pop('n_estimators', 100))

cache_dir = results_dir / 'xgb_cache'
cache_dir.mkdir(parents = True, exist_ok = True)

print('building xgboost external memory training matrix (train + val)')
it = ChunkIter(
    x = x_trainval,
    y = y_trainval,
    chunk_size = 200000,
    cache_prefix = str(cache_dir / 'iter'),
)
dtrain = xgb.ExtMemQuantileDMatrix(it, max_bin = 64)
print(f'matrix built: {dtrain.num_row():,} rows, {dtrain.num_col()} columns')

del it
gc.collect()

t0 = time.time()
xgb_booster = xgb.train(
    params = xgb_final_params,
    dtrain = dtrain,
    num_boost_round = n_estimators,
    verbose_eval = False,
)
xgb_train_time = time.time() - t0
xgb_model = XGBPredictor(xgb_booster)

del dtrain
gc.collect()
print(f'XGBoost final model trained in {xgb_train_time:.1f} s')


_gpu_cleanup()


lgb_final_params = dict(lgb_best)
lgb_final_params['max_bin'] = 128
lgb_final_params['device'] = 'cpu'

lgb_model = lgb.LGBMRegressor(
    **lgb_final_params,
    random_state = optuna_seed,
    bagging_seed = optuna_seed,
    feature_fraction_seed = optuna_seed,
    data_random_seed = optuna_seed,
    deterministic = True,
    force_row_wise = True,
    n_jobs = -1,
    verbose = -1,
)
t0 = time.time()
lgb_model.fit(x_trainval, y_trainval)
lgb_train_time = time.time() - t0
print(f'LightGBM final model trained in {lgb_train_time:.1f} s')


_gpu_cleanup()


xgb_model.save_model(str(results_dir / 'xgb_model.json'))
lgb_model.booster_.save_model(str(results_dir / 'lgb_model.txt'))
print('models saved in native formats')



xgb_rc_val = rank_correlation_oos(xgb_model.predict, val_dates, all_months)
xgb_rc_test = rank_correlation_oos(xgb_model.predict, test_dates, all_months)
lgb_rc_val = rank_correlation_oos(lgb_model.predict, val_dates, all_months)
lgb_rc_test = rank_correlation_oos(lgb_model.predict, test_dates, all_months)

print(f'XGBoost rank corr: val = {xgb_rc_val:.4f}, test = {xgb_rc_test:.4f}')
print(f'LightGBM rank corr: val = {lgb_rc_val:.4f}, test = {lgb_rc_test:.4f}')



def evaluate_and_save(model, name):
    sim = run_mean_split_simulation(model, sorted_dates)
    ls = sim['long_short']
    lo = sim['long_only']

    ls_scaled_full, ls_unscaled_full, ls_lev = apply_overlay_and_costs(
        ls['returns'], ls['tc'], target_vol, n_vol_periods, periods_per_year, max_leverage_long_short,
    )
    lo_scaled_full, lo_unscaled_full, lo_lev = apply_overlay_and_costs(
        lo['returns'], lo['tc'], target_vol, n_vol_periods, periods_per_year, max_leverage_long_only,
    )

    test_set = set(test_dates)
    ls_mask = np.array([d in test_set for d in ls['dates']])
    lo_mask = np.array([d in test_set for d in lo['dates']])

    ls_raw_test = ls_unscaled_full[ls_mask]
    ls_scaled_test = ls_scaled_full[ls_mask]
    lo_raw_test = lo_unscaled_full[lo_mask]
    lo_scaled_test = lo_scaled_full[lo_mask]

    # test window rebalance dates, used both for the returns dataframe and
    # the per year breakdown inside portfolio_metrics
    ls_dates_test = [d for d, m in zip(ls['dates'], ls_mask) if m]
    lo_dates_test = [d for d, m in zip(lo['dates'], lo_mask) if m]

    ls_ret_df = pd.DataFrame({
        'eom': ls_dates_test,
        'return_unscaled': ls_raw_test,
        'return_scaled': ls_scaled_test,
        'leverage': ls_lev[ls_mask],
    })
    lo_ret_df = pd.DataFrame({
        'eom': lo_dates_test,
        'return_unscaled': lo_raw_test,
        'return_scaled': lo_scaled_test,
        'leverage': lo_lev[lo_mask],
    })

    ls_hold_df = ls['holdings_df'][ls['holdings_df']['eom'].isin(test_set)].copy().reset_index(drop = True)
    lo_hold_df = lo['holdings_df'][lo['holdings_df']['eom'].isin(test_set)].copy().reset_index(drop = True)

    m_ls_raw = portfolio_metrics(ls_raw_test, periods_per_year, dates = ls_dates_test)
    m_ls_scaled = portfolio_metrics(ls_scaled_test, periods_per_year, dates = ls_dates_test)
    m_lo_raw = portfolio_metrics(lo_raw_test, periods_per_year, dates = lo_dates_test)
    m_lo_scaled = portfolio_metrics(lo_scaled_test, periods_per_year, dates = lo_dates_test)

    ls_ret_df.to_csv(results_dir / f'{name}_returns_long_short.csv', index = False)
    lo_ret_df.to_csv(results_dir / f'{name}_returns_long_only.csv', index = False)
    ls_hold_df.to_csv(results_dir / f'{name}_holdings_long_short.csv', index = False)
    lo_hold_df.to_csv(results_dir / f'{name}_holdings_long_only.csv', index = False)

    predict_at_dates(model.predict, test_dates, all_months).to_csv(
        results_dir / f'{name}_test_predictions.csv', index = False,
    )

    return {
        'returns_ls_raw': ls_raw_test, 'returns_ls_scaled': ls_scaled_test,
        'returns_lo_raw': lo_raw_test, 'returns_lo_scaled': lo_scaled_test,
        'dates_ls': ls_dates_test, 'dates_lo': lo_dates_test,
        'metrics': {
            'long_short_raw': m_ls_raw, 'long_short_scaled': m_ls_scaled,
            'long_only_raw': m_lo_raw, 'long_only_scaled': m_lo_scaled,
        },
    }


xgb_eval = evaluate_and_save(xgb_model, 'xgb')
lgb_eval = evaluate_and_save(lgb_model, 'lgb')

for name, ev in [('XGBoost', xgb_eval), ('LightGBM', lgb_eval)]:
    mls = ev['metrics']['long_short_scaled']
    mlo = ev['metrics']['long_only_scaled']
    print(f'{name} long short (scaled): sharpe = {mls["sharpe"]:.4f}, ann_ret = {mls["ann_ret"] * 100:.2f}%, ann_vol = {mls["ann_vol"] * 100:.2f}%')
    print(f'{name} long only  (scaled): sharpe = {mlo["sharpe"]:.4f}, ann_ret = {mlo["ann_ret"] * 100:.2f}%, ann_vol = {mlo["ann_vol"] * 100:.2f}%')



xgb_imp = pd.DataFrame({'feature': feature_cols,'importance': xgb_model.feature_importances_}).sort_values('importance', ascending = False)
lgb_imp = pd.DataFrame({'feature': feature_cols,'importance': lgb_model.feature_importances_}).sort_values('importance', ascending = False)

xgb_imp.to_csv(results_dir / 'xgb_feature_importance.csv', index = False)
lgb_imp.to_csv(results_dir / 'lgb_feature_importance.csv', index = False)

print('top 10 xgboost features')
print(xgb_imp.head(10).to_string(index = False))
print('top 10 lightgbm features')
print(lgb_imp.head(10).to_string(index = False))



def _round_or_none(x, ndigits):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), ndigits)


def _strip_per_year(m):
    if not isinstance(m, dict):
        return m
    return {k: v for k, v in m.items() if k != 'per_year'}


def _strip_metrics_block(metrics):
    return {k: _strip_per_year(v) for k, v in metrics.items()}


summary = {
    'construction': 'mean_split_softmax_cap_6m',
    'target_column': ret_col,
    'n_features': len(feature_cols),
    'feature_cols': feature_cols,
    'split': {
        'train': {'start': str(train_dates[0].date()), 'end': str(train_dates[-1].date()),
                  'n_months': len(train_dates), 'n_obs': int(x_train.shape[0])},
        'val': {'start': str(val_dates[0].date()), 'end': str(val_dates[-1].date()),
                'n_months': len(val_dates)},
        'test': {'start': str(test_dates[0].date()), 'end': str(test_dates[-1].date()),
                 'n_months': len(test_dates)},
        'hpo': {'start': str(hpo_dates[0].date()), 'end': str(hpo_dates[-1].date()),
                'n_months': len(hpo_dates), 'n_obs': int(x_hpo.shape[0])},
    },
    'config': {
        'rebalance_freq': rebalance_freq,
        'horizon_months': horizon_months,
        'tc_bps': tc_bps,
        'min_stocks': min_stocks,
        'min_leg_stocks': min_leg_stocks,
        'ret_clip': [ret_clip_low, ret_clip_high],
        'target_vol': target_vol,
        'vol_lookback_months': vol_lookback_months,
        'vol_lookback_periods': vol_lookback_periods,
        'n_vol_periods': n_vol_periods,
        'periods_per_year': periods_per_year,
        'max_leverage_long_only': max_leverage_long_only,
        'max_leverage_long_short': max_leverage_long_short,
        'max_position_weight': max_position_weight,
        'optuna_seed': optuna_seed,
        'n_trials_xgb': n_trials_xgb,
        'n_trials_lgb': n_trials_lgb,
    },
    'xgboost': {
        'best_params': xgb_best,
        'best_val_long_short_sharpe': float(xgb_study.best_value) if xgb_study is not None else float(xgb_best_value),
        'rc_val': float(xgb_rc_val), 'rc_test': float(xgb_rc_test),
        'final_training_time_seconds': float(xgb_train_time),
        'portfolio_metrics': _strip_metrics_block(xgb_eval['metrics']),
    },
    'lightgbm': {
        'best_params': lgb_best,
        'best_val_long_short_sharpe': float(lgb_study.best_value) if lgb_study is not None else float(lgb_best_value),
        'rc_val': float(lgb_rc_val), 'rc_test': float(lgb_rc_test),
        'final_training_time_seconds': float(lgb_train_time),
        'portfolio_metrics': _strip_metrics_block(lgb_eval['metrics']),
    },
}

with open(results_dir / 'tree_summary.json', 'w') as gbms:
    json.dump(summary, gbms, indent = 2, default = float)
print(f'summary saved to {results_dir / "tree_summary.json"}')


rows = []
for name, ev, rc in [('xgboost', xgb_eval, xgb_rc_test), ('lightgbm', lgb_eval, lgb_rc_test)]:
    for portfolio, scaling, key in [
        ('long_short', 'unscaled', 'long_short_raw'),
        ('long_short', 'scaled', 'long_short_scaled'),
        ('long_only', 'unscaled', 'long_only_raw'),
        ('long_only', 'scaled', 'long_only_scaled'),
    ]:
        m = ev['metrics'][key]
        rows.append({
            'model': name, 'portfolio': portfolio,
            'scaling': scaling, 'rc_test': round(rc, 4),
            'sharpe': _round_or_none(m['sharpe'], 4),
            'se': _round_or_none(m['se_sharpe'], 4),
            'ann_ret': _round_or_none(m['ann_ret'] * 100, 2),
            'ann_vol': _round_or_none(m['ann_vol'] * 100, 2),
            'cagr': _round_or_none(m['cagr'] * 100, 2),
            'cum_return': _round_or_none(m['cum_return'] * 100, 2),
            'max_dd': _round_or_none(m['max_dd'] * 100, 2),
            'n_obs': m['n_obs'],
        })
summary_table = pd.DataFrame(rows)
print('Tree Benchmark, EM Universe, mean split capped softmax, 6m rebalance')
print(summary_table.to_string(index = False))
summary_table.to_csv(results_dir / 'tree_summary.csv', index = False)
print('summary csv saved')

# per year breakdown across both models. one row per (model, portfolio,
# scaling, year).

per_year_rows = []

def _flush_per_year(model, portfolio, scaling, metrics):
    py = metrics.get('per_year', {}) if isinstance(metrics, dict) else {}
    for year in sorted(py.keys()):
        ym = py[year]
        per_year_rows.append({
            'model': model, 'portfolio': portfolio,
            'scaling': scaling, 'year': int(year),
            'ann_ret': round(float(ym['ann_ret']) * 100, 4),
            'ann_vol': round(float(ym['ann_vol']) * 100, 4),
            'sharpe': (round(float(ym['sharpe']), 4)
                      if not (isinstance(ym['sharpe'], float) and np.isnan(ym['sharpe']))
                      else None),
            'max_dd': round(float(ym['max_dd']) * 100, 4),
            'cum_return': round(float(ym['cum_return']) * 100, 4),
            'n_obs': int(ym['n_obs'])
        })

for name, ev in [('xgboost', xgb_eval), ('lightgbm', lgb_eval)]:
    _flush_per_year(name, 'long_short', 'unscaled', ev['metrics']['long_short_raw'])
    _flush_per_year(name, 'long_short', 'scaled', ev['metrics']['long_short_scaled'])
    _flush_per_year(name, 'long_only', 'unscaled', ev['metrics']['long_only_raw'])
    _flush_per_year(name, 'long_only', 'scaled', ev['metrics']['long_only_scaled'])

per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(results_dir / 'tree_per_year_metrics.csv', index = False)
print(f'per year metrics saved, {len(per_year_df)} rows')
