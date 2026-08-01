# XGBoost and LightGBM Benchmark

import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
import torch

from benchmark_common import (
	BenchmarkConfig, load_universe, make_splits, stack_months,
	validation_sharpes, evaluate_and_save, hpo_run_or_load,
	rank_correlation_oos, round_or_none, strip_per_year, flush_per_year_rows,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


cuda_available = torch.cuda.is_available()
print(f'cuda available (torch): {cuda_available}')
if cuda_available:
	print(f'cuda device: {torch.cuda.get_device_name(0)}')


def _gpu_cleanup():
	gc.collect()
	if cuda_available:
		torch.cuda.empty_cache()


def _probe(fit_fn, label):
	try:
		x = np.random.randn(200, 10).astype(np.float32)
		y = np.random.randn(200).astype(np.float32)
		fit_fn(x, y)
		return True
	except Exception as exc:
		print(f'{label}: gpu probe failed ({type(exc).__name__}: {exc})')
		return False


xgb_use_cuda = False
lgb_use_gpu = False
if cuda_available:
	xgb_use_cuda = _probe(lambda x, y: xgb.XGBRegressor(n_estimators = 5, tree_method = 'hist', device = 'cuda', verbosity = 0).fit(x, y), 'xgboost')
	lgb_use_gpu = _probe(lambda x, y: lgb.LGBMRegressor(n_estimators = 5, device = 'gpu', verbose = -1).fit(x, y), 'lightgbm')

xgb_device_params = {'tree_method': 'hist', 'device': 'cuda'} if xgb_use_cuda else {'tree_method': 'hist'}
lgb_device_params = {'device': 'gpu'} if lgb_use_gpu else {}


data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/tree_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

train_end = pd.Timestamp('2015-12-31')
val_end = pd.Timestamp('2020-12-31')
ret_col_1m = 'ret_exc_lead1m'
ret_col = 'ret_exc_lead6m'

cfg = BenchmarkConfig()

n_trials_xgb = 50
n_trials_lgb = 50
optuna_seed = 24
n_hpo_months = 36


feature_cols, all_months, sorted_dates = load_universe(data_path, ret_col_1m, ret_col, cfg, train_end)
train_dates, val_dates, test_dates = make_splits(sorted_dates, train_end, val_end)
print(f'train: {len(train_dates)} months, val: {len(val_dates)} months, test: {len(test_dates)} months')

x_train, y_train = stack_months(all_months, train_dates)
print(f'x_train: {x_train.shape}')

hpo_dates = train_dates[-n_hpo_months:]
x_hpo, y_hpo = stack_months(all_months, hpo_dates)
print(f'x_hpo: {x_hpo.shape}')

trainval_dates = train_dates + val_dates
x_trainval, y_trainval = stack_months(all_months, trainval_dates)
print(f'x_trainval: {x_trainval.shape}')


def _trial_oom(exc):
	s = str(exc).lower()
	return 'out of memory' in s or 'cudaerrormemoryallocation' in s


# xgboost hyperparameter search

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
		ls_sharpe, lo_sharpe = validation_sharpes(model.predict, val_dates, all_months, cfg)
		trial.set_user_attr('val_sharpe_long_only', lo_sharpe)
		return ls_sharpe
	finally:
		del model                               #type:ignore
		_gpu_cleanup()


xgb_cached, xgb_study = hpo_run_or_load(results_dir, 'xgb', xgb_objective, n_trials_xgb, optuna_seed)
xgb_best = xgb_cached['best_params']
xgb_best_value = xgb_cached['best_value']


# lightgbm hyperparameter search

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
		ls_sharpe, lo_sharpe = validation_sharpes(model.predict, val_dates, all_months, cfg)
		trial.set_user_attr('val_sharpe_long_only', lo_sharpe)
		return ls_sharpe
	finally:
		del model                                   #type:ignore
		_gpu_cleanup()


lgb_cached, lgb_study = hpo_run_or_load(results_dir, 'lgb', lgb_objective, n_trials_lgb, optuna_seed)
lgb_best = lgb_cached['best_params']
lgb_best_value = lgb_cached['best_value']

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
			idx = int(k.lstrip('f'))
			if 0 <= idx < len(imp):
				imp[idx] = v
		return imp


class ChunkIter(xgb.DataIter):
	"""Hand the training matrix to xgboost in chunks so the quantised form
	of each chunk is written to disk and the raw chunk discarded after
	consumption, bounding peak memory to the chunk size rather than the
	full matrix."""

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


# final xgboost training on train + val, via external memory since the
# cpu device is required by this mode
xgb_final_params = dict(xgb_best)
xgb_final_params.update({
	'tree_method': 'hist', 'device': 'cpu', 'max_bin': 64,
	'objective': 'reg:squarederror', 'seed': optuna_seed, 'nthread': -1,
})
n_estimators = int(xgb_final_params.pop('n_estimators', 100))

cache_dir = results_dir / 'xgb_cache'
cache_dir.mkdir(parents = True, exist_ok = True)

print('building xgboost external memory training matrix (train + val)')
it = ChunkIter(x = x_trainval, y = y_trainval, chunk_size = 200000, cache_prefix = str(cache_dir / 'iter'))
dtrain = xgb.ExtMemQuantileDMatrix(it, max_bin = 64)
print(f'matrix built: {dtrain.num_row():,} rows, {dtrain.num_col()} columns')
del it
gc.collect()

t0 = time.time()
xgb_booster = xgb.train(params = xgb_final_params, dtrain = dtrain, num_boost_round = n_estimators, verbose_eval = False)
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
	random_state = optuna_seed, bagging_seed = optuna_seed,
	feature_fraction_seed = optuna_seed, data_random_seed = optuna_seed,
	deterministic = True, force_row_wise = True, n_jobs = -1, verbose = -1,
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


xgb_eval = evaluate_and_save(xgb_model.predict, 'xgb', all_months, sorted_dates, test_dates, results_dir, cfg)
lgb_eval = evaluate_and_save(lgb_model.predict, 'lgb', all_months, sorted_dates, test_dates, results_dir, cfg)

for name, ev in [('XGBoost', xgb_eval), ('LightGBM', lgb_eval)]:
	mls = ev['metrics']['long_short_scaled']
	mlo = ev['metrics']['long_only_scaled']
	print(f'{name} long short (scaled): sharpe = {mls["sharpe"]:.4f}, ann_ret = {mls["ann_ret"] * 100:.2f}%, ann_vol = {mls["ann_vol"] * 100:.2f}%')
	print(f'{name} long only  (scaled): sharpe = {mlo["sharpe"]:.4f}, ann_ret = {mlo["ann_ret"] * 100:.2f}%, ann_vol = {mlo["ann_vol"] * 100:.2f}%')


xgb_imp = pd.DataFrame({'feature': feature_cols, 'importance': xgb_model.feature_importances_}).sort_values('importance', ascending = False)
lgb_imp = pd.DataFrame({'feature': feature_cols, 'importance': lgb_model.feature_importances_}).sort_values('importance', ascending = False)
xgb_imp.to_csv(results_dir / 'xgb_feature_importance.csv', index = False)
lgb_imp.to_csv(results_dir / 'lgb_feature_importance.csv', index = False)

print('top 10 xgboost features')
print(xgb_imp.head(10).to_string(index = False))
print('top 10 lightgbm features')
print(lgb_imp.head(10).to_string(index = False))


summary = {
	'construction': 'mean_split_softmax_cap_6m',
	'target_column': ret_col,
	'n_features': len(feature_cols),
	'feature_cols': feature_cols,
	'split': {
		'train': {'start': str(train_dates[0].date()), 'end': str(train_dates[-1].date()), 'n_months': len(train_dates), 'n_obs': int(x_train.shape[0])},
		'val': {'start': str(val_dates[0].date()), 'end': str(val_dates[-1].date()), 'n_months': len(val_dates)},
		'test': {'start': str(test_dates[0].date()), 'end': str(test_dates[-1].date()), 'n_months': len(test_dates)},
		'hpo': {'start': str(hpo_dates[0].date()), 'end': str(hpo_dates[-1].date()), 'n_months': len(hpo_dates), 'n_obs': int(x_hpo.shape[0])},
	},
	'config': {
		'rebalance_freq': cfg.rebalance_freq, 'horizon_months': cfg.horizon_months,
		'tc_bps': cfg.tc_bps, 'min_stocks': cfg.min_stocks, 'min_leg_stocks': cfg.min_leg_stocks,
		'ret_clip': [cfg.ret_clip_low, cfg.ret_clip_high], 'target_vol': cfg.target_vol,
		'vol_lookback_months': cfg.vol_lookback_months, 'vol_lookback_periods': cfg.vol_lookback_periods,
		'n_vol_periods': cfg.n_vol_periods, 'periods_per_year': cfg.periods_per_year,
		'max_leverage_long_only': cfg.max_leverage_long_only, 'max_leverage_long_short': cfg.max_leverage_long_short,
		'max_position_weight': cfg.max_position_weight, 'optuna_seed': optuna_seed,
		'n_trials_xgb': n_trials_xgb, 'n_trials_lgb': n_trials_lgb,
	},
	'xgboost': {
		'best_params': xgb_best, 'best_val_long_short_sharpe': xgb_best_value,
		'rc_val': float(xgb_rc_val), 'rc_test': float(xgb_rc_test),
		'final_training_time_seconds': float(xgb_train_time),
		'portfolio_metrics': {k: strip_per_year(v) for k, v in xgb_eval['metrics'].items()},
	},
	'lightgbm': {
		'best_params': lgb_best, 'best_val_long_short_sharpe': lgb_best_value,
		'rc_val': float(lgb_rc_val), 'rc_test': float(lgb_rc_test),
		'final_training_time_seconds': float(lgb_train_time),
		'portfolio_metrics': {k: strip_per_year(v) for k, v in lgb_eval['metrics'].items()},
	},
}

with open(results_dir / 'tree_summary.json', 'w') as fh:
	json.dump(summary, fh, indent = 2, default = float)
print(f'summary saved to {results_dir / "tree_summary.json"}')


rows = []
for name, ev, rc in [('xgboost', xgb_eval, xgb_rc_test), ('lightgbm', lgb_eval, lgb_rc_test)]:
	for portfolio, scaling, key in [
		('long_short', 'unscaled', 'long_short_unscaled'), ('long_short', 'scaled', 'long_short_scaled'),
		('long_only', 'unscaled', 'long_only_unscaled'), ('long_only', 'scaled', 'long_only_scaled'),
	]:
		m = ev['metrics'][key]
		rows.append({
			'model': name, 'portfolio': portfolio, 'scaling': scaling, 'rc_test': round(rc, 4),
			'sharpe': round_or_none(m['sharpe'], 4), 'se': round_or_none(m['se_sharpe'], 4),
			'ann_ret': round_or_none(m['ann_ret'] * 100, 2), 'ann_vol': round_or_none(m['ann_vol'] * 100, 2),
			'cagr': round_or_none(m['cagr'] * 100, 2), 'cum_return': round_or_none(m['cum_return'] * 100, 2),
			'max_dd': round_or_none(m['max_dd'] * 100, 2), 'n_obs': m['n_obs'],
		})

summary_table = pd.DataFrame(rows)
print('Tree Benchmark, EM Universe, mean split capped softmax, 6m rebalance')
print(summary_table.to_string(index = False))
summary_table.to_csv(results_dir / 'tree_summary.csv', index = False)
print('summary csv saved')


per_year_rows = []
for name, ev in [('xgboost', xgb_eval), ('lightgbm', lgb_eval)]:
	flush_per_year_rows(per_year_rows, name, 'long_short', 'unscaled', ev['metrics']['long_short_unscaled'])
	flush_per_year_rows(per_year_rows, name, 'long_short', 'scaled', ev['metrics']['long_short_scaled'])
	flush_per_year_rows(per_year_rows, name, 'long_only', 'unscaled', ev['metrics']['long_only_unscaled'])
	flush_per_year_rows(per_year_rows, name, 'long_only', 'scaled', ev['metrics']['long_only_scaled'])

per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(results_dir / 'tree_per_year_metrics.csv', index = False)
print(f'per year metrics saved, {len(per_year_df)} rows')
