import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from safetensors.torch import save_file as safetensors_save

from benchmark_common import (
	BenchmarkConfig, load_universe, make_splits, stack_months,
	validation_sharpes, evaluate_and_save, hpo_run_or_load,
	run_mean_split_simulation_monthly, apply_vol_target_monthly,
	build_diagnostic_rows, portfolio_metrics, rank_correlation_oos,
	round_or_none, strip_per_year, flush_per_year_rows,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
print(f'cuda available: {cuda_available}, device: {device}')
if cuda_available:
	print(f'device name: {torch.cuda.get_device_name(0)}')


data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/mlp_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

train_end = pd.Timestamp('2015-12-31')
val_end = pd.Timestamp('2020-12-31')
ret_col_1m = 'ret_exc_lead1m'
ret_col = 'ret_exc_lead6m'

cfg = BenchmarkConfig()

n_epochs_hpo = 100
patience = 10
grad_clip_norm = 1.0
n_trials = 30
optuna_seed = 24
torch_seed = 24


feature_cols, all_months, sorted_dates = load_universe(data_path, ret_col_1m, ret_col, cfg, store_r1m = True)
train_dates, val_dates, test_dates = make_splits(sorted_dates, train_end, val_end)
print(f'train: {len(train_dates)} months, val: {len(val_dates)} months, test: {len(test_dates)} months')

x_train, y_train = stack_months(all_months, train_dates)
y_train = y_train.astype(np.float32)
print(f'x_train: {x_train.shape}')


# model

class MLP(nn.Module):
	def __init__(self, n_features, d_model, dropout):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_features, d_model), nn.ELU(), nn.Dropout(dropout),
			nn.Linear(d_model, d_model), nn.ELU(), nn.Dropout(dropout),
			nn.Linear(d_model, d_model), nn.ELU(), nn.Dropout(dropout),
			nn.Linear(d_model, 1),
		)

	def forward(self, x):
		return self.net(x).squeeze(-1)


class MLPPredictor:
	def __init__(self, model, dev):
		self.model = model
		self.dev = dev

	def predict(self, x):
		self.model.eval()
		with torch.no_grad():
			x_t = torch.from_numpy(x).float().to(self.dev)
			return self.model(x_t).cpu().numpy()


def train_mlp(params, x_pool, y_pool, val_dates_local, n_epochs, patience_val, dev, seed, early_stop = True):
	torch.manual_seed(seed)
	np.random.seed(seed)
	if dev.type == 'cuda':
		torch.cuda.manual_seed_all(seed)

	model = MLP(n_features = x_pool.shape[1], d_model = params['d_model'], dropout = params['dropout']).to(dev)
	optimizer = torch.optim.AdamW(model.parameters(), lr = params['learning_rate'], weight_decay = params['weight_decay'])
	criterion = nn.MSELoss()

	x_t = torch.from_numpy(x_pool).float().to(dev)
	y_t = torch.from_numpy(y_pool).float().to(dev)
	n_total = len(x_t)
	batch_size = params['batch_size']
	predictor = MLPPredictor(model, dev)

	best_rc = -np.inf
	best_state = None
	best_epoch = 0
	patience_ctr = 0
	train_losses, val_rank_corrs = [], []

	for epoch in range(n_epochs):
		model.train()
		perm = torch.randperm(n_total, device = dev)
		epoch_loss = 0.0
		n_batches = 0
		for i in range(0, n_total, batch_size):
			idx = perm[i:i + batch_size]
			pred = model(x_t[idx])
			loss = criterion(pred, y_t[idx])
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip_norm)
			optimizer.step()
			epoch_loss += float(loss.item())
			n_batches += 1

		train_losses.append(epoch_loss / max(n_batches, 1))

		if early_stop:
			val_rc = rank_correlation_oos(predictor.predict, val_dates_local, all_months)
			val_rank_corrs.append(val_rc)
			if val_rc > best_rc:
				best_rc = val_rc
				best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
				best_epoch = epoch
				patience_ctr = 0
			else:
				patience_ctr += 1
				if patience_ctr >= patience_val:
					break
		else:
			best_epoch = epoch

	if early_stop and best_state is not None:
		model.load_state_dict(best_state)

	return model, {
		'train_losses': train_losses, 'val_rank_corrs': val_rank_corrs,
		'best_epoch': best_epoch, 'best_val_rc': float(best_rc) if early_stop else float('nan'),
		'n_epochs_run': len(train_losses),
	}


# hyperparameter search. the objective is the validation long short sharpe
# under the mean split capped softmax construction with the volatility
# overlay applied, matching the construction used for the other benchmarks

def mlp_objective(trial):
	params = {
		'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
		'dropout': trial.suggest_float('dropout', 0.0, 0.5),
		'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log = True),
		'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log = True),
		'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
	}
	model, log = train_mlp(params, x_train, y_train, val_dates, n_epochs_hpo, patience, device, torch_seed, early_stop = True)
	predictor = MLPPredictor(model, device)
	ls_sharpe, lo_sharpe = validation_sharpes(predictor.predict, val_dates, all_months, cfg)

	trial.set_user_attr('best_epoch', int(log['best_epoch']))
	trial.set_user_attr('n_epochs_run', int(log['n_epochs_run']))
	trial.set_user_attr('best_val_rc', float(log['best_val_rc']))
	trial.set_user_attr('val_sharpe_long_only', float(lo_sharpe))
	return ls_sharpe


mlp_cached, mlp_study = hpo_run_or_load(
	results_dir, 'mlp', mlp_objective, n_trials, optuna_seed,
	extra_fields = lambda study: {'best_epoch': int(study.best_trial.user_attrs.get('best_epoch', n_epochs_hpo - 1))},
)
mlp_best = mlp_cached['best_params']
mlp_best_value = mlp_cached['best_value']
mlp_best_epoch = int(mlp_cached['best_epoch'])
mlp_hpo_time = mlp_cached['hpo_time_seconds']
print(f'mlp best epoch: {mlp_best_epoch}')


# final training on training data only, for best_epoch + 1 epochs with no
# early stopping. the validation set was already used to pick best_epoch
n_final_epochs = mlp_best_epoch + 1

t0 = time.time()
mlp_model, mlp_log = train_mlp(mlp_best, x_train, y_train, None, n_final_epochs, patience, device, torch_seed, early_stop = False)
mlp_train_time = time.time() - t0
mlp_predictor = MLPPredictor(mlp_model, device)
n_params = sum(p.numel() for p in mlp_model.parameters())

print(f'mlp final model trained in {mlp_train_time:.1f} s, {n_final_epochs} epochs')
print(f'parameter count: {n_params:,}')

safetensors_save(mlp_model.state_dict(), str(results_dir / 'mlp_weights.safetensors'))

with open(results_dir / 'mlp_train_log.json', 'w') as fh:
	json.dump({
		'train_losses': mlp_log['train_losses'],
		'best_epoch_from_hpo': mlp_best_epoch,
		'n_final_epochs': n_final_epochs,
		'training_time_seconds': float(mlp_train_time),
		'parameter_count': int(n_params),
	}, fh, indent = 2, default = float)


mlp_rc_val = rank_correlation_oos(mlp_predictor.predict, val_dates, all_months)
mlp_rc_test = rank_correlation_oos(mlp_predictor.predict, test_dates, all_months)
print(f'mlp rank corr: val = {mlp_rc_val:.4f}, test = {mlp_rc_test:.4f}')

mlp_eval = evaluate_and_save(mlp_predictor.predict, 'mlp', all_months, sorted_dates, test_dates, results_dir, cfg)

mls = mlp_eval['metrics']['long_short_scaled']
mlo = mlp_eval['metrics']['long_only_scaled']
print(f'mlp long short scaled: sharpe = {mls["sharpe"]:.4f}, ann_ret = {mls["ann_ret"] * 100:.2f}%, ann_vol = {mls["ann_vol"] * 100:.2f}%')
print(f'mlp long only scaled: sharpe = {mlo["sharpe"]:.4f}, ann_ret = {mlo["ann_ret"] * 100:.2f}%, ann_vol = {mlo["ann_vol"] * 100:.2f}%')


summary = {
	'construction': 'mean_split_softmax_cap_6m',
	'target_column': ret_col,
	'n_features': len(feature_cols),
	'feature_cols': feature_cols,
	'architecture': {
		'name': 'three_layer_mlp', 'n_hidden_layers': 3, 'hidden_width': mlp_best['d_model'],
		'activation': 'elu', 'dropout': mlp_best['dropout'], 'parameter_count': int(n_params),
	},
	'split': {
		'train': {'start': str(train_dates[0].date()), 'end': str(train_dates[-1].date()), 'n_months': len(train_dates), 'n_obs': int(x_train.shape[0])},
		'val': {'start': str(val_dates[0].date()), 'end': str(val_dates[-1].date()), 'n_months': len(val_dates)},
		'test': {'start': str(test_dates[0].date()), 'end': str(test_dates[-1].date()), 'n_months': len(test_dates)},
	},
	'config': {
		'rebalance_freq': cfg.rebalance_freq, 'horizon_months': cfg.horizon_months,
		'tc_bps': cfg.tc_bps, 'min_stocks': cfg.min_stocks, 'min_leg_stocks': cfg.min_leg_stocks,
		'ret_clip': [cfg.ret_clip_low, cfg.ret_clip_high], 'target_vol': cfg.target_vol,
		'vol_lookback_months': cfg.vol_lookback_months, 'vol_lookback_periods': cfg.vol_lookback_periods,
		'n_vol_periods': cfg.n_vol_periods, 'periods_per_year': cfg.periods_per_year,
		'max_leverage_long_only': cfg.max_leverage_long_only, 'max_leverage_long_short': cfg.max_leverage_long_short,
		'max_position_weight': cfg.max_position_weight, 'n_epochs_hpo': n_epochs_hpo,
		'n_final_epochs': n_final_epochs, 'patience': patience, 'grad_clip_norm': grad_clip_norm,
		'optuna_seed': optuna_seed, 'torch_seed': torch_seed, 'n_trials': n_trials,
	},
	'mlp': {
		'best_params': mlp_best, 'best_val_long_short_sharpe': mlp_best_value,
		'best_trial_number': mlp_study.best_trial.number if mlp_study is not None else None,
		'n_trials_completed': sum(1 for t in mlp_study.trials if t.state.name == 'COMPLETE') if mlp_study is not None else None,
		'hpo_time_seconds': float(mlp_hpo_time), 'final_training_time_seconds': float(mlp_train_time),
		'best_epoch_from_hpo': mlp_best_epoch, 'n_final_epochs': n_final_epochs,
		'rc_val': float(mlp_rc_val), 'rc_test': float(mlp_rc_test),
		'portfolio_metrics': {k: strip_per_year(v) for k, v in mlp_eval['metrics'].items()},
	},
}

with open(results_dir / 'mlp_summary.json', 'w') as fh:
	json.dump(summary, fh, indent = 2, default = float)
print('summary json saved')


rows = []
for portfolio, scaling, key in [
	('long_short', 'unscaled', 'long_short_unscaled'), ('long_short', 'scaled', 'long_short_scaled'),
	('long_only', 'unscaled', 'long_only_unscaled'), ('long_only', 'scaled', 'long_only_scaled'),
]:
	m = mlp_eval['metrics'][key]
	rows.append({
		'model': 'mlp', 'portfolio': portfolio, 'scaling': scaling, 'rc_test': round(mlp_rc_test, 4),
		'sharpe': round_or_none(m['sharpe'], 4), 'se': round_or_none(m['se_sharpe'], 4),
		'ann_ret': round_or_none(m['ann_ret'] * 100, 2), 'ann_vol': round_or_none(m['ann_vol'] * 100, 2),
		'cagr': round_or_none(m['cagr'] * 100, 2), 'cum_return': round_or_none(m['cum_return'] * 100, 2),
		'max_dd': round_or_none(m['max_dd'] * 100, 2), 'n_obs': m['n_obs'],
	})

summary_table = pd.DataFrame(rows)
print('MLP Benchmark, EM Universe, mean split capped softmax, 6m rebalance')
print(summary_table.to_string(index = False))
summary_table.to_csv(results_dir / 'mlp_summary.csv', index = False)
print('summary csv saved')


per_year_rows = []
flush_per_year_rows(per_year_rows, 'mlp', 'long_short', 'unscaled', mlp_eval['metrics']['long_short_unscaled'])
flush_per_year_rows(per_year_rows, 'mlp', 'long_short', 'scaled', mlp_eval['metrics']['long_short_scaled'])
flush_per_year_rows(per_year_rows, 'mlp', 'long_only', 'unscaled', mlp_eval['metrics']['long_only_unscaled'])
flush_per_year_rows(per_year_rows, 'mlp', 'long_only', 'scaled', mlp_eval['metrics']['long_only_scaled'])

per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(results_dir / 'mlp_per_year_metrics.csv', index = False)
print(f'per year metrics saved, {len(per_year_df)} rows')


period_rows = []
for portfolio, scaling, rets, dates in [
	('long_short', 'unscaled', mlp_eval['returns_ls_unscaled'], mlp_eval['dates_ls']),
	('long_short', 'scaled', mlp_eval['returns_ls_scaled'], mlp_eval['dates_ls']),
	('long_only', 'unscaled', mlp_eval['returns_lo_unscaled'], mlp_eval['dates_lo']),
	('long_only', 'scaled', mlp_eval['returns_lo_scaled'], mlp_eval['dates_lo']),
]:
	period_rows.extend(build_diagnostic_rows('mlp', portfolio, scaling, rets, dates, cfg.periods_per_year, 4, '4p'))

per_period_df = pd.DataFrame(period_rows)
per_period_df.to_csv(results_dir / 'mlp_per_period_metrics.csv', index = False)
print(f'per period metrics saved, {len(per_period_df)} rows')


# monthly diagnostic simulation, run over sorted_dates so the vol overlay
# has full warm up before the test window, then sliced to the test window
mo_sim = run_mean_split_simulation_monthly(mlp_predictor.predict, sorted_dates, all_months, cfg)
mo_ls = mo_sim['long_short']
mo_lo = mo_sim['long_only']

mo_ls_lev = apply_vol_target_monthly(mo_ls['returns'] - mo_ls['tc'], mo_ls['rb_indices'], cfg.target_vol, cfg.vol_lookback_months, cfg.max_leverage_long_short)
mo_lo_lev = apply_vol_target_monthly(mo_lo['returns'] - mo_lo['tc'], mo_lo['rb_indices'], cfg.target_vol, cfg.vol_lookback_months, cfg.max_leverage_long_only)

mo_ls_unscaled_full = mo_ls['returns'] - mo_ls['tc']
mo_ls_scaled_full = mo_ls_lev * mo_ls['returns'] - mo_ls_lev * mo_ls['tc']
mo_lo_unscaled_full = mo_lo['returns'] - mo_lo['tc']
mo_lo_scaled_full = mo_lo_lev * mo_lo['returns'] - mo_lo_lev * mo_lo['tc']

test_set = set(test_dates)
mo_ls_mask = np.array([d in test_set for d in mo_ls['dates']])
mo_lo_mask = np.array([d in test_set for d in mo_lo['dates']])

mo_ls_unscaled_test = mo_ls_unscaled_full[mo_ls_mask]
mo_ls_scaled_test = mo_ls_scaled_full[mo_ls_mask]
mo_lo_unscaled_test = mo_lo_unscaled_full[mo_lo_mask]
mo_lo_scaled_test = mo_lo_scaled_full[mo_lo_mask]
mo_ls_dates_test = [d for d, keep in zip(mo_ls['dates'], mo_ls_mask) if keep]
mo_lo_dates_test = [d for d, keep in zip(mo_lo['dates'], mo_lo_mask) if keep]

mo_ls_scaled_m = portfolio_metrics(mo_ls_scaled_test, 12.0, dates = mo_ls_dates_test)
mo_lo_scaled_m = portfolio_metrics(mo_lo_scaled_test, 12.0, dates = mo_lo_dates_test)

print(f'mlp monthly long short scaled: sharpe = {mo_ls_scaled_m["sharpe"]:.4f}, ann_ret = {mo_ls_scaled_m["ann_ret"] * 100:.2f}%, ann_vol = {mo_ls_scaled_m["ann_vol"] * 100:.2f}%')
print(f'mlp monthly long only scaled: sharpe = {mo_lo_scaled_m["sharpe"]:.4f}, ann_ret = {mo_lo_scaled_m["ann_ret"] * 100:.2f}%, ann_vol = {mo_lo_scaled_m["ann_vol"] * 100:.2f}%')

monthly_rows = []
for portfolio, scaling, rets, dates in [
	('long_short', 'unscaled', mo_ls_unscaled_test, mo_ls_dates_test),
	('long_short', 'scaled', mo_ls_scaled_test, mo_ls_dates_test),
	('long_only', 'unscaled', mo_lo_unscaled_test, mo_lo_dates_test),
	('long_only', 'scaled', mo_lo_scaled_test, mo_lo_dates_test),
]:
	monthly_rows.extend(build_diagnostic_rows('mlp', portfolio, scaling, rets, dates, 12.0, 12, '12m'))

per_month_df = pd.DataFrame(monthly_rows)
per_month_df.to_csv(results_dir / 'mlp_per_month_metrics.csv', index = False)
print(f'per month metrics saved, {len(per_month_df)} rows')
