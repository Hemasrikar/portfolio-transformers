import gc
import json
import math
import time
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from scipy.stats import spearmanr

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

train_path = Path('data/processed/train.parquet')
val_path = Path('data/processed/val.parquet')
test_path = Path('data/processed/test.parquet')

results_dir = Path('results/benchmark/ft_transformer_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

target_col = 'target_6m'

rebalance_freq = 6
tc_bps = 25
min_stocks = 30

target_vol = 0.10
vol_lookback_months = 36
n_vol_periods = max(1, vol_lookback_months // rebalance_freq)
max_position_weight = 0.05
max_leverage_long_only = 3.0
max_leverage_long_short = 3.0
periods_per_year = 12.0 / rebalance_freq

n_epochs_train = 60
patience = 8
grad_clip_norm = 1.0

n_trials = 30
optuna_seed = 24
torch_seed = 24

run_timestamp = datetime.utcnow().isoformat(timespec = 'seconds')

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
use_amp = cuda_available


class reglu(nn.Module):
	def forward(self, x):
		a, b = x.chunk(2, dim = -1)
		return a * F.relu(b)


class feature_tokeniser(nn.Module):
	def __init__(self, n_features, d_model):
		super().__init__()
		self.weights = nn.Parameter(torch.empty(n_features, d_model))
		self.biases = nn.Parameter(torch.empty(n_features, d_model))
		bound = 1.0 / math.sqrt(d_model)
		nn.init.uniform_(self.weights, -bound, bound)
		nn.init.uniform_(self.biases, -bound, bound)

	def forward(self, x):
		return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class ft_transformer_block(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, dropout, first_block):
		super().__init__()
		self.norm1 = nn.Identity() if first_block else nn.LayerNorm(d_model)
		self.attn = nn.MultiheadAttention(d_model, n_heads, dropout = dropout, batch_first = True)
		self.norm2 = nn.LayerNorm(d_model)
		self.ffn = nn.Sequential(
			nn.Linear(d_model, 2 * d_ff),
			reglu(),
			nn.Dropout(dropout),
			nn.Linear(d_ff, d_model),
		)
		self.residual_drop = nn.Dropout(dropout)

	def forward(self, x):
		normed = self.norm1(x)
		attn_out, _ = self.attn(normed, normed, normed, need_weights = False)
		x = x + self.residual_drop(attn_out)
		x = x + self.residual_drop(self.ffn(self.norm2(x)))
		return x


class ft_transformer(nn.Module):
	def __init__(self, n_features, d_model, n_heads, n_layers, d_ff, dropout):
		super().__init__()
		self.tokeniser = feature_tokeniser(n_features, d_model)
		self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
		self.blocks = nn.ModuleList([
			ft_transformer_block(d_model, n_heads, d_ff, dropout, first_block = (i == 0)) for i in range(n_layers)
		])
		self.head = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.ReLU(),
			nn.Linear(d_model, 1),
		)

	def forward(self, x):
		b = x.size(0)
		tokens = self.tokeniser(x)
		cls = self.cls_token.expand(b, -1, -1)
		tokens = torch.cat([cls, tokens], dim = 1)
		for block in self.blocks:
			tokens = block(tokens)
		cls_out = tokens[:, 0, :]
		return self.head(cls_out).squeeze(-1)


class ft_predictor:
	def __init__(self, model, device, batch_size = 1024):
		self.model = model
		self.device = device
		self.batch_size = batch_size

	def predict(self, x):
		self.model.eval()
		preds = []
		with torch.no_grad():
			x_t = torch.from_numpy(x).float().to(self.device)
			for i in range(0, len(x_t), self.batch_size):
				batch = x_t[i:i + self.batch_size]
				if use_amp:
					with torch.autocast('cuda'):
						out = self.model(batch)
				else:
					out = self.model(batch)
				preds.append(out.float().cpu().numpy())
		return np.concatenate(preds, axis = 0)


def capped_softmax_weights(scores, max_weight, max_iter = 20):
	scores = np.asarray(scores, dtype = np.float64)
	n = scores.shape[0]
	if n == 0:
		return np.zeros(0, dtype = np.float64)
	if max_weight <= 1.0 / n + 1e-12:
		return np.full(n, 1.0 / n, dtype = np.float64)
	z = scores - scores.max()
	w = np.exp(z)
	s = w.sum()
	if s <= 0 or not np.isfinite(s):
		return np.full(n, 1.0 / n, dtype = np.float64)
	w = w / s
	for _ in range(max_iter):
		over = w > max_weight
		if not over.any():
			break
		excess = float((w[over] - max_weight).sum())
		residual = ~over
		residual_total = float(w[residual].sum())
		if residual_total <= 1e-12:
			break
		w = np.where(over, max_weight, w)
		w = np.where(residual, w * (1.0 + excess / residual_total), w)
	return w


def renorm_over_valid(weights, valid):
	weights = np.asarray(weights, dtype = np.float64)
	valid = np.asarray(valid, dtype = bool)
	if not valid.any():
		return weights
	valid_total = float(weights[valid].sum())
	if valid_total <= 1e-12:
		return weights
	out = np.zeros_like(weights)
	out[valid] = weights[valid] / valid_total
	return out


def drift_weights(prev_ids, prev_w, realised_by_id):
	if prev_ids is None or prev_w is None or len(prev_w) == 0:
		return None, None
	n = len(prev_w)
	ids_list = []
	growth = np.zeros(n, dtype = np.float64)
	for j in range(n):
		fid = int(prev_ids[j])
		ids_list.append(fid)
		growth[j] = float(prev_w[j]) * (1.0 + float(realised_by_id.get(fid, 0.0)))
	g_sum = float(growth.sum())
	drifted = growth / g_sum if g_sum > 1e-12 else growth
	return ids_list, drifted


def weight_l1_turnover(prev_ids, prev_w, curr_ids, curr_w):
	if curr_ids is None or curr_w is None or len(curr_w) == 0:
		return 0.0
	curr_map = {int(curr_ids[j]): float(curr_w[j]) for j in range(len(curr_ids))}
	if prev_ids is None or prev_w is None or len(prev_w) == 0:
		return float(sum(abs(v) for v in curr_map.values()))
	prev_map = {int(prev_ids[j]): float(prev_w[j]) for j in range(len(prev_ids))}
	all_ids = set(prev_map) | set(curr_map)
	return float(sum(abs(curr_map.get(fid, 0.0) - prev_map.get(fid, 0.0)) for fid in all_ids))


def portfolio_metrics(rets, periods_per_year):
	rets = np.asarray(rets, dtype = np.float64)
	if len(rets) == 0:
		return {}
	n = len(rets)
	ann_ret = float(rets.mean() * periods_per_year)
	ann_vol = float(rets.std(ddof = 1) * np.sqrt(periods_per_year))
	sharpe = ann_ret / max(ann_vol, 1e-8)
	se = float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / n))
	cum_return = float((1.0 + rets).prod() - 1.0)
	cagr = float((1.0 + cum_return) ** (periods_per_year / n) - 1.0) if (1.0 + cum_return) > 0 else float('nan')
	cw = np.cumprod(1.0 + rets)
	pk = np.maximum.accumulate(cw)
	max_dd = float(((pk - cw) / pk).max())

	return {
		'ann_ret': ann_ret,
		'ann_vol': ann_vol,
		'sharpe': sharpe,
		'se_sharpe': se,
		'max_dd': max_dd,
		'cagr': cagr,
		'cumulative_return': cum_return,
		'n_obs': n,
	}


def overlay_leverage(raw_hist, periods_per_year, max_leverage):
	if len(raw_hist) >= n_vol_periods:
		recent = np.array(raw_hist[-n_vol_periods:])
		realised_vol = recent.std(ddof = 1) * np.sqrt(periods_per_year)
		lev = target_vol / max(realised_vol, 1e-6)
		return float(np.clip(lev, 1.0 / max_leverage, max_leverage))
	return 1.0


def seed_vol_history(predictor, month_dates, all_months, leg_kind):
	returns = []
	for pos, eom in enumerate(month_dates):
		if pos % rebalance_freq != 0:
			continue
		if eom not in all_months:
			continue
		m = all_months[eom]
		pred = predictor.predict(m['x'])
		r = m['r']
		valid = np.isfinite(pred) & np.isfinite(r)
		n_firms = len(pred)
		if leg_kind == 'long_only':
			valid_pred = np.isfinite(pred)
			lo_w = capped_softmax_weights(pred[valid_pred], max_position_weight)
			w_full = np.zeros(n_firms, dtype = np.float64)
			w_full[valid_pred] = lo_w
			w_full = renorm_over_valid(w_full, valid)
			returns.append(float(np.sum(w_full[valid] * r[valid])))
		else:
			mean_s = float(pred[np.isfinite(pred)].mean())
			long_idx = np.where((pred > mean_s) & valid)[0]
			short_idx = np.where((pred <= mean_s) & valid)[0]
			if len(long_idx) == 0 or len(short_idx) == 0:
				returns.append(0.0)
				continue
			long_w = renorm_over_valid(capped_softmax_weights(pred[long_idx] - mean_s, max_position_weight), valid[long_idx])
			short_w = renorm_over_valid(capped_softmax_weights(mean_s - pred[short_idx], max_position_weight), valid[short_idx])
			long_ret = float(np.sum(long_w * r[long_idx]))
			short_ret = float(np.sum(short_w * r[short_idx]))
			returns.append(long_ret - short_ret)
	return returns[-n_vol_periods:] if returns else []


def rank_correlation_oos(predictor, month_dates, all_months):
	corrs = []
	for eom in month_dates:
		if eom not in all_months:
			continue
		m = all_months[eom]
		pred = predictor.predict(m['x'])
		valid = np.isfinite(pred) & np.isfinite(m['r'])
		if valid.sum() < 10:
			continue
		result = spearmanr(pred[valid], m['r'][valid])
		c = float(result[0])
		if not np.isnan(c):
			corrs.append(c)
	return float(np.mean(corrs)) if corrs else 0.0


def run_mean_split_simulation(predictor, month_dates, all_months, ls_seed = None, lo_seed = None, record_holdings = False):
	ls_scaled, ls_unscaled, ls_dates, ls_lev = [], [], [], []
	lo_scaled, lo_unscaled, lo_dates, lo_lev = [], [], [], []
	raw_ls_hist = list(ls_seed) if ls_seed else []
	raw_lo_hist = list(lo_seed) if lo_seed else []

	prev_long_ids = prev_long_w = prev_long_realised = None
	prev_short_ids = prev_short_w = prev_short_realised = None
	prev_lo_ids = prev_lo_w = prev_lo_realised = None

	ls_holdings, lo_holdings = [], []
	rb = -1

	for pos, eom in enumerate(month_dates):
		if pos % rebalance_freq != 0:
			continue
		if eom not in all_months:
			continue
		m = all_months[eom]
		ids, r, x = m['ids'], m['r'], m['x']
		n_firms = len(ids)
		if n_firms < min_stocks:
			continue

		pred = predictor.predict(x)
		valid_pred = np.isfinite(pred)
		if valid_pred.sum() < min_stocks:
			continue
		valid = valid_pred & np.isfinite(r)
		rb += 1

		mean_score = float(pred[valid_pred].mean())
		long_idx = np.where((pred > mean_score) & valid_pred)[0]
		short_idx = np.where((pred <= mean_score) & valid_pred)[0]
		long_ids = [int(v) for v in ids[long_idx]]
		short_ids = [int(v) for v in ids[short_idx]]

		long_w = renorm_over_valid(capped_softmax_weights(pred[long_idx] - mean_score, max_position_weight), valid[long_idx])
		short_w = renorm_over_valid(capped_softmax_weights(mean_score - pred[short_idx], max_position_weight), valid[short_idx])

		long_realised, long_ret = {}, 0.0
		for i, fi in enumerate(long_idx):
			rr = float(r[fi]) if valid[fi] else 0.0
			long_realised[int(ids[fi])] = rr
			long_ret += long_w[i] * rr
		short_realised, short_ret = {}, 0.0
		for i, fi in enumerate(short_idx):
			rr = float(r[fi]) if valid[fi] else 0.0
			short_realised[int(ids[fi])] = rr
			short_ret += short_w[i] * rr
		ls_ret = long_ret - short_ret

		d_long_ids, d_long_w = drift_weights(prev_long_ids, prev_long_w, prev_long_realised)
		d_short_ids, d_short_w = drift_weights(prev_short_ids, prev_short_w, prev_short_realised)
		lt = weight_l1_turnover(d_long_ids, d_long_w, long_ids, long_w)
		st = weight_l1_turnover(d_short_ids, d_short_w, short_ids, short_w)
		ls_flat_tc = (lt + st) * tc_bps / 10000.0

		ls_leverage = overlay_leverage(raw_ls_hist, periods_per_year, max_leverage_long_short)
		ls_scaled.append(ls_leverage * ls_ret - ls_leverage * ls_flat_tc)
		ls_unscaled.append(ls_ret - ls_flat_tc)
		ls_lev.append(ls_leverage)
		ls_dates.append(eom)
		raw_ls_hist.append(ls_ret)

		prev_long_ids, prev_long_w, prev_long_realised = long_ids, long_w, long_realised
		prev_short_ids, prev_short_w, prev_short_realised = short_ids, short_w, short_realised

		lo_w = capped_softmax_weights(pred[valid_pred], max_position_weight)
		lo_w_full = np.zeros(n_firms, dtype = np.float64)
		lo_w_full[valid_pred] = lo_w
		lo_w_full = renorm_over_valid(lo_w_full, valid)
		lo_ids = [int(v) for v in ids]
		lo_realised, lo_ret = {}, 0.0
		for fi in range(n_firms):
			rr = float(r[fi]) if valid[fi] else 0.0
			lo_realised[int(ids[fi])] = rr
			lo_ret += lo_w_full[fi] * rr

		d_lo_ids, d_lo_w = drift_weights(prev_lo_ids, prev_lo_w, prev_lo_realised)
		lo_turn = weight_l1_turnover(d_lo_ids, d_lo_w, lo_ids, lo_w_full)
		lo_flat_tc = lo_turn * tc_bps / 10000.0

		lo_leverage = overlay_leverage(raw_lo_hist, periods_per_year, max_leverage_long_only)
		lo_scaled.append(lo_leverage * lo_ret - lo_leverage * lo_flat_tc)
		lo_unscaled.append(lo_ret - lo_flat_tc)
		lo_lev.append(lo_leverage)
		lo_dates.append(eom)
		raw_lo_hist.append(lo_ret)

		prev_lo_ids, prev_lo_w, prev_lo_realised = lo_ids, lo_w_full, lo_realised

		if record_holdings:
			for i, fi in enumerate(long_idx):
				ls_holdings.append({'rebalance_index': rb, 'eom': eom, 'leg': 'long', 'id': int(ids[fi]), 'weight': float(long_w[i]), 'realised_return': float(r[fi]) if valid[fi] else float('nan')})
			for i, fi in enumerate(short_idx):
				ls_holdings.append({'rebalance_index': rb, 'eom': eom, 'leg': 'short', 'id': int(ids[fi]), 'weight': float(-short_w[i]), 'realised_return': float(r[fi]) if valid[fi] else float('nan')})
			for fi in range(n_firms):
				if valid_pred[fi]:
					lo_holdings.append({'rebalance_index': rb, 'eom': eom, 'leg': 'long', 'id': int(ids[fi]), 'weight': float(lo_w_full[fi]), 'realised_return': float(r[fi]) if valid[fi] else float('nan')})

	return {
		'long_short': {
			'scaled': np.array(ls_scaled),
			'unscaled': np.array(ls_unscaled),
			'leverage': np.array(ls_lev),
			'dates': ls_dates,
			'holdings_df': pd.DataFrame(ls_holdings),
		},
		'long_only': {
			'scaled': np.array(lo_scaled),
			'unscaled': np.array(lo_unscaled),
			'leverage': np.array(lo_lev),
			'dates': lo_dates,
			'holdings_df': pd.DataFrame(lo_holdings),
		},
	}


def load_data():
	train_schema = pq.read_schema(train_path)
	non_feature = {
		'id', 'gvkey', 'iid', 'isin', 'cusip', 'permno', 'permco',
		'eom', 'date', 'excntry', 'curcd', 'size_grp',
		'ret_exc_lead1m', 'ret_exc_lead6m',
		'target_1m', 'target_3m', 'target_6m', 'target_12m',
		'sic', 'naics', 'gics', 'ff49',
		'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd',
		'obs_main', 'exch_main', 'primary_sec', 'common', 'bidask',
		'source_crsp',
		'adjfct', 'fx', 'ret_lag_dif',
		'ret', 'ret_exc', 'ret_local',
		'me', 'me_company', 'prc', 'prc_local', 'prc_high', 'prc_low',
		'dolvol', 'shares', 'tvol',
		'enterprise_value', 'book_equity', 'assets', 'sales',
		'net_income', 'intrinsic_value',
	}
	feature_cols = [
		c for c in train_schema.names
		if c not in non_feature
		and pa.types.is_floating(train_schema.field(c).type)
		and '_lag' not in c
		and not c.startswith('target_')
	]
	print('feature columns selected', ',', len(feature_cols))

	needed = list(dict.fromkeys(
		[c for c in ['id', 'eom', target_col] + feature_cols
		 if c in train_schema.names]
	))
	train_df = pd.read_parquet(train_path, columns = needed)
	val_df = pd.read_parquet(val_path, columns = needed)
	test_df = pd.read_parquet(test_path, columns = needed)

	for d in (train_df, val_df, test_df):
		d['eom'] = pd.to_datetime(d['eom'])

	train_end = train_df['eom'].max()
	val_end = val_df['eom'].max()

	print('train rows', ',', len(train_df))
	print('val rows', ',', len(val_df))
	print('test rows', ',', len(test_df))

	for label, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
		non_null = split[target_col].notna().sum()
		print(label, target_col, 'non-null', ',', non_null, ',', round(100.0 * non_null / len(split), 1))

	check = train_df[target_col].dropna().abs().mean()
	assert check < 2.0

	n_feat = len(feature_cols)
	all_months = {}

	for split_df in (train_df, val_df, test_df):
		for eom, group in split_df.groupby('eom', sort = True):
			if len(group) < min_stocks:
				continue
			ids = group['id'].to_numpy()
			r = group[target_col].to_numpy().astype(np.float64)
			x = group[feature_cols].to_numpy(dtype = np.float32, copy = True)
			x = np.nan_to_num(x, nan = 0.0, copy = False)
			all_months[eom] = {'ids': ids, 'r': r, 'x': x}

	sorted_dates = sorted(all_months.keys())
	train_dates = [d for d in sorted_dates if d <= train_end]
	val_dates = [d for d in sorted_dates if train_end < d <= val_end]

	x_train_full = np.vstack([all_months[d]['x'] for d in train_dates])
	y_train_full = np.concatenate([all_months[d]['r'] for d in train_dates]).astype(np.float32)
	train_mask = np.isfinite(y_train_full)
	x_train = x_train_full[train_mask]
	y_train = y_train_full[train_mask]
	print('x_train shape', ',', x_train.shape)

	return {
		'x_train': x_train, 'y_train': y_train,
		'all_months': all_months, 'feature_cols': feature_cols,
		'val_dates': val_dates, 'n_feat': n_feat,
	}


def train_ft_transformer(params, x_train_pool, y_train_pool, val_dates_local, all_months, n_epochs, patience, device, seed, n_features):
	torch.manual_seed(seed)
	np.random.seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed_all(seed)

	d_model = params['n_heads'] * params['d_model_per_head']
	d_ff = d_model * params['d_ff_ratio']

	model = ft_transformer(
		n_features = n_features,
		d_model = d_model,
		n_heads = params['n_heads'],
		n_layers = params['n_layers'],
		d_ff = d_ff,
		dropout = params['dropout'],
	).to(device)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr = params['learning_rate'],
		weight_decay = params['weight_decay'],
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 3)
	scaler = torch.GradScaler('cuda', enabled = use_amp)
	criterion = nn.MSELoss()

	x_train_t = torch.from_numpy(x_train_pool).float()
	y_train_t = torch.from_numpy(y_train_pool).float()
	if cuda_available:
		x_train_t = x_train_t.pin_memory()
		y_train_t = y_train_t.pin_memory()
	n_train = len(x_train_t)
	batch_size = params['batch_size']

	predictor = ft_predictor(model, device, batch_size = 1024)

	best_rc = -np.inf
	best_state = None
	best_epoch = 0
	patience_ctr = 0
	train_losses = []
	val_rank_corr = []

	for epoch in range(n_epochs):
		model.train()
		perm = torch.randperm(n_train)
		epoch_loss = 0.0
		n_batches = 0

		for i in range(0, n_train, batch_size):
			idx = perm[i:i + batch_size]
			x_batch = x_train_t[idx].to(device, non_blocking = True)
			y_batch = y_train_t[idx].to(device, non_blocking = True)
			optimizer.zero_grad()
			if use_amp:
				with torch.autocast('cuda'):
					pred = model(x_batch)
					loss = criterion(pred, y_batch)
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip_norm)
				scaler.step(optimizer)
				scaler.update()
			else:
				pred = model(x_batch)
				loss = criterion(pred, y_batch)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip_norm)
				optimizer.step()
			epoch_loss += float(loss.item())
			n_batches += 1

		avg_loss = epoch_loss / max(n_batches, 1)
		val_rc = rank_correlation_oos(predictor, val_dates_local, all_months)
		scheduler.step(val_rc)
		train_losses.append(avg_loss)
		val_rank_corr.append(val_rc)

		if val_rc > best_rc:
			best_rc = val_rc
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			best_epoch = epoch
			patience_ctr = 0
		else:
			patience_ctr += 1
			if patience_ctr >= patience:
				break

	if best_state is not None:
		model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

	return model, {
		'train_losses': train_losses, 'val_rank_corr': val_rank_corr,
		'best_epoch': best_epoch, 'best_val_rc': float(best_rc),
		'n_epochs_run': len(train_losses), 'd_model': d_model,
		'd_ff': d_ff,
	}


def main():
	print('device', ',', device)
	if cuda_available:
		print('device name', ',', cuda_device_name)
		print('mixed precision', ',', 'enabled')
	print('run timestamp utc', ',', run_timestamp)
	print('results_dir', ',', results_dir)
	print('n_trials', ',', n_trials)
	print('construction', ',', 'mean_split_softmax_cap_6m')
	print('periods_per_year', ',', periods_per_year)
	print('n_vol_periods', ',', n_vol_periods)

	best_params_path = results_dir / 'ft_best_params.json'
	trials_path = results_dir / 'ft_optuna_trials.csv'
	study_path = results_dir / 'ft_optuna_study.pkl'

	data = load_data()
	x_train = data['x_train']
	y_train = data['y_train']
	all_months = data['all_months']
	val_dates = data['val_dates']
	n_feat = data['n_feat']

	def objective(trial):
		params = {
			'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
			'd_model_per_head': trial.suggest_categorical('d_model_per_head', [8, 16, 32]),
			'd_ff_ratio': trial.suggest_categorical('d_ff_ratio', [2, 4]),
			'n_layers': trial.suggest_int('n_layers', 1, 4),
			'dropout': trial.suggest_float('dropout', 0.0, 0.3),
			'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
			'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-3, log = True),
			'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log = True),
		}
		model, log = train_ft_transformer(
			params = params,
			x_train_pool = x_train,
			y_train_pool = y_train,
			val_dates_local = val_dates,
			all_months = all_months,
			n_epochs = n_epochs_train,
			patience = patience,
			device = device,
			seed = torch_seed,
			n_features = n_feat,
		)
		predictor = ft_predictor(model, device, batch_size = 1024)

		val_rc = float(log['best_val_rc'])

		sim = run_mean_split_simulation(predictor, val_dates, all_months)
		ls, lo = sim['long_short'], sim['long_only']
		if len(ls['scaled']) == 0:
			ls_sharpe, lo_sharpe = float('nan'), float('nan')
		else:
			ls_sharpe = portfolio_metrics(ls['scaled'], periods_per_year).get('sharpe', float('nan'))
			lo_sharpe = portfolio_metrics(lo['scaled'], periods_per_year).get('sharpe', float('nan'))

		trial.set_user_attr('best_epoch', log['best_epoch'])
		trial.set_user_attr('n_epochs_run', log['n_epochs_run'])
		trial.set_user_attr('best_val_rc', val_rc)
		trial.set_user_attr('val_sharpe_long_short', float(ls_sharpe))
		trial.set_user_attr('val_sharpe_long_only', float(lo_sharpe))
		trial.set_user_attr('d_model', log['d_model'])
		trial.set_user_attr('d_ff', log['d_ff'])
		trial.set_user_attr('param_count', sum(p.numel() for p in model.parameters()))

		del model, predictor
		gc.collect()
		if cuda_available:
			torch.cuda.empty_cache()
		return val_rc

	study = optuna.create_study(
		direction = 'maximize',
		sampler = optuna.samplers.TPESampler(seed = optuna_seed),
		study_name = f'ft_em_{run_timestamp}',
	)
	t0 = time.time()
	study.optimize(objective, n_trials = n_trials, show_progress_bar = True)
	hpo_time = time.time() - t0
	best_params = study.best_params

	print('best val rank corr', ',', round(study.best_value, 4))
	print('best params', ',', best_params)
	print('hpo time', ',', round(hpo_time, 1), 's', ',', round(hpo_time / 60, 2), 'min')

	trials_df = study.trials_dataframe()
	trials_df.to_csv(trials_path, index = False)
	with open(study_path, 'wb') as fh:
		pickle.dump(study, fh)

	with open(best_params_path, 'w') as fh:
		json.dump({
			'construction': 'mean_split_softmax_cap_6m',
			'target_column': target_col,
			'best_params': best_params,
			'best_val_rank_corr': float(study.best_value),
			'best_val_long_short_sharpe': float(study.best_trial.user_attrs.get('val_sharpe_long_short', float('nan'))),
			'best_trial_number': int(study.best_trial.number),
			'best_trial_user_attrs': dict(study.best_trial.user_attrs),
			'n_trials_completed': sum(1 for t in study.trials if t.state.name == 'COMPLETE'),
			'hpo_time_seconds': float(hpo_time),
			'optuna_seed': optuna_seed,
			'torch_seed': torch_seed,
			'run_timestamp_utc': run_timestamp,
		}, fh, indent = 2, default = float)
	print('best params saved', ',', 'ft_best_params.json')
	print('optuna trials saved', ',', 'ft_optuna_trials.csv')
	print('optuna study saved', ',', 'ft_optuna_study.pkl')


if __name__ == '__main__':
	main()
