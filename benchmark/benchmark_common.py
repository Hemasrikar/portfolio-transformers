import gc
import json
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import optuna
from scipy.stats import spearmanr, rankdata


# shared across all benchmark scripts so a fix made here applies everywhere
# rather than being duplicated and risking silent divergence between files


def portfolio_metrics(rets, periods_per_year, dates = None, vol_floor = 1e-8, sharpe_vol_floor = 1e-12):
	rets = np.asarray(rets, dtype = np.float64)
	if len(rets) == 0:
		out = {
			'ann_ret': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan,
			'se_sharpe': np.nan, 'max_dd': np.nan, 'cagr': np.nan,
			'cum_return': np.nan, 'n_obs': 0,
		}
		if dates is not None:
			out['per_year'] = {}
		return out

	n = len(rets)
	ann_ret = float(rets.mean() * periods_per_year)
	ann_vol = float(rets.std() * np.sqrt(periods_per_year)) if n > 1 else 0.0

	if ann_vol > sharpe_vol_floor:
		sharpe = ann_ret / max(ann_vol, vol_floor)
		se = float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / n))
	else:
		sharpe = float('nan')
		se = float('nan')

	cw = np.cumprod(1.0 + rets)
	cw_dd = np.maximum(cw, 1e-12)
	pk = np.maximum.accumulate(np.concatenate(([1.0], cw_dd)))[1:]
	max_dd = float(((pk - cw_dd) / pk).max()) if len(cw) > 0 else 0.0
	cum_return = float(cw[-1] - 1.0)
	years_elapsed = n / periods_per_year
	cagr = float(cw[-1] ** (1.0 / years_elapsed) - 1.0) if cw[-1] > 0 and years_elapsed > 0 else float('nan')

	out = {
		'ann_ret': ann_ret, 'ann_vol': ann_vol,
		'sharpe': sharpe, 'se_sharpe': se,
		'max_dd': max_dd, 'cagr': cagr,
		'cum_return': cum_return, 'n_obs': n,
	}

	if dates is not None:
		years = pd.DatetimeIndex(dates).year.to_numpy()
		per_year = {}
		for y in sorted(set(years.tolist())):
			mask = years == y
			sub = rets[mask]
			if len(sub) < 1:
				continue
			y_ret = float(sub.mean() * periods_per_year)
			y_vol = float(sub.std() * np.sqrt(periods_per_year)) if len(sub) > 1 else 0.0
			y_sharpe = y_ret / max(y_vol, vol_floor) if y_vol > sharpe_vol_floor else float('nan')
			ycw = np.cumprod(1.0 + sub)
			ycw_dd = np.maximum(ycw, 1e-12)
			ypk = np.maximum.accumulate(np.concatenate(([1.0], ycw_dd)))[1:]
			y_dd = float(((ypk - ycw_dd) / ypk).max())
			per_year[int(y)] = {
				'ann_ret': y_ret, 'ann_vol': y_vol,
				'sharpe': y_sharpe, 'max_dd': y_dd,
				'cum_return': float(ycw[-1] - 1.0),
				'n_obs': int(len(sub)),
			}
		out['per_year'] = per_year

	return out


def capped_softmax_weights(scores, max_weight):
	"""Softmax weights capped at max_weight via exact water filling. Once a
	weight is fixed at the cap it is never reconsidered, so remaining mass
	is always reallocated only among weights not yet capped. Terminates in
	at most n passes and respects the cap exactly wherever capping is
	mechanically feasible."""
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

	fixed = np.zeros(n, dtype = bool)
	result = np.zeros(n, dtype = np.float64)

	for _ in range(n):
		free = ~fixed
		if not free.any():
			break
		free_mass_target = 1.0 - fixed.sum() * max_weight
		free_raw_total = w[free].sum()
		if free_raw_total <= 1e-15:
			result[free] = free_mass_target / free.sum()
			break
		scale = free_mass_target / free_raw_total
		candidate = w[free] * scale
		over_local = candidate > max_weight + 1e-15
		if not over_local.any():
			result[free] = candidate
			break
		free_idx = np.where(free)[0]
		newly_fixed = free_idx[over_local]
		result[newly_fixed] = max_weight
		fixed[newly_fixed] = True

	return result


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


def mean_split_legs(pred, valid_pred, valid_ret, min_leg_stocks):
	"""Long and short leg index sets for a mean split. Both legs are
	restricted to firms with a valid prediction and a valid realised return
	before any weight is computed, so no later renormalisation is needed.
	Returns None if either leg falls below min_leg_stocks."""
	valid = valid_pred & valid_ret
	mean_score = float(pred[valid_pred].mean())
	long_idx = np.where((pred > mean_score) & valid)[0]
	short_idx = np.where((pred <= mean_score) & valid)[0]
	if len(long_idx) < min_leg_stocks or len(short_idx) < min_leg_stocks:
		return None
	return mean_score, long_idx, short_idx


def long_only_leg(valid_pred, valid_ret, min_leg_stocks):
	"""Index set for the long only leg, restricted to firms with a valid
	prediction and a valid realised return. Returns None if the leg falls
	below min_leg_stocks."""
	valid = valid_pred & valid_ret
	idx = np.where(valid)[0]
	if len(idx) < min_leg_stocks:
		return None
	return idx


def firm_id_turnover(prev_ids, curr_ids):
	prev = set(prev_ids.tolist()) if prev_ids is not None else set()
	curr = set(curr_ids.tolist())
	if not curr:
		return 0.0
	return (len(curr - prev) + len(prev - curr)) / max(len(curr), 1)


def weight_l1_turnover(prev_ids, prev_w, curr_ids, curr_w):
	if curr_ids is None or curr_w is None or len(curr_w) == 0:
		return 0.0
	curr_map = {int(curr_ids[j]): float(curr_w[j]) for j in range(len(curr_ids))}
	if prev_ids is None or prev_w is None or len(prev_w) == 0:
		return float(sum(abs(v) for v in curr_map.values()))
	prev_map = {int(prev_ids[j]): float(prev_w[j]) for j in range(len(prev_ids))}
	all_ids = set(prev_map.keys()) | set(curr_map.keys())
	return float(sum(abs(curr_map.get(fid, 0.0) - prev_map.get(fid, 0.0)) for fid in all_ids))


def drift_weights(prev_ids, prev_w, realised_returns_by_id):
	if prev_ids is None or prev_w is None or len(prev_w) == 0:
		return None, None
	n = len(prev_w)
	ids_list = [int(prev_ids[j]) for j in range(n)]
	growth = np.array([prev_w[j] * (1.0 + realised_returns_by_id.get(ids_list[j], 0.0)) for j in range(n)])
	g_sum = float(growth.sum())
	drifted = growth / g_sum if g_sum > 1e-12 else growth
	return ids_list, drifted


def apply_period_vol_overlay(period_rets, target_vol, n_vol_periods, periods_per_year, max_leverage):
	period_rets = np.asarray(period_rets, dtype = np.float64)
	n = len(period_rets)
	leverage = np.ones(n, dtype = np.float64)
	for t in range(n):
		if t < n_vol_periods:
			continue
		trailing = period_rets[t - n_vol_periods:t]
		if len(trailing) < 2:
			continue
		realised_vol = float(trailing.std() * np.sqrt(periods_per_year))
		lev = target_vol / max(realised_vol, 1e-8)
		leverage[t] = float(np.clip(lev, 1.0 / max_leverage, max_leverage))
	return leverage


def apply_overlay_and_costs(leg_gross_rets, leg_tc, target_vol, n_vol_periods, periods_per_year, max_leverage):
	leg_gross_rets = np.asarray(leg_gross_rets, dtype = np.float64)
	leg_tc = np.asarray(leg_tc, dtype = np.float64)
	leverage_path = apply_period_vol_overlay(leg_gross_rets, target_vol, n_vol_periods, periods_per_year, max_leverage)
	unscaled_net = leg_gross_rets - leg_tc
	scaled_net = leverage_path * leg_gross_rets - leverage_path * leg_tc
	return scaled_net, unscaled_net, leverage_path


def apply_vol_target_monthly(monthly_rets, rebalance_indices, target_vol, lookback_months, max_leverage):
	"""Monthly resolution counterpart to apply_period_vol_overlay, for
	diagnostics that need a trailing window measured in individual monthly
	observations rather than in rebalance periods. The two overlays are
	intentionally different estimators and are not interchangeable."""
	monthly_rets = np.asarray(monthly_rets, dtype = np.float64)
	n = len(monthly_rets)
	leverage = np.ones(n, dtype = np.float64)
	n_rb = len(rebalance_indices)
	for i in range(n_rb):
		rb_idx = rebalance_indices[i]
		start = max(0, rb_idx - lookback_months)
		trailing = monthly_rets[start:rb_idx]
		if len(trailing) < lookback_months:
			continue
		sigma_ann = float(trailing.std() * np.sqrt(12.0))
		lev = float(np.clip(target_vol / max(sigma_ann, 1e-8), 1.0 / max_leverage, max_leverage))
		next_rb = rebalance_indices[i + 1] if i + 1 < n_rb else n
		leverage[rb_idx:next_rb] = lev
	return leverage


def predict_at_dates(predict_fn, month_dates, all_months):
	rows = []
	for eom in month_dates:
		if eom not in all_months:
			continue
		m = all_months[eom]
		pred = predict_fn(m['x'])
		for k in range(len(m['ids'])):
			rows.append({
				'eom': eom,
				'id': m['ids'][k],
				'prediction': float(pred[k]),
				'realised_return': float(m['r'][k]),
			})
	return pd.DataFrame(rows)


def rank_correlation_oos(predict_fn, month_dates, all_months, min_cross_section = 10):
	corrs = []
	for eom in month_dates:
		if eom not in all_months:
			continue
		m = all_months[eom]
		pred = predict_fn(m['x'])
		valid = np.isfinite(pred) & np.isfinite(m['r'])
		if valid.sum() < min_cross_section:
			continue
		c = float(np.asarray(spearmanr(pred[valid], m['r'][valid]))[0])
		if not np.isnan(c):
			corrs.append(c)
	return float(np.mean(corrs)) if corrs else 0.0


# configuration shared across the mlp, xgboost and lightgbm benchmarks. the
# defaults below are identical across all three so that results stay
# directly comparable; a script can still override individual fields

@dataclass
class BenchmarkConfig:
	rebalance_freq: int = 6
	horizon_months: int = 6
	tc_bps: float = 25
	min_stocks: int = 30
	min_leg_stocks: int = 10
	ret_clip_low: float = -1.0
	ret_clip_high: float = 1.0
	target_vol: float = 0.10
	vol_lookback_months: int = 36
	vol_lookback_periods: int = 6
	max_leverage_long_only: float = 3.0
	max_leverage_long_short: float = 3.0
	max_position_weight: float = 0.05

	@property
	def periods_per_year(self):
		return 12.0 / self.rebalance_freq

	@property
	def n_vol_periods(self):
		return self.vol_lookback_periods


# non feature columns, common to every benchmark: identifiers, dates and
# grouping keys, classification codes, quality filters, return calculation
# metadata, and level forms of characteristics that are redundant with their
# ranked counterparts
NON_FEATURE_COLS = {
	'id', 'gvkey', 'iid', 'isin', 'cusip', 'permno', 'permco',
	'eom', 'date', 'excntry', 'curcd', 'size_grp',
	'sic', 'naics', 'gics', 'ff49',
	'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd',
	'obs_main', 'exch_main', 'primary_sec', 'common', 'bidask',
	'source_crsp',
	'adjfct', 'fx', 'ret_lag_dif',
	'ret', 'ret_exc', 'ret_local',
	'me', 'me_company', 'prc', 'prc_local', 'prc_high', 'prc_low',
	'dolvol', 'shares', 'tvol',
	'enterprise_value', 'book_equity', 'assets', 'sales', 'net_income',
	'intrinsic_value',
}


def load_universe(
	data_path, ret_col_1m, ret_col, cfg, train_end, store_r1m = False,
	missing_col_threshold = 0.30, row_missing_threshold = 1.0 / 3.0,
):
	train_end = pd.Timestamp(train_end)
	schema = pq.read_schema(data_path)
	non_feature = NON_FEATURE_COLS | {ret_col_1m}
	candidate_cols = [
		c for c in schema.names
		if c not in non_feature
		and pa.types.is_floating(schema.field(c).type)
		and '_lag' not in c
	]
	print(f'candidate feature columns: {len(candidate_cols)}')

	needed = list(dict.fromkeys([c for c in ['id', 'eom', 'excntry', ret_col_1m] + candidate_cols if c in schema.names]))
	table = pq.read_table(data_path, columns = needed)
	cast_fields = [
		field.with_type(pa.float32()) if field.name in candidate_cols and pa.types.is_float64(field.type) else field
		for field in table.schema
	]
	table = table.cast(pa.schema(cast_fields))

	df = table.to_pandas()
	del table
	gc.collect()
	df['eom'] = pd.to_datetime(df['eom'])

	# column level missingness filter, computed on the train period only
	# so no information from val or test leaks into feature selection
	train_mask = df['eom'] <= train_end
	null_rates = df.loc[train_mask, candidate_cols].isnull().mean()
	feature_cols = [c for c in candidate_cols if null_rates[c] <= missing_col_threshold]
	print(f'feature columns retained after missingness filter: {len(feature_cols)} of {len(candidate_cols)}')

	# row level missingness filter on the retained columns
	n_before = len(df)
	row_miss_frac = df[feature_cols].isnull().mean(axis = 1)
	df = df.loc[row_miss_frac <= row_missing_threshold].reset_index(drop = True)
	print(f'rows retained after row missingness filter: {df.shape[0]:,} of {n_before:,}')

	# winsorise the one month return at train period percentiles only,
	# then apply those fixed bounds across the full sample
	ret_train = df.loc[df['eom'] <= train_end, ret_col_1m].dropna()
	lo = float(ret_train.quantile(0.001))
	hi = float(ret_train.quantile(0.999))
	n_clipped = int(((df[ret_col_1m] < lo) | (df[ret_col_1m] > hi)).sum())
	df[ret_col_1m] = df[ret_col_1m].clip(lower = lo, upper = hi)
	print(f'winsorised {ret_col_1m} at [{lo:.4f}, {hi:.4f}] (train-period thresholds), {n_clipped:,} observations clipped')

	print(f'loaded: {df.shape[0]:,} rows, {len(feature_cols)} characteristic columns')
	print(f'date range: {df["eom"].min().date()} to {df["eom"].max().date()}')

	# horizon_months cumulative forward target: compound the next
	# horizon_months one month forward returns per firm, requiring the
	# whole block to be present. No separate clip is applied to the
	# compounded target
	df = df.sort_values(['id', 'eom']).reset_index(drop = True)
	shifted = np.stack([
		df.groupby('id', sort = False)[ret_col_1m].shift(-k).to_numpy(dtype = np.float64)
		for k in range(cfg.horizon_months)
	], axis = 1)
	valid_block = np.isfinite(shifted).all(axis = 1)
	cum = np.where(valid_block, np.prod(1.0 + shifted, axis = 1) - 1.0, np.nan)
	df[ret_col] = cum.astype(np.float32)

	retained = int(np.isfinite(cum).sum())
	print(f'cumulative target retained: {retained:,} of {len(df):,} rows ({100.0 * retained / len(df):.2f}%)')
	del shifted
	gc.collect()

	sorted_eoms = sorted(df['eom'].unique())
	all_months = {}
	n_feat = len(feature_cols)

	for eom in sorted_eoms:
		month = df[df['eom'] == eom]
		month = month[month[ret_col].notna()]
		n = len(month)
		if n < cfg.min_stocks:
			continue

		if n > 1:
			vals_block = month[feature_cols].to_numpy(dtype = np.float64, copy = True)
			col_medians = np.nanmedian(vals_block, axis = 0)
			nan_rows, nan_cols = np.where(np.isnan(vals_block))
			vals_block[nan_rows, nan_cols] = col_medians[nan_cols]
			ranks = rankdata(vals_block, method = 'average', axis = 0) - 1
			x = (ranks / (n - 1) - 0.5).astype(np.float32)
			residual_nan = np.isnan(x)
			if residual_nan.any():
				x[residual_nan] = 0.0
		else:
			x = np.zeros((n, n_feat), dtype = np.float32)

		entry = {
			'ids': month['id'].to_numpy(),
			'r': month[ret_col].to_numpy().astype(np.float64),
			'x': x,
		}
		if store_r1m:
			entry['r1m'] = month[ret_col_1m].to_numpy().astype(np.float64)
		all_months[eom] = entry

	sorted_dates = sorted(all_months.keys())
	avg_firms = np.mean([len(m['ids']) for m in all_months.values()])
	print(f'processed: {len(sorted_dates)} months, avg {avg_firms:.0f} firms/month')

	return feature_cols, all_months, sorted_dates


def make_splits(sorted_dates, train_end, val_end):
	train_dates = [d for d in sorted_dates if d <= train_end]
	val_dates = [d for d in sorted_dates if train_end < d <= val_end]
	test_dates = [d for d in sorted_dates if d > val_end]
	return train_dates, val_dates, test_dates


def stack_months(all_months, dates):
	x = np.vstack([all_months[d]['x'] for d in dates])
	y = np.concatenate([all_months[d]['r'] for d in dates])
	return x, y


def run_mean_split_simulation(predict_fn, month_dates, all_months, cfg):
	"""Rebalance every cfg.rebalance_freq months on a mean split of the
	prediction, with capped softmax weights within each leg. Returns period
	returns, transaction costs, and holdings for both the long short and
	long only books."""
	ls_period_rets, ls_period_dates, ls_tc_history = [], [], []
	lo_period_rets, lo_period_dates, lo_tc_history = [], [], []

	# state for drift-based l1 turnover accounting per leg
	prev_long_ids = prev_long_w = prev_long_realised = None
	prev_short_ids = prev_short_w = prev_short_realised = None
	prev_lo_ids = prev_lo_w = prev_lo_realised = None

	ls_holdings, lo_holdings = [], []
	rb_counter = -1

	for pos, eom in enumerate(month_dates):
		if pos % cfg.rebalance_freq != 0 or eom not in all_months:
			continue
		m = all_months[eom]
		ids, r, x = m['ids'], m['r'], m['x']

		if len(ids) < cfg.min_stocks:
			continue

		pred = predict_fn(x)
		valid_pred = np.isfinite(pred)
		if valid_pred.sum() < cfg.min_stocks:
			continue
		valid_ret = np.isfinite(r)

		legs = mean_split_legs(pred, valid_pred, valid_ret, cfg.min_leg_stocks)
		lo_idx = long_only_leg(valid_pred, valid_ret, cfg.min_leg_stocks)
		if legs is None or lo_idx is None:
			continue
		mean_score, long_idx, short_idx = legs
		rb_counter += 1

		# long_idx and short_idx are already restricted to firms with a
		# valid prediction and a valid realised return, so weights below
		# sum to one over exactly the firms held, with no renormalisation
		# step that could reintroduce a position above the cap
		long_w = capped_softmax_weights(pred[long_idx] - mean_score, cfg.max_position_weight)
		short_w = capped_softmax_weights(mean_score - pred[short_idx], cfg.max_position_weight)

		long_ids_list = ids[long_idx].tolist()
		short_ids_list = ids[short_idx].tolist()
		long_realised = {int(ids[fi]): float(r[fi]) for fi in long_idx}
		short_realised = {int(ids[fi]): float(r[fi]) for fi in short_idx}
		ls_ret = float(np.dot(long_w, r[long_idx])) - float(np.dot(short_w, r[short_idx]))

		d_long_ids, d_long_w = drift_weights(prev_long_ids, prev_long_w, prev_long_realised)
		d_short_ids, d_short_w = drift_weights(prev_short_ids, prev_short_w, prev_short_realised)
		lt = weight_l1_turnover(d_long_ids, d_long_w, long_ids_list, long_w)
		st = weight_l1_turnover(d_short_ids, d_short_w, short_ids_list, short_w)

		ls_period_rets.append(ls_ret)
		ls_period_dates.append(eom)
		ls_tc_history.append((lt + st) * cfg.tc_bps / 10000.0)
		prev_long_ids, prev_long_w, prev_long_realised = long_ids_list, long_w, long_realised
		prev_short_ids, prev_short_w, prev_short_realised = short_ids_list, short_w, short_realised

		lo_ids_list = ids[lo_idx].tolist()
		lo_w = capped_softmax_weights(pred[lo_idx], cfg.max_position_weight)
		lo_realised = {int(ids[fi]): float(r[fi]) for fi in lo_idx}
		lo_ret = float(np.dot(lo_w, r[lo_idx]))

		d_lo_ids, d_lo_w = drift_weights(prev_lo_ids, prev_lo_w, prev_lo_realised)
		lo_turn = weight_l1_turnover(d_lo_ids, d_lo_w, lo_ids_list, lo_w)

		lo_period_rets.append(lo_ret)
		lo_period_dates.append(eom)
		lo_tc_history.append(lo_turn * cfg.tc_bps / 10000.0)
		prev_lo_ids, prev_lo_w, prev_lo_realised = lo_ids_list, lo_w, lo_realised

		for i, fi in enumerate(long_idx):
			ls_holdings.append({
				'rebalance_index': rb_counter, 'eom': eom, 'leg': 'long',
				'id': int(ids[fi]), 'weight': float(long_w[i]), 'realised_return': float(r[fi]),
			})
		for i, fi in enumerate(short_idx):
			ls_holdings.append({
				'rebalance_index': rb_counter, 'eom': eom, 'leg': 'short',
				'id': int(ids[fi]), 'weight': float(-short_w[i]), 'realised_return': float(r[fi]),
			})
		for i, fi in enumerate(lo_idx):
			lo_holdings.append({
				'rebalance_index': rb_counter, 'eom': eom, 'leg': 'long',
				'id': int(ids[fi]), 'weight': float(lo_w[i]), 'realised_return': float(r[fi]),
			})

	return {
		'long_short': {
			'returns': np.array(ls_period_rets), 'tc': np.array(ls_tc_history),
			'dates': ls_period_dates, 'holdings_df': pd.DataFrame(ls_holdings),
		},
		'long_only': {
			'returns': np.array(lo_period_rets), 'tc': np.array(lo_tc_history),
			'dates': lo_period_dates, 'holdings_df': pd.DataFrame(lo_holdings),
		},
	}


def _drift_weight_dict(weight_dict, id_to_r1m):
	growth = {fid: w * (1.0 + id_to_r1m.get(fid, 0.0)) for fid, w in weight_dict.items()}
	g_sum = sum(growth.values())
	return {fid: v / g_sum for fid, v in growth.items()} if g_sum > 1e-12 else growth


def _weighted_return_from_dict(weight_dict, id_to_r1m):
	return sum(w * id_to_r1m.get(fid, 0.0) for fid, w in weight_dict.items())


def run_mean_split_simulation_monthly(predict_fn, month_dates, all_months, cfg):
	"""Monthly resolution counterpart to run_mean_split_simulation: targets
	are re-derived every cfg.rebalance_freq months as before, but positions
	drift with realised one month returns in between, giving a monthly
	return series suitable for a monthly volatility overlay."""
	ls_rets, ls_tc, ls_dates, ls_rb_indices = [], [], [], []
	lo_rets, lo_tc, lo_dates, lo_rb_indices = [], [], [], []

	long_weight_dict, short_weight_dict, lo_weight_dict = {}, {}, {}

	for pos, eom in enumerate(month_dates):
		if eom not in all_months:
			continue
		m = all_months[eom]
		ids, r1m = m['ids'], m['r1m']
		valid_r1m = np.isfinite(r1m)

		ls_tc_this = lo_tc_this = 0.0

		if pos % cfg.rebalance_freq == 0:
			pred = predict_fn(m['x'])
			valid_pred = np.isfinite(pred)

			if valid_pred.sum() >= cfg.min_stocks:
				mean_score = float(pred[valid_pred].mean())
				long_idx = np.where((pred > mean_score) & valid_pred)[0]
				short_idx = np.where((pred <= mean_score) & valid_pred)[0]
				lo_idx = np.where(valid_pred)[0]

				# a mean split gives no guarantee on leg size, so a thin or
				# empty leg is skipped rather than collapsing into an
				# uncontrolled single name concentration
				if len(long_idx) >= cfg.min_leg_stocks and len(short_idx) >= cfg.min_leg_stocks:
					new_long_ids = ids[long_idx].tolist()
					new_short_ids = ids[short_idx].tolist()
					new_lo_ids = ids[lo_idx].tolist()

					lw = capped_softmax_weights(pred[long_idx] - mean_score, cfg.max_position_weight)
					sw = capped_softmax_weights(mean_score - pred[short_idx], cfg.max_position_weight)
					low = capped_softmax_weights(pred[lo_idx], cfg.max_position_weight)

					# snapshot the drifted previous weights for turnover
					# before overwriting them with the new targets
					prev_long_ids = list(long_weight_dict.keys()) or None
					prev_long_w = list(long_weight_dict.values()) or None
					prev_short_ids = list(short_weight_dict.keys()) or None
					prev_short_w = list(short_weight_dict.values()) or None
					prev_lo_ids = list(lo_weight_dict.keys()) or None
					prev_lo_w = list(lo_weight_dict.values()) or None

					lt = weight_l1_turnover(prev_long_ids, prev_long_w, new_long_ids, lw)
					st = weight_l1_turnover(prev_short_ids, prev_short_w, new_short_ids, sw)
					lo_turn = weight_l1_turnover(prev_lo_ids, prev_lo_w, new_lo_ids, low)

					ls_tc_this = (lt + st) * cfg.tc_bps / 10000.0
					lo_tc_this = lo_turn * cfg.tc_bps / 10000.0

					long_weight_dict = dict(zip(new_long_ids, lw.tolist()))
					short_weight_dict = dict(zip(new_short_ids, sw.tolist()))
					lo_weight_dict = dict(zip(new_lo_ids, low.tolist()))

					ls_rb_indices.append(len(ls_rets))
					lo_rb_indices.append(len(lo_rets))

		if not long_weight_dict:
			continue

		id_to_r1m = {int(fid): float(r1m[k]) for k, fid in enumerate(ids.tolist()) if valid_r1m[k]}

		long_ret = _weighted_return_from_dict(long_weight_dict, id_to_r1m)
		short_ret = _weighted_return_from_dict(short_weight_dict, id_to_r1m)
		lo_ret = _weighted_return_from_dict(lo_weight_dict, id_to_r1m)

		ls_rets.append(long_ret - short_ret)
		ls_tc.append(ls_tc_this)
		ls_dates.append(eom)

		lo_rets.append(lo_ret)
		lo_tc.append(lo_tc_this)
		lo_dates.append(eom)

		# drift weights forward by this month's realised returns so next
		# month uses buy-and-hold weights rather than the original targets
		long_weight_dict = _drift_weight_dict(long_weight_dict, id_to_r1m)
		short_weight_dict = _drift_weight_dict(short_weight_dict, id_to_r1m)
		lo_weight_dict = _drift_weight_dict(lo_weight_dict, id_to_r1m)

	return {
		'long_short': {'returns': np.array(ls_rets), 'tc': np.array(ls_tc), 'dates': ls_dates, 'rb_indices': ls_rb_indices},
		'long_only': {'returns': np.array(lo_rets), 'tc': np.array(lo_tc), 'dates': lo_dates, 'rb_indices': lo_rb_indices},
	}


def build_diagnostic_rows(model_name, portfolio, scaling, rets, dates, periods_per_year, roll_window, roll_label):
	"""Per-observation cumulative wealth, drawdown, and rolling sharpe over
	the last roll_window observations, at whatever frequency rets/dates are
	sampled at (rebalance periods or months)."""
	rets = np.asarray(rets, dtype = np.float64)
	if len(rets) == 0:
		return []
	cw = np.cumprod(1.0 + rets)
	cw_dd = np.maximum(cw, 1e-12)
	peak = np.maximum.accumulate(np.concatenate(([1.0], cw_dd)))[1:]
	dd = (peak - cw_dd) / peak

	rolling_sharpe = np.full(len(rets), np.nan)
	rolling_ret = np.full(len(rets), np.nan)
	for i in range(roll_window - 1, len(rets)):
		w = rets[i - roll_window + 1:i + 1]
		mu = float(w.mean() * periods_per_year)
		sigma = float(w.std() * np.sqrt(periods_per_year))
		rolling_ret[i] = mu
		if sigma > 1e-12:
			rolling_sharpe[i] = mu / sigma

	sharpe_col = f'rolling_sharpe_{roll_label}'
	ret_col = f'rolling_ann_ret_{roll_label}'
	rows = []
	for i, eom in enumerate(dates):
		rows.append({
			'model': model_name, 'portfolio': portfolio, 'scaling': scaling,
			'eom': pd.Timestamp(eom).strftime('%Y-%m-%d'),
			'return': round(float(rets[i]), 6),
			'cumulative_wealth': round(float(cw[i]), 6),
			'drawdown': round(float(dd[i]), 6),
			sharpe_col: None if np.isnan(rolling_sharpe[i]) else round(float(rolling_sharpe[i]), 4),
			ret_col: None if np.isnan(rolling_ret[i]) else round(float(rolling_ret[i]) * 100, 4),
		})
	return rows


def validation_sharpes(predict_fn, val_dates, all_months, cfg):
	"""Scaled long short and long only sharpe on val_dates, used as the
	optuna objective for every model. A degenerate near zero volatility
	window returns nan from portfolio_metrics, which optuna would mark as a
	failed trial rather than a low scoring one, so nan is floored instead."""
	sim = run_mean_split_simulation(predict_fn, val_dates, all_months, cfg)
	ls, lo = sim['long_short'], sim['long_only']
	if len(ls['returns']) == 0:
		return -999.0, -999.0

	ls_scaled, _, _ = apply_overlay_and_costs(ls['returns'], ls['tc'], cfg.target_vol, cfg.n_vol_periods, cfg.periods_per_year, cfg.max_leverage_long_short)
	lo_scaled, _, _ = apply_overlay_and_costs(lo['returns'], lo['tc'], cfg.target_vol, cfg.n_vol_periods, cfg.periods_per_year, cfg.max_leverage_long_only)
	ls_sharpe = portfolio_metrics(ls_scaled, cfg.periods_per_year).get('sharpe', -999.0)
	lo_sharpe = portfolio_metrics(lo_scaled, cfg.periods_per_year).get('sharpe', -999.0)

	if not np.isfinite(ls_sharpe):
		ls_sharpe = -999.0
	if not np.isfinite(lo_sharpe):
		lo_sharpe = -999.0
	return float(ls_sharpe), float(lo_sharpe)


def evaluate_and_save(predict_fn, name, all_months, sorted_dates, test_dates, results_dir, cfg):
	"""Simulate over the full history so the volatility overlay has warm up
	before the test window, slice to the test window, write returns,
	holdings and predictions to csv, and return test set metrics."""
	sim = run_mean_split_simulation(predict_fn, sorted_dates, all_months, cfg)
	ls, lo = sim['long_short'], sim['long_only']

	ls_scaled_full, ls_unscaled_full, ls_lev = apply_overlay_and_costs(ls['returns'], ls['tc'], cfg.target_vol, cfg.n_vol_periods, cfg.periods_per_year, cfg.max_leverage_long_short)
	lo_scaled_full, lo_unscaled_full, lo_lev = apply_overlay_and_costs(lo['returns'], lo['tc'], cfg.target_vol, cfg.n_vol_periods, cfg.periods_per_year, cfg.max_leverage_long_only)

	test_set = set(test_dates)
	ls_mask = np.array([d in test_set for d in ls['dates']])
	lo_mask = np.array([d in test_set for d in lo['dates']])

	ls_unscaled_test, ls_scaled_test = ls_unscaled_full[ls_mask], ls_scaled_full[ls_mask]
	lo_unscaled_test, lo_scaled_test = lo_unscaled_full[lo_mask], lo_scaled_full[lo_mask]
	ls_dates_test = [d for d, keep in zip(ls['dates'], ls_mask) if keep]
	lo_dates_test = [d for d, keep in zip(lo['dates'], lo_mask) if keep]

	pd.DataFrame({
		'eom': ls_dates_test, 'return_unscaled': ls_unscaled_test,
		'return_scaled': ls_scaled_test, 'leverage': ls_lev[ls_mask],
	}).to_csv(results_dir / f'{name}_returns_long_short.csv', index = False)
	pd.DataFrame({
		'eom': lo_dates_test, 'return_unscaled': lo_unscaled_test,
		'return_scaled': lo_scaled_test, 'leverage': lo_lev[lo_mask],
	}).to_csv(results_dir / f'{name}_returns_long_only.csv', index = False)

	ls['holdings_df'][ls['holdings_df']['eom'].isin(test_set)].to_csv(results_dir / f'{name}_holdings_long_short.csv', index = False)
	lo['holdings_df'][lo['holdings_df']['eom'].isin(test_set)].to_csv(results_dir / f'{name}_holdings_long_only.csv', index = False)
	predict_at_dates(predict_fn, test_dates, all_months).to_csv(results_dir / f'{name}_test_predictions.csv', index = False)

	return {
		'returns_ls_unscaled': ls_unscaled_test, 'returns_ls_scaled': ls_scaled_test,
		'returns_lo_unscaled': lo_unscaled_test, 'returns_lo_scaled': lo_scaled_test,
		'dates_ls': ls_dates_test, 'dates_lo': lo_dates_test,
		'metrics': {
			'long_short_unscaled': portfolio_metrics(ls_unscaled_test, cfg.periods_per_year, dates = ls_dates_test),
			'long_short_scaled': portfolio_metrics(ls_scaled_test, cfg.periods_per_year, dates = ls_dates_test),
			'long_only_unscaled': portfolio_metrics(lo_unscaled_test, cfg.periods_per_year, dates = lo_dates_test),
			'long_only_scaled': portfolio_metrics(lo_scaled_test, cfg.periods_per_year, dates = lo_dates_test),
		},
	}


def round_or_none(x, ndigits):
	if x is None or (isinstance(x, float) and np.isnan(x)):
		return None
	return round(float(x), ndigits)


def strip_per_year(m):
	if not isinstance(m, dict):
		return m
	return {k: v for k, v in m.items() if k != 'per_year'}


def flush_per_year_rows(rows, model_name, portfolio, scaling, metrics):
	py = metrics.get('per_year', {}) if isinstance(metrics, dict) else {}
	for year in sorted(py.keys()):
		ym = py[year]
		rows.append({
			'model': model_name, 'portfolio': portfolio, 'scaling': scaling, 'year': int(year),
			'ann_ret': round(float(ym['ann_ret']) * 100, 4),
			'ann_vol': round(float(ym['ann_vol']) * 100, 4),
			'sharpe': round_or_none(ym['sharpe'], 4),
			'max_dd': round(float(ym['max_dd']) * 100, 4),
			'cum_return': round(float(ym['cum_return']) * 100, 4),
			'n_obs': int(ym['n_obs']),
		})


def hpo_run_or_load(results_dir, label, objective, n_trials, seed, extra_fields = None):
	best_params_path = results_dir / f'{label}_best_params.json'
	study_path = results_dir / f'{label}_optuna_study.pkl'
	trials_path = results_dir / f'{label}_optuna_trials.csv'

	if best_params_path.exists():
		with open(best_params_path) as fh:
			cached = json.load(fh)
		study = None
		if study_path.exists():
			with open(study_path, 'rb') as fh:
				study = pickle.load(fh)
		print(f'{label} hyperparameters loaded from {best_params_path.name}')
		print(f'{label} best val ls sharpe: {cached["best_value"]:.4f}')
		print(f'{label} best params: {cached["best_params"]}')
		return cached, study

	study = optuna.create_study(direction = 'maximize', sampler = optuna.samplers.TPESampler(seed = seed))
	t0 = time.time()
	study.optimize(objective, n_trials = n_trials, show_progress_bar = True)
	hpo_time = time.time() - t0

	cached = {
		'construction': 'mean_split_softmax_cap_6m',
		'best_params': study.best_params,
		'best_value': float(study.best_value),
		'best_trial_number': int(study.best_trial.number),
		'best_trial_user_attrs': dict(study.best_trial.user_attrs),
		'n_trials_completed': sum(1 for t in study.trials if t.state.name == 'COMPLETE'),
		'hpo_time_seconds': float(hpo_time),
	}
	if extra_fields is not None:
		cached.update(extra_fields(study))

	with open(best_params_path, 'w') as fh:
		json.dump(cached, fh, indent = 2, default = float)
	study.trials_dataframe().to_csv(trials_path, index = False)
	with open(study_path, 'wb') as fh:
		pickle.dump(study, fh)

	print(f'{label} best val ls sharpe: {cached["best_value"]:.4f}')
	print(f'{label} best params: {cached["best_params"]}')
	print(f'{label} hpo time: {hpo_time:.1f} s')
	return cached, study
