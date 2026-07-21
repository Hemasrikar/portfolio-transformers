
## Fama-French Five-Factors Benchmark

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')


# configuration
data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/ff_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

# factor name, jkp column, sort direction, tail fraction
factor_defs = {
	'value': ('be_me', False, 0.30),
	'momentum': ('ret_12_1', False, 0.30),
	'profitability': ('ope_be', False, 0.30),
	'investment': ('at_gr1', True, 0.30),
	'size': ('me', True, 0.50),
}
char_map = {name: col for name, (col, _, _) in factor_defs.items()}

ret_col = 'ret_exc_lead1m'
rebalance_freq = 3
tc_bps = 25
min_stocks = 30
ret_clip = 1.0

target_vol = 0.10
vol_lookback_months = 12
max_leverage_ls = 3.0
max_leverage_lo = 3.0

min_history = 60
test_start = pd.Timestamp('2021-01-01')
id_cols = ['id', 'eom', 'excntry', ret_col, 'me']


available = set(pq.read_schema(data_path).names)
fm_available = [c for c in char_map.values() if c in available]
needed = list(dict.fromkeys([c for c in id_cols + fm_available if c in available]))

df = pd.read_parquet(data_path, columns = needed)
df['eom'] = pd.to_datetime(df['eom'])
for col in fm_available:
	if df[col].dtype == np.float64:
		df[col] = df[col].astype(np.float32)
df[ret_col] = df[ret_col].clip(lower = -ret_clip, upper = ret_clip)

print(f'rows loaded, {df.shape[0]:,}')
print(f'columns loaded, {df.shape[1]}')
print(f'date range, {df["eom"].min().date()} to {df["eom"].max().date()}')


## Build monthly cross-sections

def rank_centred(vals):
	# percentile rank in [-0.5, 0.5], zero where the characteristic is missing
	out = np.zeros(len(vals))
	valid = np.isfinite(vals)
	if valid.sum() > 5:
		out[valid] = pd.Series(vals[valid]).rank(pct = True).to_numpy() - 0.5
	return out, valid

all_months = {}
for eom in sorted(df['eom'].unique()):
	month = df[df['eom'] == eom]
	if len(month) < min_stocks:
		continue

	entry = {
		'ids': month['id'].values,
		'r': month[ret_col].values.astype(np.float64),
		'me': month['me'].values.astype(np.float64),
	}
	for name, col in char_map.items():
		entry[name] = (month[col].values.astype(np.float64)
			if col in month.columns else np.full(len(month), np.nan))

	ranked = [rank_centred(month[col].values.astype(np.float64)) for col in fm_available]
	entry['fm_x'] = (np.column_stack([r for r, _ in ranked]) if ranked
		else np.empty((len(month), 0)))
	entry['fm_x_valid'] = (np.column_stack([v for _, v in ranked]) if ranked
		else np.ones((len(month), 0), dtype = bool))
	all_months[eom] = entry

sorted_dates = sorted(all_months.keys())
print(f'processed months, {len(sorted_dates)}')


## Metrics and portfolio construction helpers

def return_stats(rets):
	n = len(rets)
	ann_ret = float(rets.mean() * 12.0)
	ann_vol = float(rets.std() * np.sqrt(12.0))
	sharpe = ann_ret / max(ann_vol, 1e-8)
	wealth = np.cumprod(1.0 + rets)
	peak = np.maximum.accumulate(wealth)
	return {
		'ann_ret': ann_ret,
		'ann_vol': ann_vol,
		'sharpe': sharpe,
		'se_sharpe': float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / n)),
		'max_dd': float(((peak - wealth) / peak).max()),
		'cum_return': float(wealth[-1] - 1.0),
		'n_obs': n,
	}

def portfolio_metrics(rets, dates):
	rets = np.asarray(rets, dtype = np.float64)
	if len(rets) == 0:
		empty = dict.fromkeys(
			['ann_ret', 'ann_vol', 'sharpe', 'se_sharpe', 'max_dd', 'cum_return'], np.nan)
		return {**empty, 'n_obs': 0, 'per_year': {}}

	years = pd.DatetimeIndex(dates).year.to_numpy()
	per_year = {
		int(y): return_stats(rets[years == y])
		for y in sorted(set(years.tolist())) if (years == y).sum() >= 2
	}
	return {**return_stats(rets), 'per_year': per_year}

def apply_vol_target(rets, rb_indices, max_leverage):
	rets = np.asarray(rets, dtype = np.float64)
	scaled = rets.copy()
	for i, rb in enumerate(rb_indices):
		trailing = rets[max(0, rb - vol_lookback_months):rb]
		if len(trailing) < vol_lookback_months:
			continue
		sigma = float(trailing.std() * np.sqrt(12.0))
		lev = float(np.clip(target_vol / max(sigma, 1e-8), 1.0 / max_leverage, max_leverage))
		end = rb_indices[i + 1] if i + 1 < len(rb_indices) else len(rets)
		scaled[rb:end] = rets[rb:end] * lev
	return scaled

def filter_to_test_window(rets, dates):
	dates = pd.DatetimeIndex(dates)
	mask = dates >= test_start
	return np.asarray(rets, dtype = np.float64)[mask], list(dates[mask])

def clean_weights(weight_vals, mask):
	w = np.asarray(weight_vals, dtype = np.float64)[mask]
	return np.where(np.isfinite(w) & (w > 0), w, 0.0)

def weight_map(ids, selected_ids, weight_vals, gross = 1.0):
	if not selected_ids:
		return {}
	ids = np.asarray(ids)
	mask = np.array([sid in selected_ids for sid in ids.tolist()])
	w = clean_weights(weight_vals, mask)
	if w.sum() <= 0:
		w = np.ones(mask.sum())
	return {sid: float(gross * wi / w.sum()) for sid, wi in zip(ids[mask].tolist(), w)}

def turnover(new_weights, old_weights):
	ids = set(new_weights) | set(old_weights)
	return float(sum(abs(new_weights.get(i, 0.0) - old_weights.get(i, 0.0)) for i in ids))

def portfolio_return(ids, rets, selected_ids, weight_vals):
	if not selected_ids:
		return 0.0
	ids = np.asarray(ids)
	mask = np.array([sid in selected_ids for sid in ids.tolist()]) & np.isfinite(rets)
	if mask.sum() == 0:
		return 0.0
	w = clean_weights(weight_vals, mask)
	if w.sum() <= 0:
		return float(rets[mask].mean())
	return float((w / w.sum() * rets[mask]).sum())

def rebalance_set(dates):
	if not dates:
		return set()
	start = pd.Timestamp(dates[0])
	months = lambda d: (pd.Timestamp(d).year - start.year) * 12 + pd.Timestamp(d).month - start.month
	return {d for d in dates if months(d) % rebalance_freq == 0}


## Backtest engine, shared by the sorted factor and Fama-MacBeth portfolios

def run_backtest(dates, month_fn, tail_frac = 0.30, reverse = False):
	rset = rebalance_set(dates)
	ls_rets, lo_rets, out_dates, rb_indices = [], [], [], []
	long_ids, short_ids = set(), set()
	prev_ls, prev_lo = {}, {}

	for eom in dates:
		ids, r, me, signal = month_fn(eom)
		if signal is None:
			continue
		ls_cost = lo_cost = 0.0

		if eom in rset:
			rb_indices.append(len(out_dates))
			valid = np.isfinite(signal)
			if valid.sum() < 10:
				# too few names to sort, exit the book and pay the exit cost
				ls_rets.append(-turnover({}, prev_ls) * tc_bps / 10000.0)
				lo_rets.append(-turnover({}, prev_lo) * tc_bps / 10000.0)
				out_dates.append(eom)
				prev_ls, prev_lo = {}, {}
				long_ids, short_ids = set(), set()
				continue

			vi, vc = ids[valid], signal[valid]
			nq = max(1, int(len(vi) * tail_frac))
			order = np.argsort(vc)
			top = set(vi[order[::-1][:nq]].tolist())
			bottom = set(vi[order[:nq]].tolist())
			long_ids, short_ids = (bottom, top) if reverse else (top, bottom)

			long_w = weight_map(ids, long_ids, me, gross = 1.0)
			ls_w = {**long_w, **weight_map(ids, short_ids, me, gross = -1.0)}
			ls_cost = turnover(ls_w, prev_ls) * tc_bps / 10000.0
			lo_cost = turnover(long_w, prev_lo) * tc_bps / 10000.0
			prev_ls, prev_lo = ls_w, long_w

		if not long_ids:
			continue
		long_ret = portfolio_return(ids, r, long_ids, me)
		short_ret = portfolio_return(ids, r, short_ids, me)
		ls_rets.append(long_ret - short_ret - ls_cost)
		lo_rets.append(long_ret - lo_cost)
		out_dates.append(eom)

	return {
		'long_short': np.array(ls_rets),
		'long_only': np.array(lo_rets),
		'dates': out_dates,
		'rb_indices': rb_indices,
	}


## Result registry, every series is stored once and reused for all outputs

series = []

def register(strategy, portfolio, rets, dates, rb_indices, max_leverage):
	scaled = apply_vol_target(rets, rb_indices, max_leverage)
	unscaled_test, dates_test = filter_to_test_window(rets, dates)
	scaled_test, _ = filter_to_test_window(scaled, dates)
	for scaling, r in [('unscaled', unscaled_test), ('scaled', scaled_test)]:
		series.append({
			'strategy': strategy,
			'portfolio': portfolio,
			'scaling': scaling,
			'returns': r,
			'dates': dates_test,
			'metrics': portfolio_metrics(r, dates_test),
		})


## Market portfolio, value weighted and long only by construction

market_rets, market_dates = [], []
for eom in sorted_dates:
	m = all_months[eom]
	valid_me = np.isfinite(m['me']) & (m['me'] > 0)
	if valid_me.sum() < 5 or (valid_me & np.isfinite(m['r'])).sum() < 5:
		continue
	market_rets.append(portfolio_return(m['ids'], m['r'], set(m['ids'][valid_me].tolist()), m['me']))
	market_dates.append(eom)

# the market book is reweighted every month, so every month is a rebalance date.
# the vol overlay uses the same trailing window as the factor portfolios so that
# the scaled column stays comparable across rows.
register('market_value_weighted', 'long_only', np.array(market_rets), market_dates,
	list(range(len(market_rets))), max_leverage_lo)


## Sorted factor portfolios

for name, (col, reverse, tail_frac) in factor_defs.items():
	month_fn = lambda eom, name = name: (
		all_months[eom]['ids'], all_months[eom]['r'],
		all_months[eom]['me'], all_months[eom].get(name),
	)
	sim = run_backtest(sorted_dates, month_fn, tail_frac = tail_frac, reverse = reverse)
	if len(sim['dates']) == 0:
		print(f'factor, {name}, no data')
		continue
	# the overlay runs on the full sample so the trailing estimator is warmed up
	# before the test window opens
	register(name, 'long_short', sim['long_short'], sim['dates'], sim['rb_indices'], max_leverage_ls)
	register(name, 'long_only', sim['long_only'], sim['dates'], sim['rb_indices'], max_leverage_lo)


## Fama-MacBeth cross-sectional regression

fm_betas, fm_dates_used = [], []
for eom in sorted_dates:
	m = all_months[eom]
	valid = np.isfinite(m['r']) & np.isfinite(m['fm_x']).all(axis = 1)
	if valid.sum() < len(fm_available) + 5:
		continue
	x = np.column_stack([np.ones(valid.sum()), m['fm_x'][valid]])
	try:
		fm_betas.append(np.linalg.lstsq(x, m['r'][valid], rcond = None)[0])
		fm_dates_used.append(eom)
	except np.linalg.LinAlgError:
		continue

n_coef = 1 + len(fm_available)
fm_betas = (np.array(fm_betas, dtype = np.float64) if fm_betas
	else np.empty((0, n_coef), dtype = np.float64))
n_months_fm = len(fm_betas)

if n_months_fm > 1:
	fm_mean = fm_betas.mean(axis = 0)
	fm_se = fm_betas.std(axis = 0, ddof = 1) / np.sqrt(n_months_fm)
	fm_tstat = fm_mean / np.maximum(fm_se, 1e-10)
	fm_pval = 2.0 * (1.0 - stats.t.cdf(np.abs(fm_tstat), df = n_months_fm - 1))
else:
	fm_mean = fm_betas.mean(axis = 0) if n_months_fm else np.full(n_coef, np.nan)
	fm_se = fm_tstat = fm_pval = np.full(n_coef, np.nan)

def stars(p):
	return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

fm_coef_table = [
	{
		'variable': name,
		'mean_coef': round(float(fm_mean[i]), 5),
		'se': round(float(fm_se[i]), 5),
		't_stat': round(float(fm_tstat[i]), 4),
		'p_value': round(float(fm_pval[i]), 4),
		'sig': stars(fm_pval[i]),
	}
	for i, name in enumerate(['intercept'] + fm_available)
]

print(f'Fama-MacBeth regression, {n_months_fm} months, {len(fm_available)} characteristics')
print(pd.DataFrame(fm_coef_table).to_string(index = False))


## Fama-MacBeth predictive portfolio

fm_predictions = {}
for t in range(min_history, len(fm_dates_used)):
	beta = fm_betas[:t].mean(axis = 0)
	eom = fm_dates_used[t]
	m = all_months[eom]
	pred = beta[0] + m['fm_x'] @ beta[1:]
	valid = np.isfinite(pred) & m['fm_x_valid'].all(axis = 1)
	if valid.sum() < 10:
		continue
	fm_predictions[eom] = {
		'w': pred[valid],
		'ids': m['ids'][valid],
		'me': m['me'][valid],
		'r': m['r'][valid],
	}

# rank correlation is computed on the test window only, matching the metric basis
fm_corrs = []
for eom, p in fm_predictions.items():
	if eom < test_start:
		continue
	valid = np.isfinite(p['w']) & np.isfinite(p['r'])
	if valid.sum() < 10:
		continue
	c = float(np.asarray(spearmanr(p['w'][valid], p['r'][valid]))[0])
	if np.isfinite(c):
		fm_corrs.append(float(c))
fm_rc = float(np.mean(fm_corrs)) if fm_corrs else 0.0

fm_keys = sorted(fm_predictions.keys())
fm_sim = run_backtest(
	fm_keys,
	lambda eom: (fm_predictions[eom]['ids'], fm_predictions[eom]['r'],
		fm_predictions[eom]['me'], fm_predictions[eom]['w']),
	tail_frac = 0.30,
)
register('fm_regression', 'long_short', fm_sim['long_short'], fm_sim['dates'],
	fm_sim['rb_indices'], max_leverage_ls)
register('fm_regression', 'long_only', fm_sim['long_only'], fm_sim['dates'],
	fm_sim['rb_indices'], max_leverage_lo)

print(f'FM out-of-sample months, {len(fm_predictions)}')
print(f'FM rank correlation, {fm_rc:.4f}')


## Save results

by_key = {(s['strategy'], s['portfolio'], s['scaling']): s for s in series}

for s in series:
	np.save(results_dir / f'{s["strategy"]}_{s["portfolio"]}_{s["scaling"]}.npy', s['returns'])

pd.DataFrame(fm_betas, columns = ['intercept'] + fm_available,
	index = pd.DatetimeIndex(fm_dates_used)).to_csv(results_dir / 'fm_monthly_betas.csv')

summary_rows = [
	{
		'strategy': s['strategy'],
		'portfolio': s['portfolio'],
		'scaling': s['scaling'],
		'sharpe': round(s['metrics']['sharpe'], 4),
		'se': round(s['metrics']['se_sharpe'], 4),
		'ann_ret': round(s['metrics']['ann_ret'] * 100, 2),
		'ann_vol': round(s['metrics']['ann_vol'] * 100, 2),
		'cum_return': round(s['metrics']['cum_return'] * 100, 2),
		'max_dd': round(s['metrics']['max_dd'] * 100, 2),
		'n_obs': s['metrics']['n_obs'],
	}
	for s in series
]
summary_table = pd.DataFrame(summary_rows)
summary_table.to_csv(results_dir / 'fama_french_summary.csv', index = False)

per_year_rows = [
	{
		'strategy': s['strategy'], 'portfolio': s['portfolio'], 'scaling': s['scaling'],
		'year': year,
		'ann_ret': round(ym['ann_ret'] * 100, 4),
		'ann_vol': round(ym['ann_vol'] * 100, 4),
		'sharpe': round(ym['sharpe'], 4),
		'max_dd': round(ym['max_dd'] * 100, 4),
		'cum_return': round(ym['cum_return'] * 100, 4),
		'n_obs': ym['n_obs'],
	}
	for s in series
	for year, ym in sorted(s['metrics']['per_year'].items())
]
pd.DataFrame(per_year_rows).to_csv(results_dir / 'ff_per_year_metrics.csv', index = False)

def monthly_rows(s):
	rets = s['returns']
	if len(rets) == 0:
		return []
	wealth = np.cumprod(1.0 + rets)
	drawdown = (np.maximum.accumulate(wealth) - wealth) / np.maximum.accumulate(wealth)
	roll_ret = np.full(len(rets), np.nan)
	roll_sharpe = np.full(len(rets), np.nan)
	for i in range(11, len(rets)):
		window = rets[i - 11:i + 1]
		mu = float(window.mean() * 12.0)
		sigma = float(window.std() * np.sqrt(12.0))
		roll_ret[i] = mu
		if sigma > 1e-12:
			roll_sharpe[i] = mu / sigma

	rows = []
	for i, eom in enumerate(s['dates']):
		rows.append({
			'strategy': s['strategy'], 'portfolio': s['portfolio'], 'scaling': s['scaling'],
			'eom': pd.Timestamp(eom).strftime('%Y-%m-%d'),
			'return': round(float(rets[i]), 6),
			'cumulative_wealth': round(float(wealth[i]), 6),
			'drawdown': round(float(drawdown[i]), 6),
			'rolling_sharpe_12m': None if np.isnan(roll_sharpe[i]) else round(float(roll_sharpe[i]), 4),
			'rolling_return_12m': None if np.isnan(roll_ret[i]) else round(float(roll_ret[i]) * 100, 4),
		})
	return rows

per_month_df = pd.DataFrame([row for s in series for row in monthly_rows(s)])
per_month_df.to_csv(results_dir / 'ff_per_month_metrics.csv', index = False)

summary = {
	'test_start': str(test_start.date()),
	'fm_characteristics': fm_available,
	'strategies': {
		f'{s["strategy"]}_{s["portfolio"]}_{s["scaling"]}':
			{k: v for k, v in s['metrics'].items() if k != 'per_year'}
		for s in series
	},
	'fm_regression': {
		'n_months': n_months_fm,
		'characteristics': fm_available,
		'coefficients': fm_coef_table,
		'rank_corr': fm_rc,
		'n_oos_months': len(fm_predictions),
	},
}
with open(results_dir / 'ff_summary.json', 'w') as f:
	json.dump(summary, f, indent = 2, default = float)

print('Fama-French benchmark, EM universe, unscaled and vol targeted')
print(summary_table.to_string(index = False))
print(f'saved, {len(per_month_df)} monthly rows and {len(per_year_rows)} annual rows')
