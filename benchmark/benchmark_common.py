import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# shared across all benchmark scripts so a fix made here applies everywhere
# rather than being duplicated and risking silent divergence between files


def portfolio_metrics(rets, periods_per_year, dates=None, vol_floor=1e-8, sharpe_vol_floor=1e-12):
    rets = np.asarray(rets, dtype=np.float64)
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
            if y_vol > sharpe_vol_floor:
                y_sharpe = y_ret / max(y_vol, vol_floor)
            else:
                y_sharpe = float('nan')
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
    """Softmax weights capped at max_weight via exact water filling.

    Weights are proportional to exp(scores) within the uncapped set, and
    the algorithm is exact rather than iterative refinement: once a weight
    is fixed at max_weight it is never reconsidered, so remaining mass is
    always reallocated only among the weights not yet at the cap. This
    terminates in at most n passes and respects the cap exactly, up to
    floating point precision, for every leg size where capping is
    mechanically feasible."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if max_weight <= 1.0 / n + 1e-12:
        return np.full(n, 1.0 / n, dtype=np.float64)

    z = scores - scores.max()
    w = np.exp(z)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = w / s

    fixed = np.zeros(n, dtype=bool)
    result = np.zeros(n, dtype=np.float64)

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
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return weights
    valid_total = float(weights[valid].sum())
    if valid_total <= 1e-12:
        return weights
    out = np.zeros_like(weights)
    out[valid] = weights[valid] / valid_total
    return out


def mean_split_legs(pred, valid_pred, valid_ret, min_leg_stocks):
    """Form the long and short leg index sets for a mean split.

    Both legs are restricted to firms with a valid prediction and a valid
    realised return before any weight is computed, so a later renormalisation
    over valid entries is never needed and cannot reintroduce a position
    above the cap. Returns None if either leg falls below min_leg_stocks,
    since a mean split gives no guarantee on leg size and a thin leg turns
    into an uncontrolled concentration once capped softmax weighting falls
    back to equal weighting."""
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
    curr_map = {}
    for j in range(len(curr_ids)):
        fid = int(curr_ids[j]) if not hasattr(curr_ids[j], 'item') else int(curr_ids[j].item())
        curr_map[fid] = float(curr_w[j])
    if prev_ids is None or prev_w is None or len(prev_w) == 0:
        return float(sum(abs(v) for v in curr_map.values()))
    prev_map = {}
    for j in range(len(prev_ids)):
        fid = int(prev_ids[j]) if not hasattr(prev_ids[j], 'item') else int(prev_ids[j].item())
        prev_map[fid] = float(prev_w[j])
    all_ids = set(prev_map.keys()) | set(curr_map.keys())
    return float(sum(
        abs(curr_map.get(fid, 0.0) - prev_map.get(fid, 0.0)) for fid in all_ids
    ))


def drift_weights(prev_ids, prev_w, realised_returns_by_id):
    if prev_ids is None or prev_w is None or len(prev_w) == 0:
        return None, None
    n = len(prev_w)
    ids_list = []
    growth = np.zeros(n, dtype=np.float64)
    for j in range(n):
        fid = int(prev_ids[j]) if not hasattr(prev_ids[j], 'item') else int(prev_ids[j].item())
        ids_list.append(fid)
        growth[j] = float(prev_w[j]) * (1.0 + float(realised_returns_by_id.get(fid, 0.0)))
    g_sum = float(growth.sum())
    if g_sum > 1e-12:
        drifted = growth / g_sum
    else:
        drifted = growth
    return ids_list, drifted


def apply_period_vol_overlay(period_rets, target_vol, n_vol_periods, periods_per_year, max_leverage):
    period_rets = np.asarray(period_rets, dtype=np.float64)
    n = len(period_rets)
    leverage = np.ones(n, dtype=np.float64)
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
    leg_gross_rets = np.asarray(leg_gross_rets, dtype=np.float64)
    leg_tc = np.asarray(leg_tc, dtype=np.float64)
    leverage_path = apply_period_vol_overlay(leg_gross_rets, target_vol, n_vol_periods, periods_per_year, max_leverage)
    unscaled_net = leg_gross_rets - leg_tc
    scaled_net = leverage_path * leg_gross_rets - leverage_path * leg_tc
    return scaled_net, unscaled_net, leverage_path


def apply_vol_target_monthly(monthly_rets, rebalance_indices, target_vol, lookback_months, max_leverage):
    """Monthly resolution counterpart to apply_period_vol_overlay, used for
    diagnostics that require a trailing window measured in individual
    monthly observations rather than in rebalance periods. The two overlays
    are intentionally different estimators, one uses lookback_months
    monthly observations, the other uses n_vol_periods rebalance period
    observations, and should not be assumed interchangeable."""
    monthly_rets = np.asarray(monthly_rets, dtype=np.float64)
    n = len(monthly_rets)
    leverage = np.ones(n, dtype=np.float64)
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


def rank_correlation_oos(predict_fn, month_dates, all_months, min_cross_section=10):
    corrs = []
    for eom in month_dates:
        if eom not in all_months:
            continue
        m = all_months[eom]
        pred = predict_fn(m['x'])
        valid = np.isfinite(pred) & np.isfinite(m['r'])
        if valid.sum() < min_cross_section:
            continue
        result = spearmanr(pred[valid], m['r'][valid])
        c = result.statistic
        if not np.isnan(c):
            corrs.append(float(c))
    return float(np.mean(corrs)) if corrs else 0.0
