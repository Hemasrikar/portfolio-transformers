import gc
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")


metadata_cols = [
    'permno', 'permco', 'gvkey', 'iid', 'id', 'date', 'excntry', 'eom',
    'obs_main', 'exch_main', 'common', 'primary_sec', 'source_crsp',
    'size_grp', 'me', 'me_company', 'prc', 'prc_local', 'prc_high',
    'prc_low', 'bidask', 'curcd', 'fx', 'gics', 'naics', 'sic', 'ff49',
    'dolvol', 'shares', 'tvol', 'adjfct', 'comp_tpci', 'crsp_shrcd',
    'comp_exchg', 'crsp_exchcd', 'ret', 'ret_exc', 'ret_local',
    'ret_exc_lead1m', 'ret_lag_dif', 'enterprise_value', 'book_equity',
    'assets', 'sales', 'net_income', 'intrinsic_value',
]


k0_characteristic_list = [
    'market_equity',
    'div1m_me', 'div3m_me', 'div6m_me', 'div12m_me',
    'divspc1m_me', 'divspc12m_me',
    'chcsho_1m', 'chcsho_3m', 'chcsho_6m', 'chcsho_12m',
    'eqnpo_1m', 'eqnpo_3m', 'eqnpo_6m', 'eqnpo_12m',
    'ret_1_0', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'ret_12_1',
    'ret_12_7', 'ret_60_12', 'ret_2_0', 'ret_3_0', 'ret_6_0',
    'ret_9_0', 'ret_12_0', 'ret_18_1', 'ret_24_1', 'ret_24_12',
    'ret_36_1', 'ret_36_12', 'ret_48_1', 'ret_48_12',
    'ret_60_1', 'ret_60_36',
    'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na',
    'seas_6_10an', 'seas_6_10na', 'seas_11_15an', 'seas_11_15na',
    'seas_16_20an', 'seas_16_20na',
    'resff3_6_1', 'resff3_12_1',
    'ivol_capm_21d', 'ivol_capm_252d', 'ivol_capm_60m',
    'ivol_ff3_21d', 'ivol_hxz4_21d',
    'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',
    'rvol_21d', 'rvol_252d', 'rvolhl_21d',
    'rmax1_21d', 'rmax5_21d', 'rmax5_rvol_21d',
    'rskew_21d', 'coskew_21d',
    'beta_60m', 'beta_21d', 'beta_252d',
    'beta_dimson_21d', 'betadown_252d', 'betabab_1260d',
    'ami_126d', 'dolvol_126d', 'dolvol_var_126d',
    'turnover_126d', 'turnover_var_126d',
    'zero_trades_21d', 'zero_trades_126d', 'zero_trades_252d',
    'bidaskhl_21d', 'corr_1260d',
    'prc_highprc_252d',
    'age',
    'aliq_mat',
    'mispricing_mgmt', 'mispricing_perf',
]

forward_horizons = [6]
target_cols = ["target_6m"]


@dataclass
class DataConfig:
    data_path: Path = Path("data/Global Factor_EM.parquet")
    output_dir: Path = Path("data/processed")
    train_path: Path = Path("data/processed/train.parquet")
    val_path: Path = Path("data/processed/val.parquet")
    test_path: Path = Path("data/processed/test.parquet")
    col_metadata_path: Path = Path("data/processed/column_metadata.json")
    country_lookup_path: Path = Path("data/processed/country_lookup.parquet")
    train_end: str = "2015-12-31"
    val_end: str = "2020-12-31"
    missing_col_threshold: float = 0.30


def load_raw_data(path):
    print("Loading raw data")
    df = pd.read_parquet(path)
    for col in ['divspc1m_me', 'divspc12m_me']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    keep_f64 = {'me', 'me_company', 'ret', 'ret_exc', 'ret_local', 'ret_exc_lead1m', 'prc', 'prc_local', 'fx'}
    f32_cols = [c for c in df.select_dtypes('float64').columns if c not in keep_f64]
    df[f32_cols] = df[f32_cols].astype('float32')
    df['eom'] = pd.to_datetime(df['eom'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Shape, {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def sort_panel(df):
    df = df.sort_values(['id', 'eom']).reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=['id', 'eom'], keep='first')
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"Removed {n_dupes:,} duplicate (id, eom) observations")
    print(f"Shape after deduplication, {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def compute_return_targets(df, horizons):
    df = df.copy()
    max_h = max(horizons)
    log_shifts = {s: np.log1p(df.groupby('id')['ret_exc'].shift(-s)) for s in range(1, max_h + 1)}
    for h in horizons:
        col = f'target_{h}m'
        df[col] = np.expm1(sum(log_shifts[s] for s in range(1, h + 1))).astype('float32')
    return df


def classify_characteristics(df, meta_cols, k0_list):
    exclude = set(meta_cols) | {c for c in df.columns if c.startswith('target_')}
    all_chars = [c for c in df.columns if c not in exclude]
    k0_cols = [c for c in k0_list if c in set(all_chars)]
    k1_cols = [c for c in all_chars if c not in set(k0_cols)]
    return k0_cols, k1_cols


def filter_columns_by_missing(df, train_end, k0_cols, k1_cols, threshold):
    train_mask = df['eom'] <= pd.Timestamp(train_end)
    null_rates = df.loc[train_mask, k0_cols + k1_cols].isnull().mean()
    retained_k0 = [c for c in k0_cols if null_rates[c] <= threshold]
    retained_k1 = [c for c in k1_cols if null_rates[c] <= threshold]
    print(f"K0, {len(retained_k0)} retained ({len(k0_cols) - len(retained_k0)} dropped)")
    print(f"K1, {len(retained_k1)} retained ({len(k1_cols) - len(retained_k1)} dropped)")
    return retained_k0, retained_k1


def split_data(df, train_end, val_end):
    t1 = pd.Timestamp(train_end)
    t2 = pd.Timestamp(val_end)
    train = df[df['eom'] <= t1].copy()
    val = df[(df['eom'] > t1) & (df['eom'] <= t2)].copy()
    test = df[df['eom'] > t2].copy()
    for name, split in [('Train', train), ('Val', val), ('Test', test)]:
        print(f"{name}, {split.shape[0]:,} rows, " f"{split['eom'].min().date()} to {split['eom'].max().date()}")
    return train, val, test


def drop_high_missing_rows(df, char_cols, threshold=1 / 3, label=""):
    miss_frac = df[char_cols].isnull().mean(axis=1)
    keep = miss_frac <= threshold
    n_drop = (~keep).sum()
    print(f"{label}, dropped {n_drop:,} rows ({n_drop / len(df):.2%}) " f"with >{threshold:.0%} missing characteristics")
    return df.loc[keep].reset_index(drop=True)


def add_missingness_flags(df, orig_char_cols) -> pd.DataFrame:
    flags = df[orig_char_cols].isnull().astype('float32').rename(columns={c: f'{c}_miss' for c in orig_char_cols})
    return pd.concat([df, flags], axis=1)


def cross_sectional_normalise(df, char_cols, verbose_every=50):
    data = df[char_cols].to_numpy(dtype='float32', na_value=np.nan)
    eoms = df['eom'].to_numpy()
    unique_eoms = np.unique(eoms)
    n_months = len(unique_eoms)
    for i, eom in enumerate(unique_eoms):
        mask = eoms == eom
        xs = data[mask]
        n = xs.shape[0]
        if n == 0:
            continue
        col_medians = np.nanmedian(xs, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(xs))
        xs[nan_rows, nan_cols] = col_medians[nan_cols]
        if n > 1:
            ranks = rankdata(xs, method='average', axis=0) - 1
            xs = (ranks / (n - 1) - 0.5).astype('float32')
        else:
            xs = np.zeros_like(xs)
        data[mask] = xs
        if (i + 1) % verbose_every == 0 or (i + 1) == n_months:
            print(f"{i + 1}/{n_months} months done")

    residual_nan = int(np.isnan(data).sum())
    if residual_nan > 0:
        print(f"Zero-filling {residual_nan:,} residual NaN cells")
        np.nan_to_num(data, nan=0.0, copy=False)

    df[char_cols] = data
    del data
    gc.collect()
    return df


def save_split_parquet(df, name, output_dir):
    path = output_dir / f'{name}.parquet'
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1e6
    print(f"{name}, {df.shape[0]:,} rows x {df.shape[1]} cols, {size_mb:.0f} MB")


def run_preprocessing(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(cfg.data_path)
    df = sort_panel(df)

    ret = df['ret_exc'].dropna()
    print(f"ret_exc diagnostic, count {len(ret):,}, " f"mean {ret.mean():.6f}, std {ret.std():.6f}, " f"p1 {ret.quantile(0.01):.4f}, p50 {ret.quantile(0.50):.4f}, " f"p99 {ret.quantile(0.99):.4f}")
    if ret.abs().median() > 0.5:
        print("Detected percentage form. Rescaling ret_exc by 1/100.")
        df['ret_exc'] = df['ret_exc'] / 100
        ret = df['ret_exc'].dropna()
        print(f"After rescaling, mean {ret.mean():.6f}, std {ret.std():.6f}")

    train_mask_ret = df['eom'] <= pd.Timestamp(cfg.train_end)
    ret_train = df.loc[train_mask_ret, 'ret_exc'].dropna()
    lo = ret_train.quantile(0.001)
    hi = ret_train.quantile(0.999)
    n_clipped = ((df['ret_exc'] < lo) | (df['ret_exc'] > hi)).sum()
    df['ret_exc'] = df['ret_exc'].clip(lo, hi)
    print(f"Winsorised ret_exc at [{lo:.4f}, {hi:.4f}] (train-period thresholds),{n_clipped:,} observations clipped ({n_clipped / len(df):.2%})")

    df = compute_return_targets(df, forward_horizons)
    for h in forward_horizons:
        col = f'target_{h}m'
        data_h = df[col].dropna()
        print(f"{col}, {data_h.count():,} non-null ({data_h.count() / len(df):.1%}), " f"mean {data_h.mean():.4f}, std {data_h.std():.4f}")

    shortest = f'target_{min(forward_horizons)}m'
    check_mean = df[shortest].dropna().abs().mean()
    assert check_mean < 2.0, f"Target sanity check failed: mean |{shortest}| = {check_mean:.2f}. " f"This suggests ret_exc is not in decimal form or contains extreme outliers."

    k0_cols, k1_cols = classify_characteristics(df, metadata_cols, k0_characteristic_list)
    print(f"K0 (market-based), {len(k0_cols)}")
    print(f"K1 (accounting-based), {len(k1_cols)}")
    print(f"Total characteristics, {len(k0_cols) + len(k1_cols)}")

    retained_k0, retained_k1 = filter_columns_by_missing(df, cfg.train_end, k0_cols, k1_cols, cfg.missing_col_threshold)
    rejected = (set(k0_cols) - set(retained_k0)) | (set(k1_cols) - set(retained_k1))
    df = df.drop(columns=list(rejected), errors='ignore')
    print(f"Dropped {len(rejected)} columns. Current shape, {df.shape[1]} columns")

    orig_char_cols = retained_k0 + retained_k1
    all_char_cols = orig_char_cols

    print(f"Total characteristic columns, {len(all_char_cols)}")
    print(f"K0 current, {len(retained_k0)},K1 current, {len(retained_k1)}")

    train, val, test = split_data(df, cfg.train_end, cfg.val_end)
    del df
    gc.collect()

    train = drop_high_missing_rows(train, orig_char_cols, threshold=1 / 3, label="Train")
    val = drop_high_missing_rows(val, orig_char_cols, threshold=1 / 3, label="Val")
    test = drop_high_missing_rows(test, orig_char_cols, threshold=1 / 3, label="Test")

    train = add_missingness_flags(train, orig_char_cols)
    val = add_missingness_flags(val, orig_char_cols)
    test = add_missingness_flags(test, orig_char_cols)
    print(f"Missingness flags added, {len([c for c in train.columns if c.endswith('_miss')])} columns")

    print("Normalising training set")
    train = cross_sectional_normalise(train, all_char_cols)
    gc.collect()
    print("Normalising validation set")
    val = cross_sectional_normalise(val, all_char_cols)
    gc.collect()
    print("Normalising test set")
    test = cross_sectional_normalise(test, all_char_cols)
    gc.collect()

    save_split_parquet(train, 'train', cfg.output_dir)
    save_split_parquet(val, 'val', cfg.output_dir)
    save_split_parquet(test, 'test', cfg.output_dir)

    all_excntry = pd.concat([train[['id', 'eom', 'excntry']], val[['id', 'eom', 'excntry']], test[['id', 'eom', 'excntry']]], ignore_index=True).drop_duplicates()

    country_codes = sorted(all_excntry['excntry'].dropna().unique().tolist())
    country_to_id = {c: i for i, c in enumerate(country_codes)}
    all_excntry['country_id'] = all_excntry['excntry'].map(country_to_id).astype('Int16')
    country_lookup_out = all_excntry[['id', 'eom', 'country_id']].dropna(subset=['country_id'])
    country_lookup_out.to_parquet(cfg.country_lookup_path, index=False)
    print(f"Country lookup saved, {cfg.country_lookup_path},{len(country_lookup_out):,} rows, {len(country_codes)} countries")
    del all_excntry, country_lookup_out
    gc.collect()

    col_metadata = {
        'retained_k0': retained_k0,
        'retained_k1': retained_k1,
        'orig_char_cols': orig_char_cols,
        'all_char_cols': all_char_cols,
        'country_to_id': country_to_id,
        'country_codes': country_codes,
    }
    with open(cfg.col_metadata_path, 'w') as f:
        json.dump(col_metadata, f, indent=2)

    print(f"Column metadata saved, {cfg.col_metadata_path}")
    print(f"Countries ({len(country_codes)}), {', '.join(country_codes)}")
    del train, val, test
    gc.collect()


if __name__ == "__main__":
    cfg = DataConfig()
    required_outputs = [cfg.train_path, cfg.val_path, cfg.test_path, cfg.col_metadata_path, cfg.country_lookup_path]
    if not all(p.exists() for p in required_outputs):
        print("Processed data not found. Running preprocessing pipeline.")
        run_preprocessing(cfg)
    else:
        print("Processed data found. Skipping preprocessing.")
