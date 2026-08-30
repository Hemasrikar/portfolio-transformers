# tests whether path 2 of the dual path transformer adds information beyond
# path 1. for every month, base_6m and path2_6m are regressed against
# target_6m, standardised within the month, once alone and once together.
# monthly estimates are pooled with fama-macbeth averaging and newey-west
# standard errors (5 lags, for the six month target's overlap). 

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load
from scipy.stats import norm
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_cols = ["target_6m"]


@dataclass
class Config:
    results_dir: Path = Path("results1/transformer")
    analysis_dir: Path = Path("results1/transformer/analysis")
    val_path: Path = Path("data/processed/val.parquet")
    test_path: Path = Path("data/processed/test.parquet")
    col_metadata_path: Path = Path("data/processed/column_metadata.json")
    country_lookup_path: Path = Path("data/processed/country_lookup.parquet")
    mlp_results_dir: Path = Path("results/benchmark/mlp_benchmark")

    d_model: int = 64
    d_ff: int = 128
    dropout: float = 0.1
    ple_num_bins: int = 16
    periodic_num_freq: int = 32

    n_mlp_layers: int = 2
    path2_n_layers: int = 2
    path2_n_heads: int = 1
    path2_d_ff: int = 256
    min_firms_attention: int = 10

    min_firms_regression: int = 30
    min_firms_country: int = 20
    min_country_months: int = 12
    newey_west_lags: int = 5

    encoding_variant: str = "linear"
    seed: int = 24


cfg = Config()
cfg.analysis_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)


with open(cfg.col_metadata_path, "r") as f:
    col_meta = json.load(f)

k0_chars = col_meta["retained_k0"]
k1_chars = col_meta["retained_k1"]
id_to_country = {v: k for k, v in col_meta.get("country_to_id", {}).items()}

parquet_schema_cols = set(pq.read_schema(cfg.test_path).names)

k0_feature_cols = [c for c in k0_chars if c in parquet_schema_cols]
k1_feature_cols = [c for c in k1_chars if c in parquet_schema_cols]

k0_miss_cols = [f"{c}_miss" for c in k0_chars if f"{c}_miss" in parquet_schema_cols]
k1_miss_cols = [f"{c}_miss" for c in k1_chars if f"{c}_miss" in parquet_schema_cols]

country_lookup_df = pd.read_parquet(cfg.country_lookup_path)
country_lookup_df["eom"] = pd.to_datetime(country_lookup_df["eom"])

print(f"K0 characteristics, {len(k0_chars)}")
print(f"K1 characteristics, {len(k1_chars)}")


class CrossSectionalDataset(Dataset):

    def __init__(self, df, k0_cols, k1_cols, k0_miss_cols, k1_miss_cols, target_col_list, country_lookup):
        dates = sorted(df["eom"].unique())
        self.monthly_data = []

        df = df.merge(country_lookup, on=["id", "eom"], how="left")
        df["country_id"] = df["country_id"].fillna(-1).astype(np.int16)

        for date in dates:
            group = df[df["eom"] == date]

            k0 = torch.tensor(group[k0_cols].values, dtype=torch.float32)
            k1 = torch.tensor(group[k1_cols].values, dtype=torch.float32)
            k0_m = torch.tensor(group[k0_miss_cols].values, dtype=torch.float32)
            k1_m = torch.tensor(group[k1_miss_cols].values, dtype=torch.float32)
            cids = torch.tensor(group["country_id"].values, dtype=torch.long)

            targets = {}
            valid_masks = {}
            for tc in target_col_list:
                vals = group[tc].values.copy().astype(np.float32)
                valid_mask = ~np.isnan(vals)
                vals[~valid_mask] = 0.0
                targets[tc] = torch.tensor(vals, dtype=torch.float32)
                valid_masks[tc] = torch.tensor(valid_mask, dtype=torch.bool)

            self.monthly_data.append(
                {
                    "k0": k0,
                    "k1": k1,
                    "k0_miss": k0_m,
                    "k1_miss": k1_m,
                    "country_ids": cids,
                    "eom": pd.Timestamp(date),
                    "targets": targets,
                    "valid_masks": valid_masks,
                    "n_firms": len(group),
                }
            )

        del df

    def __len__(self):
        return len(self.monthly_data)

    def __getitem__(self, idx):
        return self.monthly_data[idx]


def load_dataset(path, k0_cols, k1_cols, k0_miss, k1_miss, target_col_list, country_lookup):
    available = set(pq.read_schema(path).names)
    required = ["id", "eom"] + k0_cols + k1_cols + k0_miss + k1_miss + target_col_list
    load_cols = [c for c in required if c in available]
    df = pd.read_parquet(path, columns=load_cols)
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if col not in df.columns:
            df[col] = 0.0
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)
    # match the winsorisation band eval_dual_path.py applies to target_6m so
    # the regression is run on the same realised return series the reported
    # portfolio simulations use
    if "target_6m" in df.columns:
        df["target_6m"] = df["target_6m"].clip(lower=-2.0, upper=2.0)
    return CrossSectionalDataset(df, k0_cols, k1_cols, k0_miss, k1_miss, target_col_list, country_lookup)


class GRN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model * 2)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        gated = self.fc2(h)
        value, gate = gated.chunk(2, dim=-1)
        h = value * torch.sigmoid(gate)
        return self.layer_norm(residual + h)


class LinearEncoder(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights + self.biases


class PerFeatureTokeniser(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(n_features, d_model))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x.unsqueeze(-1) * self.weights + self.biases)


class PiecewiseLinearEncoder(nn.Module):
    boundaries: torch.Tensor

    def __init__(self, n_features, d_model, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        boundaries = torch.linspace(-0.5, 0.5, num_bins + 1)
        self.register_buffer("boundaries", boundaries)
        self.feature_weights = nn.Parameter(torch.randn(n_features, num_bins, d_model) * 0.02)

    def _encode_bins(self, x):
        t_lower = self.boundaries[:-1]
        t_upper = self.boundaries[1:]
        x_exp = x.unsqueeze(-1)
        activations = torch.clamp((x_exp - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0)
        return activations

    def forward(self, x):
        bin_act = self._encode_bins(x)
        return torch.einsum("bnk,nkd->bnd", bin_act, self.feature_weights)


class PeriodicEncoder(nn.Module):
    def __init__(self, n_features, d_model, num_freq=32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.phi = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.proj = nn.Linear(num_freq, d_model)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        sinusoidal = torch.sin(x_exp * self.omega.unsqueeze(0) + self.phi.unsqueeze(0))
        return self.proj(sinusoidal)


class FourierEncoder(nn.Module):
    def __init__(self, n_features, d_model, num_freq=32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.proj = nn.Linear(num_freq * 2, d_model)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        scaled = x_exp * self.omega.unsqueeze(0)
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(features)


def build_encoder(variant, n_features, d_model, ple_bins=16, periodic_freq=32):
    if variant == "linear":
        return LinearEncoder(n_features, d_model)
    elif variant == "per_feature":
        return PerFeatureTokeniser(n_features, d_model)
    elif variant == "ple":
        return PiecewiseLinearEncoder(n_features, d_model, num_bins=ple_bins)
    elif variant == "periodic":
        return PeriodicEncoder(n_features, d_model, num_freq=periodic_freq)
    elif variant == "fourier":
        return FourierEncoder(n_features, d_model, num_freq=periodic_freq)
    else:
        raise ValueError(f"Unknown encoding variant: {variant}")


class AttentionHead(nn.Module):
    def __init__(self, d_model, init_scale):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model, d_model) * init_scale)
        self.v = nn.Parameter(torch.randn(d_model, d_model) * init_scale)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, y):
        scores = (y @ self.w @ y.t()) * self.scale
        attn = F.softmax(scores, dim=-1)
        return attn @ (y @ self.v), attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        init_scale = 1.0 / d_model
        self.heads = nn.ModuleList([AttentionHead(d_model, init_scale) for _ in range(n_heads)])
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff) * (1.0 / d_ff))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model) * init_scale)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        head_outputs = []
        head_weights = []
        for head in self.heads:
            out, attn = head(x)
            head_outputs.append(out)
            head_weights.append(attn)
        attn_out = torch.stack(head_outputs, dim=0).sum(dim=0)
        y = attn_out + x
        ffn_out = F.relu(y @ self.w1 + self.b1) @ self.w2 + self.b2
        y = ffn_out + y
        return y, torch.stack(head_weights, dim=0)


class AttentiveAggregation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.miss_penalty = nn.Parameter(torch.tensor(5.0))
        self.scale = math.sqrt(d_model)

    def forward(self, encoded, miss_mask=None):
        scores = (encoded * self.query).sum(dim=-1) / self.scale
        if miss_mask is not None:
            penalty = self.miss_penalty.clamp(min=0.0, max=20.0)
            scores = scores - penalty * miss_mask
        weights = F.softmax(scores, dim=1)
        token = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return token, weights


class FirmScoreHead(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, dropout):
        super().__init__()
        modules: list[nn.Module] = [nn.LayerNorm(d_model)]
        for i in range(n_layers):
            in_dim = d_model if i == 0 else d_ff
            modules.extend([nn.Linear(in_dim, d_ff), nn.ELU(), nn.Dropout(dropout)])
        final_in = d_ff if n_layers > 0 else d_model
        modules.append(nn.Linear(final_in, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, z):
        return self.net(z).squeeze(-1)


class Path2Transformer(nn.Module):
    def __init__(self, d_in, n_heads, n_layers, d_ff):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_in, n_heads, d_ff) for _ in range(n_layers)])
        self.lam = nn.Parameter(torch.randn(d_in) * (1.0 / d_in))

    def forward(self, x):
        y = x
        all_attn = []
        for block in self.blocks:
            y, attn_w = block(y)
            all_attn.append(attn_w)
        return y @ self.lam, all_attn


class DualPathTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_k0 = len(k0_chars)
        n_k1 = len(k1_chars)

        self.k0_encoder = build_encoder(config.encoding_variant, n_k0, config.d_model, ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq)
        self.k1_encoder = build_encoder(config.encoding_variant, n_k1, config.d_model, ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq)

        self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)
        self.k1_static_emb = nn.Parameter(torch.randn(n_k1, config.d_model) * 0.02)

        self.k0_agg = AttentiveAggregation(config.d_model)
        self.k1_agg = AttentiveAggregation(config.d_model)

        self.base_head_6m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)

        self.path2_net = Path2Transformer(n_k0 + n_k1, config.path2_n_heads, config.path2_n_layers, config.path2_d_ff)

        self.min_firms = config.min_firms_attention

    def _encode_firms(self, k0, k1, k0_miss, k1_miss):
        k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
        k0_token, k0_weights = self.k0_agg(k0_encoded, k0_miss)
        k1_encoded = self.k1_encoder(k1) + self.k1_static_emb.unsqueeze(0)
        k1_token, k1_weights = self.k1_agg(k1_encoded, k1_miss)
        z = k0_token + k1_token
        agg_info = {"k0_weights": k0_weights, "k1_weights": k1_weights}
        return z, agg_info

    def forward(self, k0, k1, k0_miss, k1_miss, country_ids):
        z, agg_info = self._encode_firms(k0, k1, k0_miss, k1_miss)

        base_6m = self.base_head_6m(z)

        x2 = torch.cat([k0, k1], dim=-1)
        path2_6m = torch.zeros_like(base_6m)
        all_attn = []

        for cid in country_ids.unique():
            mask = country_ids == cid
            if mask.sum() < self.min_firms:
                continue
            score_c, attn_c = self.path2_net(x2[mask])
            path2_6m[mask] = score_c
            all_attn.extend(attn_c)

        return {"scores_6m": base_6m + path2_6m, "base_6m": base_6m, "path2_6m": path2_6m, "attn": all_attn, "agg": agg_info}


def _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key="scores_6m"):
    return torch.stack([m(k0, k1, k0_miss, k1_miss, cids)[key] for m in models]).mean(dim=0)


def significance_stars(p):
    p = np.nan if p is None else p
    bins = [-np.inf, 0.001, 0.01, 0.05, 0.1, np.inf]
    labels = ["***", "**", "*", ".", " "]
    stars = pd.cut([p], bins=bins, labels=labels)[0]
    return stars if pd.notna(stars) else " "


def implied_std_error(mean, t_stat):
    if t_stat is None or np.isnan(t_stat) or abs(t_stat) < 1e-12:
        return float("nan")
    return abs(mean / t_stat)


def analytic_sharpe_se(sharpe, n_obs):
    if sharpe is None or n_obs is None or np.isnan(sharpe) or n_obs < 2:
        return float("nan")
    return math.sqrt((1.0 + 0.5 * sharpe * sharpe) / n_obs)


def sharpe_diff_ztest(sharpe_a, se_a, sharpe_b, se_b):
    if any(v is None or np.isnan(v) for v in (sharpe_a, se_a, sharpe_b, se_b)):
        return float("nan"), float("nan"), float("nan"), float("nan")
    se_diff = math.sqrt(se_a * se_a + se_b * se_b)
    if se_diff < 1e-12:
        return sharpe_a - sharpe_b, se_diff, float("nan"), float("nan")
    z = (sharpe_a - sharpe_b) / se_diff
    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    return sharpe_a - sharpe_b, se_diff, z, p_value


def newey_west_tstat(values, dates, max_lag_months):
    # lag here is a real calendar gap in months, not an array position, since
    # a regime filter such as vol_high can drop months unevenly and leave the
    # remaining months several calendar months apart despite sitting next to
    # each other in the filtered array. for a contiguous monthly series this
    # gives the same answer as a position based newey west correction, so one
    # function covers both the window and the regime splits correctly.
    x = np.asarray(values, dtype=np.float64)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    mask = ~np.isnan(x)
    x = x[mask]
    dates = dates[mask].reset_index(drop=True)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    order = np.argsort(dates.values)
    x = x[order]
    months = (dates.dt.year.values[order] * 12 + dates.dt.month.values[order]).astype(np.int64)

    mean_x = float(x.mean())
    demeaned = x - mean_x
    long_run_var = float((demeaned * demeaned).sum() / n)

    gap = months[:, None] - months[None, :]
    for lag in range(1, max_lag_months + 1):
        a, b = np.where(gap == lag)
        if len(a) == 0:
            continue
        gamma_lag = float((demeaned[a] * demeaned[b]).sum() / n)
        weight = 1.0 - lag / (max_lag_months + 1)
        long_run_var += 2.0 * weight * gamma_lag

    se = math.sqrt(max(long_run_var, 0.0) / n)
    if se <= 1e-12:
        return mean_x, float("nan"), float("nan")
    t_stat = mean_x / se
    p_value = 2.0 * (1.0 - student_t.cdf(abs(t_stat), df=n - 1))
    return mean_x, t_stat, p_value


def cross_sectional_regression(target, base, path2, valid, min_firms):
    valid_idx = np.where(valid)[0]
    n = len(valid_idx)
    if n < min_firms:
        return None

    y = target[valid_idx].astype(np.float64)
    b = base[valid_idx].astype(np.float64)
    p2 = path2[valid_idx].astype(np.float64)

    b_std = (b - b.mean()) / (b.std() + 1e-12)
    p2_std = (p2 - p2.mean()) / (p2.std() + 1e-12)

    pearson_corr = float((b_std * p2_std).sum() / n)
    rank_corr = float(spearmanr(b, p2).statistic)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return None

    x_base = np.column_stack([np.ones(n), b_std])
    coef_base, _, _, _ = np.linalg.lstsq(x_base, y, rcond=None)
    ss_res_base = float(((y - x_base @ coef_base) ** 2).sum())
    r2_base = 1.0 - ss_res_base / ss_tot

    x_full = np.column_stack([np.ones(n), b_std, p2_std])
    coef_full, _, _, _ = np.linalg.lstsq(x_full, y, rcond=None)
    ss_res_full = float(((y - x_full @ coef_full) ** 2).sum())
    r2_full = 1.0 - ss_res_full / ss_tot

    return {
        "n_firms": n,
        "b1_base": float(coef_base[1]),
        "r2_base": r2_base,
        "b1_full": float(coef_full[1]),
        "b2_full": float(coef_full[2]),
        "r2_full": r2_full,
        "corr_base_path2": pearson_corr,
        "rank_corr_base_path2": rank_corr,
    }


def _attention_diagnostics(models, k0, k1, k0_miss, k1_miss, country_ids, min_firms):
    # recomputes path 2's own blocks per qualifying country to get its
    # attention directly. entropy_ratio is a firm's attention entropy over
    # entropy under uniform attention, near 1 means path 2 is close to
    # averaging over peers. missingness_excess is the attention weighted
    # peer missingness rate minus the unweighted average, positive means
    # path 2 leans toward peers with more imputed values.
    x2 = torch.cat([k0, k1], dim=-1)
    n_k0_cols = k0_miss.shape[1]
    n_k1_cols = k1_miss.shape[1]
    miss_rate = ((k0_miss.sum(dim=1) + k1_miss.sum(dim=1)) / (n_k0_cols + n_k1_cols)).cpu().numpy()

    entropy_ratios = []
    missingness_excess = []

    for cid in country_ids.unique():
        mask = country_ids == cid
        n_firms_c = int(mask.sum().item())
        if n_firms_c < min_firms:
            continue

        idx = torch.nonzero(mask, as_tuple=True)[0].cpu().numpy()
        y_input = x2[mask]

        attn_stack = []
        for model in models:
            y = y_input
            attn_last = None
            for block in model.path2_net.blocks:
                y, attn_w = block(y)
                attn_last = attn_w
            attn_stack.append(attn_last.mean(dim=0))
        attn_avg = torch.stack(attn_stack).mean(dim=0).cpu().numpy()

        n = attn_avg.shape[0]
        log_n = math.log(n)
        peer_miss = miss_rate[idx]
        uniform_baseline = float(peer_miss.mean())

        attn_clipped = np.clip(attn_avg, 1e-12, 1.0)
        row_entropy = -(attn_clipped * np.log(attn_clipped)).sum(axis=1)
        entropy_ratios.extend((row_entropy / log_n).tolist())

        weighted_miss = attn_avg @ peer_miss
        missingness_excess.extend((weighted_miss - uniform_baseline).tolist())

    if not entropy_ratios:
        return None
    return {"entropy_ratio": float(np.mean(entropy_ratios)), "missingness_excess": float(np.mean(missingness_excess))}


def print_coefficient_table(title, rows):
    print(title)
    print(f"{'':<26}{'estimate':>12}{'std. error':>12}{'t value':>10}{'pr(>|t|)':>12}")
    for row in rows:
        stars = significance_stars(row["p_value"])
        est = "NA" if np.isnan(row["estimate"]) else f"{row['estimate']:.5f}"
        se = "NA" if np.isnan(row["std_error"]) else f"{row['std_error']:.5f}"
        tval = "NA" if np.isnan(row["t_value"]) else f"{row['t_value']:.3f}"
        pval = "NA" if np.isnan(row["p_value"]) else f"{row['p_value']:.4f}"
        print(f"{row['label']:<26}{est:>12}{se:>12}{tval:>10}{pval:>12} {stars}")
    print("signif. codes, 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    print()


def pooled_stats(sub, lags):
    if len(sub) == 0:
        return None

    def nw(col, offset=0.0):
        mean, t, p = newey_west_tstat(sub[col].values - offset, sub["eom"], lags)
        return mean + offset, t, p

    b1_base, t_b1_base, p_b1_base = nw("b1_base")
    b1_full, t_b1_full, p_b1_full = nw("b1_full")
    b2_full, t_b2_full, p_b2_full = nw("b2_full")
    entropy_ratio, t_entropy, p_entropy = nw("entropy_ratio", offset=1.0)
    missingness_excess, t_missingness, p_missingness = nw("missingness_excess")
    rc_base, t_rc_base, p_rc_base = nw("rank_corr_base")
    rc_path2, t_rc_path2, p_rc_path2 = nw("rank_corr_path2")
    rc_combined, t_rc_combined, p_rc_combined = nw("rank_corr_combined")

    return {
        "n_months": len(sub),
        "b1_base": b1_base, "t_b1_base": t_b1_base, "p_b1_base": p_b1_base,
        "b1_full": b1_full, "t_b1_full": t_b1_full, "p_b1_full": p_b1_full,
        "b2_full": b2_full, "t_b2_full": t_b2_full, "p_b2_full": p_b2_full,
        "r2_base": float(sub["r2_base"].mean()), "r2_full": float(sub["r2_full"].mean()),
        "corr_base_path2": float(sub["corr_base_path2"].mean()),
        "entropy_ratio": entropy_ratio, "t_entropy": t_entropy, "p_entropy": p_entropy,
        "missingness_excess": missingness_excess, "t_missingness": t_missingness, "p_missingness": p_missingness,
        "rc_base": rc_base, "t_rc_base": t_rc_base, "p_rc_base": p_rc_base,
        "rc_path2": rc_path2, "t_rc_path2": t_rc_path2, "p_rc_path2": p_rc_path2,
        "rc_combined": rc_combined, "t_rc_combined": t_rc_combined, "p_rc_combined": p_rc_combined,
    }


all_results = {}
variant_seed_paths = {}
seed_pattern = re.compile(r"^metrics_(?P<variant>.+)_seed(?P<seed>\d+)\.json$")
for metrics_path in sorted(cfg.results_dir.glob("metrics_*_seed*.json")):
    m = seed_pattern.match(metrics_path.name)
    if not m:
        continue
    variant_name = m.group("variant")
    seed_idx = int(m.group("seed"))
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    if variant_name not in all_results:
        all_results[variant_name] = metrics
    variant_seed_paths.setdefault(variant_name, []).append((seed_idx, cfg.results_dir / f"weights_{variant_name}_seed{seed_idx}.safetensors"))

if not all_results:
    raise RuntimeError(f"No seeded metrics files found in {cfg.results_dir}. " f"Expected files of the form metrics_{{variant}}_seed{{i}}.json.")

for variant_name in variant_seed_paths:
    variant_seed_paths[variant_name].sort(key=lambda x: x[0])

val_ds = load_dataset(cfg.val_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df)
test_ds = load_dataset(cfg.test_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df)
print(f"Validation months, {len(val_ds)}")
print(f"Test months, {len(test_ds)}")

# validation immediately precedes test with no calendar gap, so concatenating
# the two month lists in this order keeps the pooled series in chronological
# order, which the newey west lag correction depends on. pooling gives the
# regression roughly twice the observations of test alone, at the cost of
# validation not being fully held out, since it drove early stopping and, in
# eval_dual_path.py, variant selection. results are still reported per split
# further down so that caveat can be checked rather than assumed away.
pooled_months = [(m, "val") for m in val_ds.monthly_data] + [(m, "test") for m in test_ds.monthly_data]
print(f"Pooled months, validation plus test, {len(pooled_months)}")

# regime labels, computed once from target_6m alone before any model runs,
# so they cannot depend on which variant is being scored. vol_regime is a
# median split on that month's cross sectional dispersion of target_6m,
# a turbulence proxy that does not require an external market index.
# period is a plain chronological tercile, to catch a result that is
# carried by a handful of months rather than holding across the window.
month_order = [batch["eom"].strftime("%Y-%m-%d") for batch, _ in pooled_months]
month_dispersion = {}
for batch, _ in pooled_months:
    target = batch["targets"]["target_6m"].numpy()
    valid = batch["valid_masks"]["target_6m"].numpy()
    eom = batch["eom"].strftime("%Y-%m-%d")
    month_dispersion[eom] = float(target[valid].std()) if valid.sum() > 1 else float("nan")

dispersion_median = float(np.nanmedian(list(month_dispersion.values())))
vol_regime_by_month = {eom: ("high" if d >= dispersion_median else "low") for eom, d in month_dispersion.items()}

tercile = len(month_order) // 3
period_by_month = {}
for i, eom in enumerate(month_order):
    period_by_month[eom] = "early" if i < tercile else ("mid" if i < 2 * tercile else "late")

monthly_rows = []
country_rows = []

for variant_name in all_results:
    cfg.encoding_variant = variant_name

    stored_cfg = all_results[variant_name].get("config", {})
    for field in ("d_model", "d_ff", "dropout", "n_mlp_layers", "path2_n_layers", "path2_n_heads", "path2_d_ff", "periodic_num_freq", "ple_num_bins", "min_firms_attention"):
        if field in stored_cfg:
            setattr(cfg, field, stored_cfg[field])

    variant_model_list = []
    for seed_idx, weights_path in variant_seed_paths[variant_name]:
        m = DualPathTransformer(cfg).to(device)
        m.load_state_dict(safetensors_load(str(weights_path)))
        m.eval()
        variant_model_list.append(m)

    print(f"Variant, {variant_name}, seeds, {len(variant_model_list)}")

    with torch.no_grad():
        for batch, split_label in pooled_months:
            k0 = batch["k0"].to(device)
            k1 = batch["k1"].to(device)
            k0_miss = batch["k0_miss"].to(device)
            k1_miss = batch["k1_miss"].to(device)
            cids = batch["country_ids"].to(device)

            base_scores = _ensemble_score(variant_model_list, k0, k1, k0_miss, k1_miss, cids, key="base_6m").cpu().numpy()
            path2_scores = _ensemble_score(variant_model_list, k0, k1, k0_miss, k1_miss, cids, key="path2_6m").cpu().numpy()
            combined_scores = base_scores + path2_scores
            target = batch["targets"]["target_6m"].numpy()
            valid = batch["valid_masks"]["target_6m"].numpy()

            result = cross_sectional_regression(target, base_scores, path2_scores, valid, cfg.min_firms_regression)
            if result is None:
                continue

            valid_idx_month = np.where(valid)[0]
            result["rank_corr_base"] = float(spearmanr(base_scores[valid_idx_month], target[valid_idx_month]).statistic)
            result["rank_corr_path2"] = float(spearmanr(path2_scores[valid_idx_month], target[valid_idx_month]).statistic)
            result["rank_corr_combined"] = float(spearmanr(combined_scores[valid_idx_month], target[valid_idx_month]).statistic)

            attn_diag = _attention_diagnostics(variant_model_list, k0, k1, k0_miss, k1_miss, cids, cfg.min_firms_attention)
            result["entropy_ratio"] = attn_diag["entropy_ratio"] if attn_diag is not None else float("nan")
            result["missingness_excess"] = attn_diag["missingness_excess"] if attn_diag is not None else float("nan")

            result["variant"] = variant_name
            result["eom"] = batch["eom"].strftime("%Y-%m-%d")
            result["split"] = split_label
            result["vol_regime"] = vol_regime_by_month[result["eom"]]
            result["period"] = period_by_month[result["eom"]]
            monthly_rows.append(result)

            cids_np = cids.cpu().numpy()
            country_threshold = max(cfg.min_firms_country, cfg.min_firms_attention)
            for country_id in np.unique(cids_np):
                if country_id < 0:
                    continue
                country_valid = valid & (cids_np == country_id)
                if country_valid.sum() < country_threshold:
                    continue
                result_c = cross_sectional_regression(target, base_scores, path2_scores, country_valid, country_threshold)
                if result_c is None:
                    continue
                idx_c = np.where(country_valid)[0]
                result_c["rank_corr_base"] = float(spearmanr(base_scores[idx_c], target[idx_c]).statistic)
                result_c["rank_corr_path2"] = float(spearmanr(path2_scores[idx_c], target[idx_c]).statistic)
                result_c["rank_corr_combined"] = float(spearmanr(combined_scores[idx_c], target[idx_c]).statistic)
                result_c["entropy_ratio"] = float("nan")
                result_c["missingness_excess"] = float("nan")
                result_c["variant"] = variant_name
                result_c["eom"] = result["eom"]
                result_c["country"] = id_to_country.get(int(country_id), str(int(country_id)))
                country_rows.append(result_c)

    for m in variant_model_list:
        del m

variant_monthly_df = pd.DataFrame(monthly_rows)
country_monthly_df = pd.DataFrame(country_rows)

summary_rows = []
country_summary_rows = []

for variant_name in all_results:
    sub_variant = variant_monthly_df[variant_monthly_df["variant"] == variant_name]
    windows = {
        "val": sub_variant[sub_variant["split"] == "val"],
        "test": sub_variant[sub_variant["split"] == "test"],
        "pooled": sub_variant,
    }
    stats = {window: pooled_stats(sub, cfg.newey_west_lags) for window, sub in windows.items()}
    if stats["pooled"] is None:
        continue

    for window, s in stats.items():
        if s is not None:
            summary_rows.append({"variant": variant_name, "window": window, **s})

    print(f"{variant_name}")
    rows = [
        {
            "window": window,
            "n": s["n_months"],
            "b1_base": round(s["b1_base"], 4),
            "b1_full": round(s["b1_full"], 4),
            "b2_full": f"{s['b2_full']:.4f} {significance_stars(s['p_b2_full'])}",
            "rc_base": round(s["rc_base"], 4),
            "rc_combined": round(s["rc_combined"], 4),
        }
        for window, s in stats.items() if s is not None
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    regime_defs = {
        "vol_high": sub_variant[sub_variant["vol_regime"] == "high"],
        "vol_low": sub_variant[sub_variant["vol_regime"] == "low"],
        "period_early": sub_variant[sub_variant["period"] == "early"],
        "period_mid": sub_variant[sub_variant["period"] == "mid"],
        "period_late": sub_variant[sub_variant["period"] == "late"],
    }
    regime_stats = {label: pooled_stats(sub, cfg.newey_west_lags) for label, sub in regime_defs.items()}
    for label, s in regime_stats.items():
        if s is not None:
            summary_rows.append({"variant": variant_name, "window": label, **s})

    regime_rows = [
        {
            "regime": label,
            "n": s["n_months"],
            "b1_base": round(s["b1_base"], 4),
            "b2_full": f"{s['b2_full']:.4f} {significance_stars(s['p_b2_full'])}",
            "rc_base": round(s["rc_base"], 4),
            "rc_combined": round(s["rc_combined"], 4),
        }
        for label, s in regime_stats.items() if s is not None
    ]
    print(pd.DataFrame(regime_rows).to_string(index=False))
    print()

    country_sub = country_monthly_df[country_monthly_df["variant"] == variant_name]
    country_stats = {}
    for country, sub_c in country_sub.groupby("country"):
        s = pooled_stats(sub_c, cfg.newey_west_lags)
        if s is not None and s["n_months"] >= cfg.min_country_months:
            country_stats[country] = s
            country_summary_rows.append({"variant": variant_name, "country": country, **s})

    if country_stats:
        country_table_rows = [
            {
                "country": country,
                "n": s["n_months"],
                "b1_base": round(s["b1_base"], 4),
                "b2_full": f"{s['b2_full']:.4f} {significance_stars(s['p_b2_full'])}",
                "rc_base": round(s["rc_base"], 4),
                "rc_combined": round(s["rc_combined"], 4),
            }
            for country, s in sorted(country_stats.items(), key=lambda kv: kv[1]["n_months"], reverse=True)
        ]
        print(f"{variant_name}, by country, pooled, at least {cfg.min_country_months} months")
        print(pd.DataFrame(country_table_rows).to_string(index=False))
    else:
        print(f"{variant_name}, no country cleared {cfg.min_country_months} pooled months at the {cfg.min_firms_country} firm threshold")
    print()

    pooled = stats["pooled"]
    print_coefficient_table(
        f"{variant_name}, pooled diagnostics",
        [
            {"label": "corr, base_6m vs path2_6m", "estimate": pooled["corr_base_path2"], "std_error": float("nan"), "t_value": float("nan"), "p_value": float("nan")},
            {"label": "entropy ratio vs uniform (1)", "estimate": pooled["entropy_ratio"], "std_error": implied_std_error(pooled["entropy_ratio"] - 1.0, pooled["t_entropy"]), "t_value": pooled["t_entropy"], "p_value": pooled["p_entropy"]},
            {"label": "missingness excess vs zero", "estimate": pooled["missingness_excess"], "std_error": implied_std_error(pooled["missingness_excess"], pooled["t_missingness"]), "t_value": pooled["t_missingness"], "p_value": pooled["p_missingness"]},
            {"label": "r squared, base only", "estimate": pooled["r2_base"], "std_error": float("nan"), "t_value": float("nan"), "p_value": float("nan")},
            {"label": "r squared, full", "estimate": pooled["r2_full"], "std_error": float("nan"), "t_value": float("nan"), "p_value": float("nan")},
        ],
    )

summary_df = pd.DataFrame(summary_rows)
summary_path = cfg.analysis_dir / "score_orthogonality_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"summary written, {summary_path}, {len(summary_df)} rows across the val, test, and pooled windows")

monthly_path = cfg.analysis_dir / "score_orthogonality_monthly.csv"
variant_monthly_df.to_csv(monthly_path, index=False)
print(f"monthly detail written, {monthly_path}")

country_summary_df = pd.DataFrame(country_summary_rows)
country_summary_path = cfg.analysis_dir / "score_orthogonality_country.csv"
country_summary_df.to_csv(country_summary_path, index=False)
print(f"country summary written, {country_summary_path}, {len(country_summary_df)} rows")

country_monthly_path = cfg.analysis_dir / "score_orthogonality_country_monthly.csv"
country_monthly_df.to_csv(country_monthly_path, index=False)
print(f"country monthly detail written, {country_monthly_path}")


# mlp comparison, test window only so it lines up with mlp_summary.csv's own
# test only figure. sharpe still comes from eval_dual_path.py's saved
# metrics_{variant}.json rather than being recomputed here.

def mlp_row(mlp_df, portfolio, scaling):
    if mlp_df is None:
        return None
    sub = mlp_df[(mlp_df["portfolio"] == portfolio) & (mlp_df["scaling"] == scaling)]
    return sub.iloc[0] if len(sub) else None


def dppt_portfolio_metrics(results_dir, variant_name):
    path = results_dir / f"metrics_{variant_name}.json"
    return json.loads(path.read_text()).get("portfolio_metrics") if path.exists() else None


mlp_summary_path = cfg.mlp_results_dir / "mlp_summary.csv"
mlp_df = pd.read_csv(mlp_summary_path) if mlp_summary_path.exists() else None
if mlp_df is None:
    print(f"mlp summary not found at {mlp_summary_path}, skipping the mlp comparison.")

comparison_df = None
if mlp_df is not None:
    mlp_ls = mlp_row(mlp_df, "long_short", "scaled")
    mlp_lo = mlp_row(mlp_df, "long_only", "scaled")
    if mlp_ls is None or mlp_lo is None:
        print(f"{mlp_summary_path} is missing a long_short or long_only scaled row, skipping the mlp comparison.")
    else:
        test_stats = summary_df[summary_df["window"] == "test"].set_index("variant")

        comparison_rows = [
            {
                "model": "mlp benchmark",
                "rank_corr": round(float(mlp_ls["rc_test"]), 4),
                "sharpe_ls": round(float(mlp_ls["sharpe"]), 4),
                "se_ls": round(float(mlp_ls["se"]), 4),
                "sharpe_lo": round(float(mlp_lo["sharpe"]), 4),
                "n_obs_ls": int(mlp_ls["n_obs"]),
            }
        ]
        ztest_rows = []

        for variant_name, row in test_stats.iterrows():
            pm = dppt_portfolio_metrics(cfg.results_dir, variant_name)
            ls_sharpe = float(pm["long_short"]["sharpe_ratio"]) if pm else float("nan")
            ls_n = float(pm["long_short"]["n_rebalances"]) if pm else float("nan")
            lo_sharpe = float(pm["long_only"]["sharpe_ratio"]) if pm else float("nan")
            ls_se = analytic_sharpe_se(ls_sharpe, ls_n)

            for label, rc in [("base_6m only", row["rc_base"]), ("path2_6m only", row["rc_path2"]), ("combined", row["rc_combined"])]:
                is_combined = label == "combined"
                comparison_rows.append(
                    {
                        "model": f"{variant_name}, {label}",
                        "rank_corr": round(rc, 4),
                        "sharpe_ls": round(ls_sharpe, 4) if is_combined and not np.isnan(ls_sharpe) else None,
                        "se_ls": round(ls_se, 4) if is_combined and not np.isnan(ls_se) else None,
                        "sharpe_lo": round(lo_sharpe, 4) if is_combined and not np.isnan(lo_sharpe) else None,
                        "n_obs_ls": int(ls_n) if is_combined and not np.isnan(ls_n) else None,
                    }
                )

            diff, se_diff, z, p = sharpe_diff_ztest(ls_sharpe, ls_se, float(mlp_ls["sharpe"]), float(mlp_ls["se"]))
            ztest_rows.append({"label": f"{variant_name}, combined vs mlp", "estimate": diff, "std_error": se_diff, "t_value": z, "p_value": p})

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df["n_obs_ls"] = comparison_df["n_obs_ls"].astype("Int64")
        print()
        print("model comparison, mlp benchmark vs dppt, test window only")
        print(comparison_df.to_string(index=False))
        print()
        print_coefficient_table("sharpe difference vs mlp benchmark, long short, scaled, z test", ztest_rows)

        comparison_path = cfg.analysis_dir / "score_orthogonality_mlp_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"mlp comparison written, {comparison_path}")
