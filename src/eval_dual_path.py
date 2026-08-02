import gc
import json
import math
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
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_cols = ["target_1m", "target_6m"]


@dataclass
class Config:
    results_dir: Path = Path("results1/transformer")
    val_path: Path = Path("data/processed/val.parquet")
    test_path: Path = Path("data/processed/test.parquet")
    col_metadata_path: Path = Path("data/processed/column_metadata.json")
    country_lookup_path: Path = Path("data/processed/country_lookup.parquet")

    d_model: int = 64
    d_ff: int = 128
    dropout: float = 0.1
    ple_num_bins: int = 16
    periodic_num_freq: int = 32

    n_mlp_layers: int = 2
    path2_n_layers: int = 2
    path2_n_heads: int = 1
    path2_d_ff: int = 256
    lambda_aux: float = 0.3
    lambda_aux2: float = 0.3
    min_firms_attention: int = 10

    target_vol: float = 0.10
    vol_lookback_months: int = 36
    max_leverage_long_only: float = 3.0
    max_leverage_long_short: float = 3.0
    min_firms_country: int = 20
    max_position_weight: float = 0.05
    rebalance_freq: int = 3
    tc_bps: int = 25

    encoding_variant: str = "linear"
    seed: int = 24


cfg = Config()

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)


with open(cfg.col_metadata_path, "r") as f:
    col_meta = json.load(f)

k0_chars = col_meta["retained_k0"]
k1_chars = col_meta["retained_k1"]

parquet_schema_cols = set(pq.read_schema(cfg.test_path).names)

k0_feature_cols = [c for c in k0_chars if c in parquet_schema_cols]
k1_feature_cols = [c for c in k1_chars if c in parquet_schema_cols]

k0_miss_cols = [f"{c}_miss" for c in k0_chars if f"{c}_miss" in parquet_schema_cols]
k1_miss_cols = [f"{c}_miss" for c in k1_chars if f"{c}_miss" in parquet_schema_cols]

country_lookup_df = pd.read_parquet(cfg.country_lookup_path)
country_lookup_df["eom"] = pd.to_datetime(country_lookup_df["eom"])

country_to_id = col_meta["country_to_id"]
country_codes = col_meta["country_codes"]

print(f"K0 characteristics, {len(k0_chars)}")
print(f"K1 characteristics, {len(k1_chars)}")
print(f"K0 missingness flags, {len(k0_miss_cols)}")
print(f"K1 missingness flags, {len(k1_miss_cols)}")
print(f"Countries, {len(country_codes)}")


class CrossSectionalDataset(Dataset):

    def __init__(self, df, k0_cols, k1_cols, k0_miss_cols, k1_miss_cols, target_col_list, country_lookup, has_market_cap=False):
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
            firm_ids = torch.tensor(group["id"].values, dtype=torch.long)

            if has_market_cap:
                market_cap = torch.tensor(group["me"].values, dtype=torch.float32)
            else:
                market_cap = torch.ones(len(group), dtype=torch.float32)

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
                    "firm_ids": firm_ids,
                    "eom": pd.Timestamp(date),
                    "market_cap": market_cap,
                    "targets": targets,
                    "valid_masks": valid_masks,
                    "n_firms": len(group),
                }
            )

        del df
        gc.collect()

    def __len__(self):
        return len(self.monthly_data)

    def __getitem__(self, idx):
        return self.monthly_data[idx]


def load_dataset(path, k0_cols, k1_cols, k0_miss, k1_miss, target_col_list, country_lookup):
    available = set(pq.read_schema(path).names)
    alias_1m = "target_1m" in target_col_list and "target_1m" not in available and "ret_exc_lead1m" in available
    required = ["id", "eom"] + k0_cols + k1_cols + k0_miss + k1_miss + target_col_list
    if alias_1m:
        required = [c for c in required if c != "target_1m"] + ["ret_exc_lead1m"]
    has_market_cap = "me" in available
    if has_market_cap:
        required = required + ["me"]
    else:
        print(f"'me' column not found in {path}. Country composite " f"simulation will use firm count weighting.")
    load_cols = [c for c in required if c in available]
    df = pd.read_parquet(path, columns=load_cols)
    if alias_1m:
        df["target_1m"] = df["ret_exc_lead1m"]
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if col not in df.columns:
            df[col] = 0.0
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)
    if has_market_cap and df["me"].isna().any():
        df["me"] = df["me"].fillna(0.0)
    # winsorise the realised returns to the benchmark bands so the transformer
    # is evaluated on the same return series as the benchmark models. the one
    # month return is clipped to [-1, 1] and the compounded six month return to
    # [-2, 2], matching load_universe in benchmark_common (ret_clip_low = -1.0,
    # ret_clip_high = 1.0, and the six month band widened by a factor of two)
    if "target_1m" in df.columns:
        df["target_1m"] = df["target_1m"].clip(lower=-1.0, upper=1.0)
    if "target_6m" in df.columns:
        df["target_6m"] = df["target_6m"].clip(lower=-2.0, upper=2.0)
    return CrossSectionalDataset(df, k0_cols, k1_cols, k0_miss, k1_miss, target_col_list, country_lookup, has_market_cap=has_market_cap)


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


def _renorm_over_valid(weights, valid):
    if isinstance(weights, np.ndarray):
        valid_np = np.asarray(valid, dtype=bool)
        valid_total = float(weights[valid_np].sum())
        if valid_total <= 1e-12:
            return weights
        return weights / valid_total
    valid_t = torch.as_tensor(valid, dtype=torch.bool)
    valid_total = float(weights[valid_t].sum())
    if valid_total <= 1e-12:
        return weights
    return weights / valid_total


def _capped_softmax_weights(scores, max_weight):
    n = scores.shape[0]
    if n == 0:
        return scores.new_zeros(0)
    if max_weight <= 1.0 / n + 1e-12:
        return scores.new_full((n,), 1.0 / n)
    weights = F.softmax(scores, dim=0)
    for _ in range(20):
        over = weights > max_weight
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights = torch.where(over, torch.full_like(weights, max_weight), weights)
        residual = ~over
        residual_total = weights[residual].sum()
        if residual_total <= 1e-12:
            break
        weights = torch.where(residual, weights * (1.0 + excess / residual_total), weights)
    return weights


def _cap_uniform_weights(n, max_weight):
    if n == 0:
        return torch.zeros(0)
    return torch.full((n,), 1.0 / n)


def _weight_l1_turnover(prev_ids, prev_w, curr_ids, curr_w):
    if curr_ids is None or curr_w is None or len(curr_w) == 0:
        return 0.0
    curr_map = {}
    for j in range(len(curr_ids)):
        item = curr_ids[j]
        fid = int(item.item()) if hasattr(item, "item") else int(item)
        wj = curr_w[j]
        curr_map[fid] = float(wj.item()) if hasattr(wj, "item") else float(wj)
    if prev_ids is None or prev_w is None or len(prev_w) == 0:
        return float(sum(abs(v) for v in curr_map.values()))
    prev_map = {}
    for j in range(len(prev_ids)):
        item = prev_ids[j]
        fid = int(item.item()) if hasattr(item, "item") else int(item)
        wj = prev_w[j]
        prev_map[fid] = float(wj.item()) if hasattr(wj, "item") else float(wj)
    all_ids = set(prev_map.keys()) | set(curr_map.keys())
    return float(sum(abs(curr_map.get(fid, 0.0) - prev_map.get(fid, 0.0)) for fid in all_ids))


def _drift_weights(prev_ids, prev_w, realised_returns_by_id):
    if prev_ids is None or prev_w is None or len(prev_w) == 0:
        return None, None
    n = len(prev_w)
    ids_list = []
    growth = np.zeros(n, dtype=np.float64)
    for j in range(n):
        item = prev_ids[j]
        fid = int(item.item()) if hasattr(item, "item") else int(item)
        ids_list.append(fid)
        w = prev_w[j]
        w_val = float(w.item()) if hasattr(w, "item") else float(w)
        r = float(realised_returns_by_id.get(fid, 0.0))
        growth[j] = w_val * (1.0 + r)
    g_sum = float(growth.sum())
    if g_sum > 1e-12:
        drifted = growth / g_sum
    else:
        drifted = growth
    return ids_list, drifted


def _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key="scores_6m"):
    return torch.stack([m(k0, k1, k0_miss, k1_miss, cids)[key] for m in models]).mean(dim=0)


def _seed_vol_history(models, val_dataset, config, rebalance_freq, leg_kind, score_key="scores_6m", target_key="target_6m"):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()
    n_vol_periods = max(1, config.vol_lookback_months // rebalance_freq)
    returns = []
    with torch.no_grad():
        for idx in range(0, len(val_dataset), rebalance_freq):
            batch = val_dataset[idx]
            k0 = batch["k0"].to(device)
            k1 = batch["k1"].to(device)
            k0_miss = batch["k0_miss"].to(device)
            k1_miss = batch["k1_miss"].to(device)
            cids = batch["country_ids"].to(device)
            scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)
            n_firms = scores.shape[0]
            raw = batch["targets"][target_key]
            valid = batch["valid_masks"][target_key]
            raw_np = raw.numpy()
            valid_np = valid.numpy()
            if leg_kind == "long_only":
                # seed with the same capped softmax book the test simulation trades,
                # so the volatility estimate that primes the overlay is measured on the
                # construction it is actually applied to rather than an uncapped proxy
                scores_cpu = scores.cpu()
                w = _capped_softmax_weights(scores_cpu, config.max_position_weight)
                w = _renorm_over_valid(w, valid_np)
                w_np = w.numpy().astype(np.float64)
                long_ret = 0.0
                for fi in range(n_firms):
                    r = float(raw_np[fi]) if valid_np[fi] else 0.0
                    long_ret += w_np[fi] * r
                returns.append(long_ret)
            else:
                # seed the long short overlay with the identical mean split, capped
                # softmax, demeaned construction used at test time, in place of the
                # equal weighted proxy that understated the concentrated book volatility
                scores_cpu = scores.cpu()
                mean_s = scores_cpu.mean()
                long_mask = scores_cpu > mean_s
                short_mask = ~long_mask
                long_idx_np = long_mask.nonzero(as_tuple=True)[0].numpy()
                short_idx_np = short_mask.nonzero(as_tuple=True)[0].numpy()
                long_w = _capped_softmax_weights(scores_cpu[long_mask] - mean_s, config.max_position_weight)
                short_w = _capped_softmax_weights(mean_s - scores_cpu[short_mask], config.max_position_weight)
                long_w = _renorm_over_valid(long_w, valid_np[long_idx_np])
                short_w = _renorm_over_valid(short_w, valid_np[short_idx_np])
                long_w_np = long_w.numpy().astype(np.float64)
                short_w_np = short_w.numpy().astype(np.float64)
                long_ret = 0.0
                for i, fi in enumerate(long_idx_np):
                    r = float(raw_np[fi]) if valid_np[fi] else 0.0
                    long_ret += long_w_np[i] * r
                short_ret = 0.0
                for i, fi in enumerate(short_idx_np):
                    r = float(raw_np[fi]) if valid_np[fi] else 0.0
                    short_ret += short_w_np[i] * r
                returns.append(long_ret - short_ret)
    return returns[-n_vol_periods:] if returns else []


def _seed_vol_history_monthly(models, val_dataset, config, rebalance_freq, leg_kind, score_key="scores_6m"):
    n = max(1, config.vol_lookback_months)
    if leg_kind == "long_only":
        out = portfolio_simulation_monthly(models, val_dataset, config, rebalance_freq=rebalance_freq, tc_bps=config.tc_bps, seed_returns=None, record_holdings=False, score_key=score_key)
    elif leg_kind == "long_short":
        out = portfolio_simulation_long_short_monthly(models, val_dataset, config, rebalance_freq=rebalance_freq, tc_bps=config.tc_bps, seed_returns=None, record_holdings=False, score_key=score_key)
    elif leg_kind == "country_long_only":
        out = portfolio_simulation_country_composite_monthly(
            models, val_dataset, config, rebalance_freq=rebalance_freq, tc_bps=config.tc_bps, long_short=False, seed_returns=None, record_holdings=False, score_key=score_key
        )
    else:
        out = portfolio_simulation_country_composite_monthly(
            models, val_dataset, config, rebalance_freq=rebalance_freq, tc_bps=config.tc_bps, long_short=True, seed_returns=None, record_holdings=False, score_key=score_key
        )
    unscaled = out[1]
    return list(unscaled[-n:]) if len(unscaled) >= n else list(unscaled)


@torch.no_grad()
def portfolio_simulation(models, dataset, config, rebalance_freq=6, tc_bps=25, seed_returns=None, record_holdings=False, score_key="scores_6m", target_key="target_6m") -> tuple:
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    periods_per_year = 12 / rebalance_freq
    portfolio_returns = []
    unscaled_returns = []
    raw_returns_hist = list(seed_returns) if seed_returns else []
    prev_firm_ids_seq = None
    prev_weights_seq = None
    prev_realised_returns_seq = None
    holdings = []
    leverage_trace = []

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        firm_ids = batch["firm_ids"]
        eom_ts = batch["eom"]

        scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)
        n_firms = scores.shape[0]

        weights = _capped_softmax_weights(scores, config.max_position_weight)

        raw_returns = batch["targets"][target_key]
        valid = batch["valid_masks"][target_key]
        weights = _renorm_over_valid(weights, valid)
        leg_return = 0.0
        firm_realised_returns = {}
        weights_np = weights.detach().cpu().numpy().astype(np.float64)
        ids_list = [int(firm_ids[fi].item()) for fi in range(n_firms)]
        for fi in range(n_firms):
            r = float(raw_returns[fi].item()) if valid[fi] else 0.0
            firm_realised_returns[ids_list[fi]] = r
            leg_return += weights_np[fi] * r

        if prev_firm_ids_seq is not None:
            drifted_ids, drifted_w = _drift_weights(prev_firm_ids_seq, prev_weights_seq, prev_realised_returns_seq)
        else:
            drifted_ids, drifted_w = None, None
        base_turnover = _weight_l1_turnover(drifted_ids, drifted_w, ids_list, weights_np)

        n_vol_periods = max(1, config.vol_lookback_months // rebalance_freq)
        if len(raw_returns_hist) >= n_vol_periods:
            recent = np.array(raw_returns_hist[-n_vol_periods:])
            realised_vol = recent.std(ddof=1) * np.sqrt(periods_per_year)
            leverage = config.target_vol / max(realised_vol, 1e-6)
            leverage = float(np.clip(leverage, 1.0 / config.max_leverage_long_only, config.max_leverage_long_only))
        else:
            leverage = 1.0

        flat_tc = base_turnover * tc_bps / 10000.0
        tc = leverage * flat_tc
        portfolio_returns.append(leverage * leg_return - tc)
        unscaled_returns.append(leg_return - flat_tc)
        raw_returns_hist.append(leg_return)

        prev_firm_ids_seq = ids_list
        prev_weights_seq = weights_np
        prev_realised_returns_seq = firm_realised_returns

        if record_holdings:
            rebal_idx = idx // rebalance_freq
            leverage_trace.append({"rebalance_index": rebal_idx, "eom": eom_ts, "portfolio": "long_only", "leverage": float(leverage)})
            for fi in range(n_firms):
                holdings.append(
                    {
                        "rebalance_index": rebal_idx,
                        "eom": eom_ts,
                        "portfolio": "long_only",
                        "leg": "long",
                        "country_id": int(cids[fi].item()),
                        "id": int(firm_ids[fi].item()),
                        "weight": float(weights[fi].item()),
                        "realised_return": (float(raw_returns[fi].item()) if valid[fi] else float("nan")),
                    }
                )

    returns_arr = np.array(portfolio_returns)
    unscaled_arr = np.array(unscaled_returns)
    if record_holdings:
        return returns_arr, unscaled_arr, holdings, leverage_trace
    return returns_arr, unscaled_arr


@torch.no_grad()
def portfolio_simulation_long_short(models, dataset, config, rebalance_freq=6, tc_bps=25, seed_returns=None, record_holdings=False, score_key="scores_6m", target_key="target_6m") -> tuple:
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    periods_per_year = 12 / rebalance_freq
    portfolio_returns = []
    unscaled_returns = []
    raw_ls_returns = list(seed_returns) if seed_returns else []
    prev_long_ids_seq = None
    prev_long_weights_seq = None
    prev_long_realised_returns_seq = None
    prev_short_ids_seq = None
    prev_short_weights_seq = None
    prev_short_realised_returns_seq = None
    holdings = []
    leverage_trace = []

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        firm_ids = batch["firm_ids"]
        eom_ts = batch["eom"]

        scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)

        mean_score = scores.mean()
        long_idx = (scores > mean_score).nonzero(as_tuple=True)[0]
        short_idx = (scores <= mean_score).nonzero(as_tuple=True)[0]
        long_idx_np = long_idx.cpu().numpy()
        short_idx_np = short_idx.cpu().numpy()
        long_firm_ids = firm_ids[long_idx_np]
        short_firm_ids = firm_ids[short_idx_np]

        long_w = _capped_softmax_weights(scores[long_idx] - mean_score, config.max_position_weight)
        short_w = _capped_softmax_weights(mean_score - scores[short_idx], config.max_position_weight)

        raw_returns = batch["targets"][target_key]
        valid = batch["valid_masks"][target_key]
        long_valid = valid[long_idx_np]
        short_valid = valid[short_idx_np]
        long_w = _renorm_over_valid(long_w, long_valid)
        short_w = _renorm_over_valid(short_w, short_valid)
        long_w_np = long_w.detach().cpu().numpy().astype(np.float64)
        short_w_np = short_w.detach().cpu().numpy().astype(np.float64)
        long_ids_list = [int(long_firm_ids[i].item()) for i in range(len(long_firm_ids))]
        short_ids_list = [int(short_firm_ids[i].item()) for i in range(len(short_firm_ids))]
        long_realised = {}
        short_realised = {}
        long_ret = 0.0
        for i, fi in enumerate(long_idx_np):
            r = float(raw_returns[fi].item()) if valid[fi] else 0.0
            long_realised[long_ids_list[i]] = r
            long_ret += long_w_np[i] * r
        short_ret = 0.0
        for i, fi in enumerate(short_idx_np):
            r = float(raw_returns[fi].item()) if valid[fi] else 0.0
            short_realised[short_ids_list[i]] = r
            short_ret += short_w_np[i] * r
        ls_ret = long_ret - short_ret

        if prev_long_ids_seq is not None:
            d_long_ids, d_long_w = _drift_weights(prev_long_ids_seq, prev_long_weights_seq, prev_long_realised_returns_seq)
        else:
            d_long_ids, d_long_w = None, None
        if prev_short_ids_seq is not None:
            d_short_ids, d_short_w = _drift_weights(prev_short_ids_seq, prev_short_weights_seq, prev_short_realised_returns_seq)
        else:
            d_short_ids, d_short_w = None, None
        lt = _weight_l1_turnover(d_long_ids, d_long_w, long_ids_list, long_w_np)
        st = _weight_l1_turnover(d_short_ids, d_short_w, short_ids_list, short_w_np)
        base_turnover = lt + st

        n_vol_periods = max(1, config.vol_lookback_months // rebalance_freq)
        if len(raw_ls_returns) >= n_vol_periods:
            recent = np.array(raw_ls_returns[-n_vol_periods:])
            realised_vol = recent.std(ddof=1) * np.sqrt(periods_per_year)
            leverage = config.target_vol / max(realised_vol, 1e-6)
            leverage = float(np.clip(leverage, 1.0 / config.max_leverage_long_short, config.max_leverage_long_short))
        else:
            leverage = 1.0

        flat_tc = base_turnover * tc_bps / 10000.0
        tc = leverage * flat_tc
        portfolio_returns.append(leverage * ls_ret - tc)
        unscaled_returns.append(ls_ret - flat_tc)
        raw_ls_returns.append(ls_ret)

        prev_long_ids_seq = long_ids_list
        prev_long_weights_seq = long_w_np
        prev_long_realised_returns_seq = long_realised
        prev_short_ids_seq = short_ids_list
        prev_short_weights_seq = short_w_np
        prev_short_realised_returns_seq = short_realised

        if record_holdings:
            rebal_idx = idx // rebalance_freq
            leverage_trace.append({"rebalance_index": rebal_idx, "eom": eom_ts, "portfolio": "long_short", "leverage": float(leverage)})
            for i, fi in enumerate(long_idx_np):
                holdings.append(
                    {
                        "rebalance_index": rebal_idx,
                        "eom": eom_ts,
                        "portfolio": "long_short",
                        "leg": "long",
                        "country_id": int(cids[fi].item()),
                        "id": int(firm_ids[fi].item()),
                        "weight": float(long_w[i].item()),
                        "realised_return": (float(raw_returns[fi].item()) if valid[fi] else float("nan")),
                    }
                )
            for i, fi in enumerate(short_idx_np):
                holdings.append(
                    {
                        "rebalance_index": rebal_idx,
                        "eom": eom_ts,
                        "portfolio": "long_short",
                        "leg": "short",
                        "country_id": int(cids[fi].item()),
                        "id": int(firm_ids[fi].item()),
                        "weight": float(-short_w[i].item()),
                        "realised_return": (float(raw_returns[fi].item()) if valid[fi] else float("nan")),
                    }
                )

    returns_arr = np.array(portfolio_returns)
    unscaled_arr = np.array(unscaled_returns)
    if record_holdings:
        return returns_arr, unscaled_arr, holdings, leverage_trace
    return returns_arr, unscaled_arr


@torch.no_grad()
def portfolio_simulation_country_composite(
    models, dataset, config, rebalance_freq=6, tc_bps=25, long_short=True, seed_returns=None, record_holdings=False, record_per_country=False, score_key="scores_6m", target_key="target_6m"
):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    periods_per_year = 12 / rebalance_freq
    portfolio_returns = []
    unscaled_returns = []
    raw_composite_returns = list(seed_returns) if seed_returns else []
    prev_long_ids = {}
    prev_long_w = {}
    prev_long_realised = {}
    prev_short_ids = {}
    prev_short_w = {}
    prev_short_realised = {}
    prev_top_ids = {}
    prev_top_w = {}
    prev_top_realised = {}
    holdings = []
    leverage_trace = []
    per_country = {}

    portfolio_label = "country_composite_long_short" if long_short else "country_composite_long_only"
    leverage_bound = config.max_leverage_long_short if long_short else config.max_leverage_long_only

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        firm_ids = batch["firm_ids"]
        eom_ts = batch["eom"]

        scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)
        raw_returns = batch["targets"][target_key]
        valid = batch["valid_masks"][target_key]
        market_cap = batch["market_cap"]
        country_ids_np = cids.cpu().numpy()
        rebal_idx = idx // rebalance_freq

        country_returns = {}
        country_costs = {}
        country_market_caps = {}

        for cid in np.unique(country_ids_np):
            if cid < 0:
                continue
            idxs = np.where(country_ids_np == cid)[0]
            n_firms_c = len(idxs)
            if n_firms_c < config.min_firms_country:
                continue
            cid_int = int(cid)

            idxs_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
            scores_c = scores[idxs_t]

            if long_short:
                mean_c = scores_c.mean()
                long_local = (scores_c > mean_c).nonzero(as_tuple=True)[0]
                short_local = (scores_c <= mean_c).nonzero(as_tuple=True)[0]
                long_pos = idxs[long_local.cpu().numpy()]
                short_pos = idxs[short_local.cpu().numpy()]
                long_firm_ids_c = firm_ids[long_pos]
                short_firm_ids_c = firm_ids[short_pos]

                long_local_scores = scores_c[long_local]
                short_local_scores = scores_c[short_local]
                n_long_c = len(long_pos)
                n_short_c = len(short_pos)
                long_w = _capped_softmax_weights(long_local_scores - mean_c, config.max_position_weight)
                short_w = _capped_softmax_weights(mean_c - short_local_scores, config.max_position_weight)
                long_valid_c = valid[long_pos]
                short_valid_c = valid[short_pos]
                long_w = _renorm_over_valid(long_w, long_valid_c)
                short_w = _renorm_over_valid(short_w, short_valid_c)
                long_w_np = long_w.detach().cpu().numpy().astype(np.float64)
                short_w_np = short_w.detach().cpu().numpy().astype(np.float64)
                long_ids_list = [int(long_firm_ids_c[i].item()) for i in range(n_long_c)]
                short_ids_list = [int(short_firm_ids_c[i].item()) for i in range(n_short_c)]
                long_realised = {}
                short_realised = {}
                long_ret = 0.0
                for i, fi in enumerate(long_pos):
                    r = float(raw_returns[fi].item()) if valid[fi] else 0.0
                    long_realised[long_ids_list[i]] = r
                    long_ret += long_w_np[i] * r
                short_ret = 0.0
                for i, fi in enumerate(short_pos):
                    r = float(raw_returns[fi].item()) if valid[fi] else 0.0
                    short_realised[short_ids_list[i]] = r
                    short_ret += short_w_np[i] * r

                if cid_int in prev_long_ids:
                    d_long_ids, d_long_w = _drift_weights(prev_long_ids[cid_int], prev_long_w[cid_int], prev_long_realised[cid_int])
                else:
                    d_long_ids, d_long_w = None, None
                if cid_int in prev_short_ids:
                    d_short_ids, d_short_w = _drift_weights(prev_short_ids[cid_int], prev_short_w[cid_int], prev_short_realised[cid_int])
                else:
                    d_short_ids, d_short_w = None, None
                lt = _weight_l1_turnover(d_long_ids, d_long_w, long_ids_list, long_w_np)
                st = _weight_l1_turnover(d_short_ids, d_short_w, short_ids_list, short_w_np)
                turnover_c = lt + st

                country_returns[cid_int] = long_ret - short_ret
                prev_long_ids[cid_int] = long_ids_list
                prev_long_w[cid_int] = long_w_np
                prev_long_realised[cid_int] = long_realised
                prev_short_ids[cid_int] = short_ids_list
                prev_short_w[cid_int] = short_w_np
                prev_short_realised[cid_int] = short_realised

                if record_holdings:
                    for i, fi in enumerate(long_pos):
                        holdings.append(
                            {
                                "rebalance_index": rebal_idx,
                                "eom": eom_ts,
                                "portfolio": portfolio_label,
                                "leg": "long",
                                "country_id": int(cid),
                                "id": int(firm_ids[fi].item()),
                                "weight": float(long_w[i].item()),
                                "realised_return": (float(raw_returns[fi].item()) if valid[fi] else float("nan")),
                            }
                        )
                    for i, fi in enumerate(short_pos):
                        holdings.append(
                            {
                                "rebalance_index": rebal_idx,
                                "eom": eom_ts,
                                "portfolio": portfolio_label,
                                "leg": "short",
                                "country_id": int(cid),
                                "id": int(firm_ids[fi].item()),
                                "weight": float(-short_w[i].item()),
                                "realised_return": (float(raw_returns[fi].item()) if valid[fi] else float("nan")),
                            }
                        )
            else:
                top_pos = idxs
                top_firm_ids_c = firm_ids[top_pos]
                n_top_c = len(top_pos)

                top_scores_c = scores_c
                top_w = _capped_softmax_weights(top_scores_c, config.max_position_weight)
                top_valid_c = valid[top_pos]
                top_w = _renorm_over_valid(top_w, top_valid_c)
                top_w_np = top_w.detach().cpu().numpy().astype(np.float64)
                top_ids_list = [int(top_firm_ids_c[i].item()) for i in range(n_top_c)]
                top_realised = {}
                top_ret = 0.0
                for i, fi in enumerate(top_pos):
                    r = float(raw_returns[fi].item()) if valid[fi] else 0.0
                    top_realised[top_ids_list[i]] = r
                    top_ret += top_w_np[i] * r

                if cid_int in prev_top_ids:
                    d_top_ids, d_top_w = _drift_weights(prev_top_ids[cid_int], prev_top_w[cid_int], prev_top_realised[cid_int])
                else:
                    d_top_ids, d_top_w = None, None
                turnover_c = _weight_l1_turnover(d_top_ids, d_top_w, top_ids_list, top_w_np)

                country_returns[cid_int] = top_ret
                prev_top_ids[cid_int] = top_ids_list
                prev_top_w[cid_int] = top_w_np
                prev_top_realised[cid_int] = top_realised

                if record_holdings:
                    for i, fi in enumerate(top_pos):
                        holdings.append(
                            {
                                "rebalance_index": rebal_idx,
                                "eom": eom_ts,
                                "portfolio": portfolio_label,
                                "leg": "long",
                                "country_id": int(cid),
                                "id": int(firm_ids[fi].item()),
                                "weight": float(top_w[i].item()),
                                "realised_return": (float(raw_returns[fi].item()) if valid[fi] else float("nan")),
                            }
                        )

            country_costs[cid_int] = turnover_c * tc_bps / 10000.0
            cap = market_cap[idxs].sum().item()
            country_market_caps[cid_int] = cap if cap > 0 else float(n_firms_c)

        if not country_returns:
            portfolio_returns.append(0.0)
            unscaled_returns.append(0.0)
            raw_composite_returns.append(0.0)
            continue

        total_weight = sum(country_market_caps.values())
        composite_ret = 0.0
        composite_cost = 0.0
        for cid_int, ret in country_returns.items():
            w = country_market_caps[cid_int] / total_weight
            composite_ret += w * ret
            composite_cost += w * country_costs[cid_int]

            if record_per_country:
                entry = per_country.setdefault(cid_int, {"rebalance_indices": [], "returns": [], "weights": []})
                entry["rebalance_indices"].append(rebal_idx)
                entry["returns"].append(float(ret))
                entry["weights"].append(float(w))

        n_vol_periods = max(1, config.vol_lookback_months // rebalance_freq)
        if len(raw_composite_returns) >= n_vol_periods:
            recent = np.array(raw_composite_returns[-n_vol_periods:])
            realised_vol = recent.std(ddof=1) * np.sqrt(periods_per_year)
            leverage = config.target_vol / max(realised_vol, 1e-6)
            leverage = float(np.clip(leverage, 1.0 / leverage_bound, leverage_bound))
        else:
            leverage = 1.0

        portfolio_returns.append(leverage * composite_ret - leverage * composite_cost)
        unscaled_returns.append(composite_ret - composite_cost)
        raw_composite_returns.append(composite_ret)

        if record_holdings:
            leverage_trace.append({"rebalance_index": rebal_idx, "eom": eom_ts, "portfolio": portfolio_label, "leverage": float(leverage)})

    returns_arr = np.array(portfolio_returns)
    unscaled_arr = np.array(unscaled_returns)
    out: list = [returns_arr, unscaled_arr]
    if record_holdings:
        out.append(holdings)
        out.append(leverage_trace)
    if record_per_country:
        out.append(per_country)
    return tuple(out)


@torch.no_grad()
def portfolio_simulation_monthly(models, dataset, config, rebalance_freq=6, tc_bps=25, seed_returns=None, record_holdings=False, score_key="scores_6m"):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    n = len(dataset)
    periods_per_year = 12.0
    monthly_scaled = []
    monthly_unscaled = []
    raw_monthly_hist = list(seed_returns) if seed_returns else []
    holdings = []
    leverage_trace = []
    n_vol_periods = max(1, config.vol_lookback_months)

    held_drifted_w = None
    held_firm_ids = None
    held_n = 0
    current_leverage = 1.0

    for idx in range(n):
        batch = dataset[idx]
        eom_ts = batch["eom"]
        firm_ids = batch["firm_ids"]
        n_firms = firm_ids.shape[0]

        is_rebal = idx % rebalance_freq == 0
        flat_tc = 0.0

        if is_rebal:
            k0 = batch["k0"].to(device)
            k1 = batch["k1"].to(device)
            k0_miss = batch["k0_miss"].to(device)
            k1_miss = batch["k1_miss"].to(device)
            cids = batch["country_ids"].to(device)
            scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)
            held_weights = _capped_softmax_weights(scores, config.max_position_weight)
            prev_drifted_ids = [int(held_firm_ids[i].item()) for i in range(held_n)] if held_drifted_w is not None and held_firm_ids is not None else None
            prev_drifted_w = held_drifted_w.copy() if held_drifted_w is not None else None

            held_firm_ids = firm_ids
            held_n = n_firms
            held_drifted_w = held_weights.detach().cpu().numpy().astype(np.float64).copy()
            curr_ids_list = [int(firm_ids[i].item()) for i in range(n_firms)]

            base_turnover = _weight_l1_turnover(prev_drifted_ids, prev_drifted_w, curr_ids_list, held_drifted_w)
            flat_tc = base_turnover * tc_bps / 10000.0

            if len(raw_monthly_hist) >= n_vol_periods:
                recent = np.array(raw_monthly_hist[-n_vol_periods:])
                realised_vol = recent.std(ddof=1) * np.sqrt(periods_per_year)
                lev = config.target_vol / max(realised_vol, 1e-6)
                current_leverage = float(np.clip(lev, 1.0 / config.max_leverage_long_only, config.max_leverage_long_only))
            else:
                current_leverage = 1.0

            if record_holdings:
                rebal_idx = idx // rebalance_freq
                leverage_trace.append({"rebalance_index": rebal_idx, "eom": eom_ts, "portfolio": "long_only_monthly", "leverage": float(current_leverage)})
                for fi in range(n_firms):
                    holdings.append(
                        {
                            "rebalance_index": rebal_idx,
                            "eom": eom_ts,
                            "portfolio": "long_only_monthly",
                            "leg": "long",
                            "country_id": int(cids[fi].item()),
                            "id": int(firm_ids[fi].item()),
                            "weight": float(held_weights[fi].item()),
                        }
                    )

        if held_drifted_w is None or held_n == 0:
            monthly_scaled.append(0.0)
            monthly_unscaled.append(0.0)
            raw_monthly_hist.append(0.0)
            continue

        raw_returns = batch["targets"]["target_1m"]
        valid = batch["valid_masks"]["target_1m"]

        current_id_to_idx = {int(firm_ids[i].item()): i for i in range(n_firms)}
        r_array = np.zeros(held_n, dtype=np.float64)
        for held_idx in range(held_n):
            assert held_firm_ids is not None
            held_id = int(held_firm_ids[held_idx].item())
            if held_id in current_id_to_idx:
                cur_pos = current_id_to_idx[held_id]
                if valid[cur_pos]:
                    r_array[held_idx] = float(raw_returns[cur_pos].item())

        leg_return = float((held_drifted_w * r_array).sum())

        growth = held_drifted_w * (1.0 + r_array)
        g_sum = float(growth.sum())
        if g_sum > 1e-12:
            held_drifted_w = growth / g_sum

        tc = current_leverage * flat_tc
        monthly_scaled.append(current_leverage * leg_return - tc)
        monthly_unscaled.append(leg_return - flat_tc)
        raw_monthly_hist.append(leg_return)

    scaled_arr = np.array(monthly_scaled)
    unscaled_arr = np.array(monthly_unscaled)
    if record_holdings:
        return scaled_arr, unscaled_arr, holdings, leverage_trace
    return scaled_arr, unscaled_arr


@torch.no_grad()
def portfolio_simulation_long_short_monthly(models, dataset, config, rebalance_freq=6, tc_bps=25, seed_returns=None, record_holdings=False, score_key="scores_6m"):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    n = len(dataset)
    periods_per_year = 12.0
    monthly_scaled = []
    monthly_unscaled = []
    raw_monthly_hist = list(seed_returns) if seed_returns else []
    holdings = []
    leverage_trace = []
    n_vol_periods = max(1, config.vol_lookback_months)

    held_long_drifted_w = None
    held_long_ids = None
    held_short_drifted_w = None
    held_short_ids = None
    current_leverage = 1.0

    for idx in range(n):
        batch = dataset[idx]
        eom_ts = batch["eom"]
        firm_ids = batch["firm_ids"]
        n_firms = firm_ids.shape[0]

        is_rebal = idx % rebalance_freq == 0
        flat_tc = 0.0

        if is_rebal:
            k0 = batch["k0"].to(device)
            k1 = batch["k1"].to(device)
            k0_miss = batch["k0_miss"].to(device)
            k1_miss = batch["k1_miss"].to(device)
            cids = batch["country_ids"].to(device)
            scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)

            mean_score = scores.mean()
            long_idx = (scores > mean_score).nonzero(as_tuple=True)[0]
            short_idx = (scores <= mean_score).nonzero(as_tuple=True)[0]
            long_idx_np = long_idx.cpu().numpy()
            short_idx_np = short_idx.cpu().numpy()
            long_firm_ids = firm_ids[long_idx_np]
            short_firm_ids = firm_ids[short_idx_np]

            long_w = _capped_softmax_weights(scores[long_idx] - mean_score, config.max_position_weight)
            short_w = _capped_softmax_weights(mean_score - scores[short_idx], config.max_position_weight)

            prev_drifted_long_ids = [int(held_long_ids[i].item()) for i in range(len(held_long_ids))] if held_long_drifted_w is not None and held_long_ids is not None else None
            prev_drifted_long_w = held_long_drifted_w.copy() if held_long_drifted_w is not None else None
            prev_drifted_short_ids = [int(held_short_ids[i].item()) for i in range(len(held_short_ids))] if held_short_drifted_w is not None and held_short_ids is not None else None
            prev_drifted_short_w = held_short_drifted_w.copy() if held_short_drifted_w is not None else None

            held_long_ids = long_firm_ids
            held_short_ids = short_firm_ids
            held_long_drifted_w = long_w.detach().cpu().numpy().astype(np.float64).copy()
            held_short_drifted_w = short_w.detach().cpu().numpy().astype(np.float64).copy()
            curr_long_ids_list = [int(long_firm_ids[i].item()) for i in range(len(long_firm_ids))]
            curr_short_ids_list = [int(short_firm_ids[i].item()) for i in range(len(short_firm_ids))]

            lt = _weight_l1_turnover(prev_drifted_long_ids, prev_drifted_long_w, curr_long_ids_list, held_long_drifted_w)
            st = _weight_l1_turnover(prev_drifted_short_ids, prev_drifted_short_w, curr_short_ids_list, held_short_drifted_w)
            base_turnover = lt + st
            flat_tc = base_turnover * tc_bps / 10000.0

            if len(raw_monthly_hist) >= n_vol_periods:
                recent = np.array(raw_monthly_hist[-n_vol_periods:])
                realised_vol = recent.std(ddof=1) * np.sqrt(periods_per_year)
                lev = config.target_vol / max(realised_vol, 1e-6)
                current_leverage = float(np.clip(lev, 1.0 / config.max_leverage_long_short, config.max_leverage_long_short))
            else:
                current_leverage = 1.0

            if record_holdings:
                rebal_idx = idx // rebalance_freq
                leverage_trace.append({"rebalance_index": rebal_idx, "eom": eom_ts, "portfolio": "long_short_monthly", "leverage": float(current_leverage)})

        if held_long_drifted_w is None:
            monthly_scaled.append(0.0)
            monthly_unscaled.append(0.0)
            raw_monthly_hist.append(0.0)
            continue

        raw_returns = batch["targets"]["target_1m"]
        valid = batch["valid_masks"]["target_1m"]
        current_id_to_idx = {int(firm_ids[i].item()): i for i in range(n_firms)}

        def _step_leg(held_ids, drifted_w):
            n_held = len(drifted_w)
            r_array = np.zeros(n_held, dtype=np.float64)
            for j in range(n_held):
                held_id = int(held_ids[j].item())
                if held_id in current_id_to_idx:
                    cur_pos = current_id_to_idx[held_id]
                    if valid[cur_pos]:
                        r_array[j] = float(raw_returns[cur_pos].item())
            leg_ret = float((drifted_w * r_array).sum())
            growth = drifted_w * (1.0 + r_array)
            g_sum = float(growth.sum())
            new_drifted_w = growth / g_sum if g_sum > 1e-12 else drifted_w
            return leg_ret, new_drifted_w

        long_ret, held_long_drifted_w = _step_leg(held_long_ids, held_long_drifted_w)
        short_ret, held_short_drifted_w = _step_leg(held_short_ids, held_short_drifted_w)
        ls_ret = long_ret - short_ret

        tc = current_leverage * flat_tc
        monthly_scaled.append(current_leverage * ls_ret - tc)
        monthly_unscaled.append(ls_ret - flat_tc)
        raw_monthly_hist.append(ls_ret)

    scaled_arr = np.array(monthly_scaled)
    unscaled_arr = np.array(monthly_unscaled)
    if record_holdings:
        return scaled_arr, unscaled_arr, holdings, leverage_trace
    return scaled_arr, unscaled_arr


@torch.no_grad()
def portfolio_simulation_country_composite_monthly(models, dataset, config, rebalance_freq=6, tc_bps=25, long_short=True, seed_returns=None, record_holdings=False, score_key="scores_6m"):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    n = len(dataset)
    periods_per_year = 12.0
    monthly_scaled = []
    monthly_unscaled = []
    raw_monthly_hist = list(seed_returns) if seed_returns else []
    n_vol_periods = max(1, config.vol_lookback_months)

    leverage_bound = config.max_leverage_long_short if long_short else config.max_leverage_long_only
    portfolio_label = "country_composite_long_short_monthly" if long_short else "country_composite_long_only_monthly"

    held_long_ids = {}
    held_short_ids = {}
    held_top_ids = {}
    held_long_drifted_w = {}
    held_short_drifted_w = {}
    held_top_drifted_w = {}
    current_leverage = 1.0

    holdings = []
    leverage_trace = []

    for idx in range(n):
        batch = dataset[idx]
        eom_ts = batch["eom"]
        firm_ids = batch["firm_ids"]
        cids = batch["country_ids"]
        country_ids_np = cids.cpu().numpy() if torch.is_tensor(cids) else cids.numpy()
        n_firms = firm_ids.shape[0]

        is_rebal = idx % rebalance_freq == 0
        flat_country_cost = {}

        if is_rebal:
            k0 = batch["k0"].to(device)
            k1 = batch["k1"].to(device)
            k0_miss = batch["k0_miss"].to(device)
            k1_miss = batch["k1_miss"].to(device)
            cids_dev = cids.to(device)
            scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids_dev, key=score_key)

            prev_drifted_long_ids = {c: [int(held_long_ids[c][i].item()) for i in range(len(held_long_ids[c]))] for c in held_long_ids}
            prev_drifted_long_w = {c: held_long_drifted_w[c].copy() for c in held_long_drifted_w}
            prev_drifted_short_ids = {c: [int(held_short_ids[c][i].item()) for i in range(len(held_short_ids[c]))] for c in held_short_ids}
            prev_drifted_short_w = {c: held_short_drifted_w[c].copy() for c in held_short_drifted_w}
            prev_drifted_top_ids = {c: [int(held_top_ids[c][i].item()) for i in range(len(held_top_ids[c]))] for c in held_top_ids}
            prev_drifted_top_w = {c: held_top_drifted_w[c].copy() for c in held_top_drifted_w}

            held_long_ids = {}
            held_short_ids = {}
            held_top_ids = {}
            held_long_drifted_w = {}
            held_short_drifted_w = {}
            held_top_drifted_w = {}

            for cid in np.unique(country_ids_np):
                if cid < 0:
                    continue
                idxs = np.where(country_ids_np == cid)[0]
                n_firms_c = len(idxs)
                if n_firms_c < config.min_firms_country:
                    continue
                scores_c = scores[torch.as_tensor(idxs, device=device, dtype=torch.long)]
                cid_int = int(cid)

                if long_short:
                    mean_c = scores_c.mean()
                    long_local = (scores_c > mean_c).nonzero(as_tuple=True)[0]
                    short_local = (scores_c <= mean_c).nonzero(as_tuple=True)[0]
                    long_pos = idxs[long_local.cpu().numpy()]
                    short_pos = idxs[short_local.cpu().numpy()]
                    held_long_ids[cid_int] = firm_ids[long_pos]
                    held_short_ids[cid_int] = firm_ids[short_pos]
                    long_w_c = _capped_softmax_weights(scores_c[long_local] - mean_c, config.max_position_weight)
                    short_w_c = _capped_softmax_weights(mean_c - scores_c[short_local], config.max_position_weight)
                    held_long_drifted_w[cid_int] = long_w_c.detach().cpu().numpy().astype(np.float64)
                    held_short_drifted_w[cid_int] = short_w_c.detach().cpu().numpy().astype(np.float64)

                    curr_long_ids = [int(firm_ids[p].item()) for p in long_pos]
                    curr_short_ids = [int(firm_ids[p].item()) for p in short_pos]
                    lt = _weight_l1_turnover(prev_drifted_long_ids.get(cid_int), prev_drifted_long_w.get(cid_int), curr_long_ids, held_long_drifted_w[cid_int])
                    st = _weight_l1_turnover(prev_drifted_short_ids.get(cid_int), prev_drifted_short_w.get(cid_int), curr_short_ids, held_short_drifted_w[cid_int])
                    flat_country_cost[cid_int] = (lt + st) * tc_bps / 10000.0
                else:
                    held_top_ids[cid_int] = firm_ids[idxs]
                    top_w_c = _capped_softmax_weights(scores_c, config.max_position_weight)
                    held_top_drifted_w[cid_int] = top_w_c.detach().cpu().numpy().astype(np.float64)
                    curr_top_ids = [int(firm_ids[p].item()) for p in idxs]
                    t_ = _weight_l1_turnover(prev_drifted_top_ids.get(cid_int), prev_drifted_top_w.get(cid_int), curr_top_ids, held_top_drifted_w[cid_int])
                    flat_country_cost[cid_int] = t_ * tc_bps / 10000.0

            if len(raw_monthly_hist) >= n_vol_periods:
                recent = np.array(raw_monthly_hist[-n_vol_periods:])
                realised_vol = recent.std(ddof=1) * np.sqrt(periods_per_year)
                lev = config.target_vol / max(realised_vol, 1e-6)
                current_leverage = float(np.clip(lev, 1.0 / leverage_bound, leverage_bound))
            else:
                current_leverage = 1.0

            if record_holdings:
                rebal_idx = idx // rebalance_freq
                leverage_trace.append({"rebalance_index": rebal_idx, "eom": eom_ts, "portfolio": portfolio_label, "leverage": float(current_leverage)})

        if (long_short and not held_long_ids) or (not long_short and not held_top_ids):
            monthly_scaled.append(0.0)
            monthly_unscaled.append(0.0)
            raw_monthly_hist.append(0.0)
            continue

        raw_returns = batch["targets"]["target_1m"]
        valid = batch["valid_masks"]["target_1m"]
        market_cap = batch["market_cap"]
        current_id_to_idx = {int(firm_ids[i].item()): i for i in range(n_firms)}

        country_returns = {}
        country_market_caps = {}

        active_country_ids = list(held_long_ids.keys()) if long_short else list(held_top_ids.keys())

        def _step_country_leg(held_ids_tensor, drifted_w):
            n_held = len(drifted_w)
            r_array = np.zeros(n_held, dtype=np.float64)
            for j in range(n_held):
                held_id = int(held_ids_tensor[j].item())
                if held_id in current_id_to_idx:
                    cp = current_id_to_idx[held_id]
                    if valid[cp]:
                        r_array[j] = float(raw_returns[cp].item())
            leg_ret = float((drifted_w * r_array).sum())
            growth = drifted_w * (1.0 + r_array)
            g_sum = float(growth.sum())
            new_drifted_w = growth / g_sum if g_sum > 1e-12 else drifted_w
            return leg_ret, new_drifted_w

        for cid_int in active_country_ids:
            country_mask = country_ids_np == cid_int
            country_positions = np.where(country_mask)[0]
            if len(country_positions) == 0:
                continue
            cap = float(market_cap[country_positions].sum().item())
            if cap <= 0:
                cap = float(len(country_positions))
            country_market_caps[cid_int] = cap

            if long_short:
                long_ret_c, held_long_drifted_w[cid_int] = _step_country_leg(held_long_ids[cid_int], held_long_drifted_w[cid_int])
                short_ret_c, held_short_drifted_w[cid_int] = _step_country_leg(held_short_ids[cid_int], held_short_drifted_w[cid_int])
                country_returns[cid_int] = long_ret_c - short_ret_c
            else:
                top_ret_c, held_top_drifted_w[cid_int] = _step_country_leg(held_top_ids[cid_int], held_top_drifted_w[cid_int])
                country_returns[cid_int] = top_ret_c

        if not country_returns:
            monthly_scaled.append(0.0)
            monthly_unscaled.append(0.0)
            raw_monthly_hist.append(0.0)
            continue

        total_weight = sum(country_market_caps.values())
        composite_ret = 0.0
        composite_cost = 0.0
        for cid_int, ret in country_returns.items():
            w = country_market_caps[cid_int] / total_weight
            composite_ret += w * ret
            if is_rebal:
                composite_cost += w * flat_country_cost.get(cid_int, 0.0)

        tc = current_leverage * composite_cost
        monthly_scaled.append(current_leverage * composite_ret - tc)
        monthly_unscaled.append(composite_ret - composite_cost)
        raw_monthly_hist.append(composite_ret)

    scaled_arr = np.array(monthly_scaled)
    unscaled_arr = np.array(monthly_unscaled)
    if record_holdings:
        return scaled_arr, unscaled_arr, holdings, leverage_trace
    return scaled_arr, unscaled_arr


def compute_portfolio_metrics(returns, periods_per_year: float = 2):
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return {"cumulative_return": 0.0, "annualised_return": 0.0, "annualised_vol": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "n_rebalances": 0}
    cum_return = (1 + returns).prod() - 1
    annualised_return = (1 + cum_return) ** (periods_per_year / max(len(returns), 1)) - 1
    annualised_vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / max(annualised_vol, 1e-8)
    cum_wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_wealth)
    drawdown = (peak - cum_wealth) / peak
    return {
        "cumulative_return": cum_return,
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.max(),
        "n_rebalances": len(returns),
    }


def compute_portfolio_metrics_extended(returns, periods_per_year: float = 2, dates=None):
    base = compute_portfolio_metrics(returns, periods_per_year)
    returns_arr = np.asarray(returns, dtype=float)
    if dates is not None and len(returns_arr) > 0:
        years = pd.DatetimeIndex(dates).year.to_numpy()
        per_year = {}
        for y in sorted(set(years.tolist())):
            mask = years == y
            sub = returns_arr[mask]
            if len(sub) < 1:
                continue
            y_ret = float(sub.mean() * periods_per_year)
            y_vol = float(sub.std(ddof=1) * np.sqrt(periods_per_year)) if len(sub) > 1 else 0.0
            y_sharpe = y_ret / max(y_vol, 1e-8) if y_vol > 1e-12 else float('nan')
            ycw = np.cumprod(1.0 + sub)
            ypk = np.maximum.accumulate(ycw)
            y_dd = float(((ypk - ycw) / ypk).max())
            per_year[int(y)] = {"ann_ret": y_ret, "ann_vol": y_vol, "sharpe": y_sharpe, "max_dd": y_dd, "cum_return": float(ycw[-1] - 1.0), "n_obs": int(len(sub))}
        base["per_year"] = per_year
    return base


def _to_native(v):
    if isinstance(v, dict):
        return {kk: _to_native(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_native(vv) for vv in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return [_to_native(x) for x in v.tolist()]
    if isinstance(v, torch.Tensor):
        if v.dim() == 0:
            return float(v.item())
        return _to_native(v.detach().cpu().numpy())
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _jsonable_metrics(metrics_dict):
    return _to_native(metrics_dict)


def _build_per_country_block(per_country_dict, country_codes_list, country_to_id_map, periods_per_year: float = 2, dates=None):
    id_to_code = {v: k for k, v in country_to_id_map.items()}
    out = {}
    for cid_int, entry in per_country_dict.items():
        country_code = id_to_code.get(cid_int, f"id_{cid_int}")
        returns_arr = np.array(entry["returns"], dtype=float)
        weights_arr = np.array(entry["weights"], dtype=float)
        base = compute_portfolio_metrics_extended(returns_arr, periods_per_year)
        contributions = weights_arr * returns_arr
        time_avg_contrib = float(contributions.mean()) if len(contributions) > 0 else 0.0
        cum_contrib = float(np.prod(1 + contributions) - 1) if len(contributions) > 0 else 0.0
        out[country_code] = {
            "country_code": country_code,
            "country_id": int(cid_int),
            "rebalance_indices": [int(r) for r in entry["rebalance_indices"]],
            "returns_6m": [float(r) for r in entry["returns"]],
            "weights": [float(w) for w in entry["weights"]],
            "annualised_return": base["annualised_return"],
            "annualised_vol": base["annualised_vol"],
            "sharpe_ratio": base["sharpe_ratio"],
            "max_drawdown": base["max_drawdown"],
            "cumulative_return": base["cumulative_return"],
            "time_average_contribution": time_avg_contrib,
            "cumulative_contribution": cum_contrib,
        }
    return out


import re

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
        all_results[variant_name]["per_seed_metrics"] = []
    all_results[variant_name]["per_seed_metrics"].append({"seed_idx": seed_idx, "metrics_path": str(metrics_path), "metrics": metrics})
    variant_seed_paths.setdefault(variant_name, []).append((seed_idx, cfg.results_dir / f"weights_{variant_name}_seed{seed_idx}.safetensors"))

if not all_results:
    raise RuntimeError(f"No seeded metrics files found in {cfg.results_dir}. " f"Expected files of the form metrics_{{variant}}_seed{{i}}.json.")

for variant_name in variant_seed_paths:
    variant_seed_paths[variant_name].sort(key=lambda x: x[0])

test_ds = load_dataset(cfg.test_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df)
val_ds = load_dataset(cfg.val_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df)

horizons = [("scores_6m", "target_6m", "6m", 6)]

variant_test_summary = {}
variant_val_summary = {}
variant_models = {}

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
    variant_models[variant_name] = variant_model_list
    variant_model = variant_model_list

    val_rebal_freq = 6
    val_periods_per_year = 12 / val_rebal_freq
    val_ls_scaled, val_ls_unscaled = portfolio_simulation_long_short(variant_model, val_ds, cfg, rebalance_freq=val_rebal_freq, tc_bps=cfg.tc_bps)
    val_ls_metrics = compute_portfolio_metrics_extended(val_ls_scaled, val_periods_per_year)
    per_seed_corrs = []
    for seed_record in all_results[variant_name].get("per_seed_metrics", []):
        c = seed_record["metrics"].get("val_metrics", {}).get("rank_corr_6m")
        if isinstance(c, (int, float)):
            per_seed_corrs.append(c)
    val_rank_corr = float(np.mean(per_seed_corrs)) if per_seed_corrs else float("nan")
    variant_val_summary[variant_name] = {"rank_corr_6m": val_rank_corr, "sharpe_ls": val_ls_metrics["sharpe_ratio"]}

    portfolio_block = {}
    per_country_blocks = {}
    holdings_records = []
    leverage_records = []

    mo_lo_returns = np.array([])
    mo_lo_unscaled = np.array([])
    mo_ls_returns = np.array([])
    mo_ls_unscaled = np.array([])
    mo_cc_lo_returns = np.array([])
    mo_cc_lo_unscaled = np.array([])
    mo_cc_ls_returns = np.array([])
    mo_cc_ls_unscaled = np.array([])
    monthly_dates = []

    for score_key, target_key, horizon_label, rebal_freq in horizons:
        periods_per_year = 12 / rebal_freq

        lo_seed = _seed_vol_history(variant_model, val_ds, cfg, rebal_freq, "long_only", score_key=score_key, target_key=target_key)
        ls_seed = _seed_vol_history(variant_model, val_ds, cfg, rebal_freq, "long_short", score_key=score_key, target_key=target_key)

        lo_result = portfolio_simulation(
            variant_model, test_ds, cfg, rebalance_freq=rebal_freq, tc_bps=cfg.tc_bps, seed_returns=lo_seed, record_holdings=(horizon_label == "6m"), score_key=score_key, target_key=target_key
        )
        lo_returns, lo_unscaled = lo_result[0], lo_result[1]
        if len(lo_result) == 4:
            lo_holdings, lo_leverage = lo_result[2], lo_result[3]
        else:
            lo_holdings, lo_leverage = [], []

        ls_result = portfolio_simulation_long_short(
            variant_model, test_ds, cfg, rebalance_freq=rebal_freq, tc_bps=cfg.tc_bps, seed_returns=ls_seed, record_holdings=(horizon_label == "6m"), score_key=score_key, target_key=target_key
        )
        ls_returns, ls_unscaled = ls_result[0], ls_result[1]
        if len(ls_result) == 4:
            ls_holdings, ls_leverage = ls_result[2], ls_result[3]
        else:
            ls_holdings, ls_leverage = [], []

        cc_lo_out = portfolio_simulation_country_composite(
            variant_model,
            test_ds,
            cfg,
            rebalance_freq=rebal_freq,
            tc_bps=cfg.tc_bps,
            long_short=False,
            seed_returns=lo_seed,
            record_holdings=(horizon_label == "6m"),
            record_per_country=(horizon_label == "6m"),
            score_key=score_key,
            target_key=target_key,
        )
        cc_lo_returns, cc_lo_unscaled = cc_lo_out[0], cc_lo_out[1]
        if horizon_label == "6m":
            cc_lo_holdings = cc_lo_out[2]
            cc_lo_leverage = cc_lo_out[3]
            cc_lo_per_country = cc_lo_out[4]
        else:
            cc_lo_holdings, cc_lo_leverage, cc_lo_per_country = [], [], {}

        cc_ls_out = portfolio_simulation_country_composite(
            variant_model,
            test_ds,
            cfg,
            rebalance_freq=rebal_freq,
            tc_bps=cfg.tc_bps,
            long_short=True,
            seed_returns=ls_seed,
            record_holdings=(horizon_label == "6m"),
            record_per_country=(horizon_label == "6m"),
            score_key=score_key,
            target_key=target_key,
        )
        cc_ls_returns, cc_ls_unscaled = cc_ls_out[0], cc_ls_out[1]
        if horizon_label == "6m":
            cc_ls_holdings = cc_ls_out[2]
            cc_ls_leverage = cc_ls_out[3]
            cc_ls_per_country = cc_ls_out[4]
        else:
            cc_ls_holdings, cc_ls_leverage, cc_ls_per_country = [], [], {}

        mo_lo_returns = np.array([])
        mo_lo_unscaled = np.array([])
        mo_ls_returns = np.array([])
        mo_ls_unscaled = np.array([])
        mo_cc_lo_returns = np.array([])
        mo_cc_lo_unscaled = np.array([])
        mo_cc_ls_returns = np.array([])
        mo_cc_ls_unscaled = np.array([])
        if horizon_label == "6m":
            mo_lo_seed = _seed_vol_history_monthly(variant_model, val_ds, cfg, rebal_freq, "long_only", score_key=score_key)
            mo_ls_seed = _seed_vol_history_monthly(variant_model, val_ds, cfg, rebal_freq, "long_short", score_key=score_key)
            mo_cc_lo_seed = _seed_vol_history_monthly(variant_model, val_ds, cfg, rebal_freq, "country_long_only", score_key=score_key)
            mo_cc_ls_seed = _seed_vol_history_monthly(variant_model, val_ds, cfg, rebal_freq, "country_long_short", score_key=score_key)

            mo_lo_result = portfolio_simulation_monthly(variant_model, test_ds, cfg, rebalance_freq=rebal_freq, tc_bps=cfg.tc_bps, seed_returns=mo_lo_seed, record_holdings=False, score_key=score_key)
            mo_lo_returns, mo_lo_unscaled = mo_lo_result[0], mo_lo_result[1]

            mo_ls_result = portfolio_simulation_long_short_monthly(
                variant_model, test_ds, cfg, rebalance_freq=rebal_freq, tc_bps=cfg.tc_bps, seed_returns=mo_ls_seed, record_holdings=False, score_key=score_key
            )
            mo_ls_returns, mo_ls_unscaled = mo_ls_result[0], mo_ls_result[1]

            mo_cc_lo_result = portfolio_simulation_country_composite_monthly(
                variant_model, test_ds, cfg, rebalance_freq=rebal_freq, tc_bps=cfg.tc_bps, long_short=False, seed_returns=mo_cc_lo_seed, record_holdings=False, score_key=score_key
            )
            mo_cc_lo_returns, mo_cc_lo_unscaled = (mo_cc_lo_result[0], mo_cc_lo_result[1])

            mo_cc_ls_result = portfolio_simulation_country_composite_monthly(
                variant_model, test_ds, cfg, rebalance_freq=rebal_freq, tc_bps=cfg.tc_bps, long_short=True, seed_returns=mo_cc_ls_seed, record_holdings=False, score_key=score_key
            )
            mo_cc_ls_returns, mo_cc_ls_unscaled = (mo_cc_ls_result[0], mo_cc_ls_result[1])

        all_test_dates = [test_ds.monthly_data[i]["eom"] for i in range(len(test_ds))]
        rebal_dates = all_test_dates[::rebal_freq]
        # target_1m is aliased from ret_exc_lead1m in load_dataset, which on a row
        # dated eom holds the excess return earned in the following month. the
        # monthly simulations therefore return, at position i, the return earned in
        # the month after all_test_dates[i]. we advance each label by one month so
        # the recorded date is the month the return was actually earned, which is
        # what any join against an externally dated series requires
        monthly_dates = [pd.Timestamp(d) + pd.offsets.MonthEnd(1) for d in all_test_dates]

        block = {
            "long_only": compute_portfolio_metrics_extended(lo_returns, periods_per_year, dates=rebal_dates),
            "long_only_unscaled": compute_portfolio_metrics_extended(lo_unscaled, periods_per_year, dates=rebal_dates),
            "long_short": compute_portfolio_metrics_extended(ls_returns, periods_per_year, dates=rebal_dates),
            "long_short_unscaled": compute_portfolio_metrics_extended(ls_unscaled, periods_per_year, dates=rebal_dates),
            "country_composite_long_only": compute_portfolio_metrics_extended(cc_lo_returns, periods_per_year, dates=rebal_dates),
            "country_composite_long_only_unscaled": compute_portfolio_metrics_extended(cc_lo_unscaled, periods_per_year, dates=rebal_dates),
            "country_composite_long_short": compute_portfolio_metrics_extended(cc_ls_returns, periods_per_year, dates=rebal_dates),
            "country_composite_long_short_unscaled": compute_portfolio_metrics_extended(cc_ls_unscaled, periods_per_year, dates=rebal_dates),
        }
        if horizon_label == "6m":
            block["long_only_monthly"] = compute_portfolio_metrics_extended(mo_lo_returns, 12.0, dates=monthly_dates)
            block["long_only_monthly_unscaled"] = compute_portfolio_metrics_extended(mo_lo_unscaled, 12.0, dates=monthly_dates)
            block["long_short_monthly"] = compute_portfolio_metrics_extended(mo_ls_returns, 12.0, dates=monthly_dates)
            block["long_short_monthly_unscaled"] = compute_portfolio_metrics_extended(mo_ls_unscaled, 12.0, dates=monthly_dates)
            block["country_composite_long_only_monthly"] = compute_portfolio_metrics_extended(mo_cc_lo_returns, 12.0, dates=monthly_dates)
            block["country_composite_long_only_monthly_unscaled"] = compute_portfolio_metrics_extended(mo_cc_lo_unscaled, 12.0, dates=monthly_dates)
            block["country_composite_long_short_monthly"] = compute_portfolio_metrics_extended(mo_cc_ls_returns, 12.0, dates=monthly_dates)
            block["country_composite_long_short_monthly_unscaled"] = compute_portfolio_metrics_extended(mo_cc_ls_unscaled, 12.0, dates=monthly_dates)
        portfolio_block[horizon_label] = block

        if horizon_label == "6m":
            per_country_blocks["country_composite_long_only"] = _build_per_country_block(cc_lo_per_country, country_codes, col_meta["country_to_id"], periods_per_year)
            per_country_blocks["country_composite_long_short"] = _build_per_country_block(cc_ls_per_country, country_codes, col_meta["country_to_id"], periods_per_year)
            holdings_records.extend(lo_holdings)
            holdings_records.extend(ls_holdings)
            holdings_records.extend(cc_lo_holdings)
            holdings_records.extend(cc_ls_holdings)
            leverage_records.extend(lo_leverage)
            leverage_records.extend(ls_leverage)
            leverage_records.extend(cc_lo_leverage)
            leverage_records.extend(cc_ls_leverage)

    per_seed_test_corrs = []
    for seed_record in all_results[variant_name].get("per_seed_metrics", []):
        c = seed_record["metrics"].get("test_metrics", {}).get("rank_corr_6m")
        if isinstance(c, (int, float)):
            per_seed_test_corrs.append(c)
    test_rank_corr = float(np.mean(per_seed_test_corrs)) if per_seed_test_corrs else float("nan")

    variant_test_summary[variant_name] = {
        "corr_6m": test_rank_corr,
        "sharpe_lo": portfolio_block["6m"]["long_only"]["sharpe_ratio"],
        "sharpe_lo_unscaled": portfolio_block["6m"]["long_only_unscaled"]["sharpe_ratio"],
        "sharpe_ls": portfolio_block["6m"]["long_short"]["sharpe_ratio"],
        "sharpe_ls_unscaled": portfolio_block["6m"]["long_short_unscaled"]["sharpe_ratio"],
        "vol_ls": portfolio_block["6m"]["long_short"]["annualised_vol"],
        "vol_ls_unscaled": portfolio_block["6m"]["long_short_unscaled"]["annualised_vol"],
        "sharpe_lo_monthly": portfolio_block["6m"]["long_only_monthly"]["sharpe_ratio"],
        "sharpe_lo_monthly_u": portfolio_block["6m"]["long_only_monthly_unscaled"]["sharpe_ratio"],
        "sharpe_ls_monthly": portfolio_block["6m"]["long_short_monthly"]["sharpe_ratio"],
        "sharpe_ls_monthly_u": portfolio_block["6m"]["long_short_monthly_unscaled"]["sharpe_ratio"],
    }

    variant_metrics_path = cfg.results_dir / f"metrics_{variant_name}.json"
    saved_metrics = dict(all_results[variant_name])
    saved_metrics.pop("per_seed_metrics", None)
    saved_metrics["portfolio_metrics"] = _jsonable_metrics(portfolio_block["6m"])
    saved_metrics["per_horizon_portfolio_metrics"] = _jsonable_metrics(portfolio_block)
    saved_metrics["per_country"] = _jsonable_metrics(per_country_blocks)
    saved_metrics["validation_portfolio_metrics"] = _jsonable_metrics({"long_short": val_ls_metrics})
    saved_metrics["ensemble_seed_count"] = len(variant_model_list)
    saved_metrics["ensemble_test_rank_corr_6m"] = test_rank_corr
    saved_metrics["ensemble_val_rank_corr_6m"] = val_rank_corr
    with open(variant_metrics_path, "w") as f:
        json.dump(saved_metrics, f, indent=2)

    per_year_rows = []
    block_6m = portfolio_block["6m"]
    for strategy_key, metrics_dict in block_6m.items():
        if not isinstance(metrics_dict, dict):
            continue
        py = metrics_dict.get("per_year", {})
        scaling = "unscaled" if strategy_key.endswith("_unscaled") else "scaled"
        strategy = strategy_key[: -len("_unscaled")] if strategy_key.endswith("_unscaled") else strategy_key
        for year, ym in sorted(py.items()):
            per_year_rows.append(
                {
                    "variant": variant_name,
                    "strategy": strategy,
                    "scaling": scaling,
                    "year": int(year),
                    "ann_ret": round(float(ym["ann_ret"]) * 100, 4),
                    "ann_vol": round(float(ym["ann_vol"]) * 100, 4),
                    "sharpe": (None if np.isnan(ym["sharpe"]) else round(float(ym["sharpe"]), 4)),
                    "max_dd": round(float(ym["max_dd"]) * 100, 4),
                    "cum_return": round(float(ym["cum_return"]) * 100, 4),
                    "n_obs": int(ym["n_obs"]),
                }
            )
    per_year_df = pd.DataFrame(per_year_rows)
    per_year_path = cfg.results_dir / f"per_year_{variant_name}.csv"
    per_year_df.to_csv(per_year_path, index=False)
    print(f"Per year metrics  {per_year_path}  ({len(per_year_df):,} rows)")

    def _build_monthly_rows(strategy, scaling, rets, dates):
        rets = np.asarray(rets, dtype=np.float64)
        if len(rets) == 0:
            return []
        cum_wealth = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(cum_wealth)
        drawdown = (cum_wealth - peak) / peak
        rolling_sharpe = np.full(len(rets), np.nan)
        rolling_ret = np.full(len(rets), np.nan)
        for i in range(11, len(rets)):
            w = rets[i - 11 : i + 1]
            mu = float(w.mean() * 12.0)
            sigma = float(w.std(ddof=1) * np.sqrt(12.0))
            rolling_ret[i] = mu
            if sigma > 1e-12:
                rolling_sharpe[i] = mu / sigma
        rows = []
        for i, eom in enumerate(dates):
            rows.append(
                {
                    "variant": variant_name,
                    "strategy": strategy,
                    "scaling": scaling,
                    "eom": pd.Timestamp(eom).strftime("%Y-%m-%d"),
                    "return": round(float(rets[i]), 6),
                    "cumulative_wealth": round(float(cum_wealth[i]), 6),
                    "drawdown": round(float(drawdown[i]), 6),
                    "rolling_sharpe_12m": (None if np.isnan(rolling_sharpe[i]) else round(float(rolling_sharpe[i]), 4)),
                    "rolling_return_12m": (None if np.isnan(rolling_ret[i]) else round(float(rolling_ret[i]) * 100, 4)),
                }
            )
        return rows

    monthly_rows = []
    monthly_rows.extend(_build_monthly_rows("long_only", "unscaled", mo_lo_unscaled, monthly_dates))
    monthly_rows.extend(_build_monthly_rows("long_only", "scaled", mo_lo_returns, monthly_dates))
    monthly_rows.extend(_build_monthly_rows("long_short", "unscaled", mo_ls_unscaled, monthly_dates))
    monthly_rows.extend(_build_monthly_rows("long_short", "scaled", mo_ls_returns, monthly_dates))
    monthly_rows.extend(_build_monthly_rows("country_composite_long_only", "unscaled", mo_cc_lo_unscaled, monthly_dates))
    monthly_rows.extend(_build_monthly_rows("country_composite_long_only", "scaled", mo_cc_lo_returns, monthly_dates))
    monthly_rows.extend(_build_monthly_rows("country_composite_long_short", "unscaled", mo_cc_ls_unscaled, monthly_dates))
    monthly_rows.extend(_build_monthly_rows("country_composite_long_short", "scaled", mo_cc_ls_returns, monthly_dates))
    per_month_df = pd.DataFrame(monthly_rows)
    per_month_path = cfg.results_dir / f"per_month_{variant_name}.csv"
    per_month_df.to_csv(per_month_path, index=False)
    print(f"Per month metrics {per_month_path}  ({len(per_month_df):,} rows)")

    if holdings_records:
        holdings_df = pd.DataFrame(holdings_records)
        holdings_path = cfg.results_dir / f"holdings_{variant_name}.parquet"
        holdings_df.to_parquet(holdings_path, index=False)
        print(f"Holdings  {holdings_path}  ({len(holdings_df):,} rows)")

    if leverage_records:
        leverage_df = pd.DataFrame(leverage_records)
        leverage_path = cfg.results_dir / f"leverage_{variant_name}.parquet"
        leverage_df.to_parquet(leverage_path, index=False)
        print(f"Leverage  {leverage_path}  ({len(leverage_df):,} rows)")

    print(f"Metrics updated  {variant_metrics_path}")


best_variant = max(variant_val_summary, key=lambda v: variant_val_summary[v]["rank_corr_6m"])
best_sharpe_variant = max(variant_val_summary, key=lambda v: variant_val_summary[v]["sharpe_ls"])

print()
print("Variant comparison (test set columns are reported; validation set")
print("columns are the selection criteria):")
print("Scaled = volatility-targeted with leverage overlay.")
print("Unscaled = flat position sizing, no leverage.")
print()
hdr = f"{'variant':<14} {'val_corr':>8}  {'val_sr_ls':>9}  " f"{'corr_6m':>8}  " f"{'sr_lo':>7}  {'sr_lo_u':>7}  " f"{'sr_ls':>7}  {'sr_ls_u':>7}  " f"{'vol_ls':>7}  {'vol_ls_u':>8}"
print(hdr)
for variant_name in all_results:
    vs = variant_val_summary[variant_name]
    ts = variant_test_summary[variant_name]
    marker = ""
    if variant_name == best_variant:
        marker += "  *corr"
    if variant_name == best_sharpe_variant:
        marker += "  *sharpe"
    print(
        f"{variant_name:<14} "
        f"{vs['rank_corr_6m']:>8.4f}  {vs['sharpe_ls']:>9.4f}  "
        f"{ts['corr_6m']:>8.4f}  "
        f"{ts['sharpe_lo']:>7.4f}  {ts['sharpe_lo_unscaled']:>7.4f}  "
        f"{ts['sharpe_ls']:>7.4f}  {ts['sharpe_ls_unscaled']:>7.4f}  "
        f"{ts['vol_ls']:>7.4f}  {ts['vol_ls_unscaled']:>8.4f}{marker}"
    )
print(f"Best by validation rank correlation, {best_variant}")
print(f"Best by validation long short Sharpe ratio, {best_sharpe_variant}")

print()
print("Monthly-frequency Sharpe ratios (6m rebalance, target_1m monthly returns):")
hdr_mo = f"{'variant':<14} " f"{'sr_lo_mo':>9}  {'sr_lo_mo_u':>11}  " f"{'sr_ls_mo':>9}  {'sr_ls_mo_u':>11}"
print(hdr_mo)
for variant_name in all_results:
    ts = variant_test_summary[variant_name]
    print(f"{variant_name:<14} " f"{ts['sharpe_lo_monthly']:>9.4f}  {ts['sharpe_lo_monthly_u']:>11.4f}  " f"{ts['sharpe_ls_monthly']:>9.4f}  {ts['sharpe_ls_monthly_u']:>11.4f}")


_report_labels = [
    ("long_only", "long_only_unscaled"),
    ("long_short", "long_short_unscaled"),
    ("country_composite_long_only", "country_composite_long_only_unscaled"),
    ("country_composite_long_short", "country_composite_long_short_unscaled"),
    ("long_only_monthly", "long_only_monthly_unscaled"),
    ("long_short_monthly", "long_short_monthly_unscaled"),
    ("country_composite_long_only_monthly", "country_composite_long_only_monthly_unscaled"),
    ("country_composite_long_short_monthly", "country_composite_long_short_monthly_unscaled"),
]
_metric_keys = ("cumulative_return", "annualised_return", "annualised_vol", "sharpe_ratio", "max_drawdown", "n_rebalances")

all_variant_metrics = {}
combined_rows = []

for variant_name in all_results:
    marker = ""
    if variant_name == best_variant:
        marker += " (best by validation rank correlation)"
    if variant_name == best_sharpe_variant:
        marker += " (best by validation long short sharpe)"

    variant_metrics_path = cfg.results_dir / f"metrics_{variant_name}.json"
    with open(variant_metrics_path, "r") as f:
        saved_metrics = json.load(f)
    six_month_metrics = saved_metrics["portfolio_metrics"]
    all_variant_metrics[variant_name] = six_month_metrics

    print(f"Detailed test set portfolio metrics for {variant_name}{marker}:")
    print("scaled means with volatility overlay leverage, unscaled means no leverage, flat tc.")
    print()
    for label, label_u in _report_labels:
        print(f"{label}:")
        m = six_month_metrics[label]
        mu = six_month_metrics[label_u]
        for k in _metric_keys:
            v = m.get(k)
            vu = mu.get(k)
            if v is None and vu is None:
                print(f"  {k:<30} n/a")
            elif isinstance(v, (int, float)) and isinstance(vu, (int, float)):
                print(f"  {k:<30} scaled {v:>9.4f}   unscaled {vu:>9.4f}")
            for scaling, value in (("scaled", v), ("unscaled", vu)):
                if isinstance(value, (int, float)):
                    combined_rows.append({"variant": variant_name, "construction": label, "scaling": scaling, "metric": k, "value": value})
        print()

combined_df = pd.DataFrame(combined_rows)
combined_path = cfg.results_dir / "all_variants_portfolio_metrics.csv"
combined_df.to_csv(combined_path, index=False)
print(f"Combined portfolio metrics for all variants written to {combined_path}")

wide_df = combined_df.pivot_table(index=["variant", "construction", "scaling"], columns="metric", values="value").reset_index()
wide_df = wide_df[["variant", "construction", "scaling"] + list(_metric_keys)]
wide_path = cfg.results_dir / "all_variants_portfolio_metrics_wide.csv"
wide_df.to_csv(wide_path, index=False)
print(f"Wide format combined portfolio metrics written to {wide_path}")


for variant_name, model_list in variant_models.items():
    for m in model_list:
        del m
del variant_models, test_ds, val_ds
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
