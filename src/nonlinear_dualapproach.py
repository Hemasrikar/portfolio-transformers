import gc
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
from data_processing import run_preprocessing, target_cols
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    data_path: Path = Path("data/Global Factor_EM.parquet")
    output_dir: Path = Path("data/processed")
    results_dir: Path = Path("results1/transformer")
    train_path: Path = Path("data/processed/train.parquet")
    val_path: Path = Path("data/processed/val.parquet")
    test_path: Path = Path("data/processed/test.parquet")
    col_metadata_path: Path = Path("data/processed/column_metadata.json")
    country_lookup_path: Path = Path("data/processed/country_lookup.parquet")

    train_end: str = "2015-12-31"
    val_end: str = "2020-12-31"
    missing_col_threshold: float = 0.30

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
    min_firms_attention: int = 30
    warmup_epochs: int = 5

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    grad_clip: float = 1.0

    n_seeds: int = 5

    target_vol: float = 0.10
    vol_lookback: int = 6
    max_leverage_long_only: float = 3.0
    max_leverage_long_short: float = 3.0
    min_firms_country: int = 20
    max_position_weight: float = 0.05

    encoding_variant: str = "linear"
    seed: int = 24


cfg = Config()
cfg.results_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)


required_outputs = [cfg.train_path, cfg.val_path, cfg.test_path, cfg.col_metadata_path, cfg.country_lookup_path]
if not all(p.exists() for p in required_outputs):
    print("Processed data not found. Running preprocessing pipeline.")
    run_preprocessing(cfg)
else:
    print("Processed data found. Skipping preprocessing.")


with open(cfg.col_metadata_path, "r") as f:
    col_meta = json.load(f)

k0_chars = col_meta['retained_k0']
k1_chars = col_meta['retained_k1']

parquet_schema_cols = set(pq.read_schema(cfg.train_path).names)

k0_feature_cols = [c for c in k0_chars if c in parquet_schema_cols]
k1_feature_cols = [c for c in k1_chars if c in parquet_schema_cols]

k0_miss_cols = [f'{c}_miss' for c in k0_chars if f'{c}_miss' in parquet_schema_cols]
k1_miss_cols = [f'{c}_miss' for c in k1_chars if f'{c}_miss' in parquet_schema_cols]

country_lookup_df = pd.read_parquet(cfg.country_lookup_path)
country_lookup_df['eom'] = pd.to_datetime(country_lookup_df['eom'])

country_to_id = col_meta['country_to_id']
country_codes = col_meta['country_codes']

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
    required = ["id", "eom"] + k0_cols + k1_cols + k0_miss + k1_miss + target_col_list
    has_market_cap = "me" in available
    if has_market_cap:
        required = required + ["me"]
    else:
        print(f"'me' column not found in {path}. Country composite simulation" f"will use firm count weighting rather than market capitalisation weighting.")
    load_cols = [c for c in required if c in available]
    df = pd.read_parquet(path, columns=load_cols)
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if col not in df.columns:
            df[col] = 0.0
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)
    if has_market_cap and df["me"].isna().any():
        df["me"] = df["me"].fillna(0.0)
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


def _differentiable_long_short_return(scores, returns, valid_mask):
    valid_scores = scores[valid_mask]
    valid_returns = returns[valid_mask]
    n_valid = valid_scores.shape[0]
    if n_valid < 10:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    mean_score = valid_scores.mean()
    long_mask = valid_scores > mean_score
    short_mask = ~long_mask
    if long_mask.sum() == 0 or short_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    long_logits = valid_scores[long_mask] - mean_score
    short_logits = mean_score - valid_scores[short_mask]
    long_w = F.softmax(long_logits, dim=0)
    short_w = F.softmax(short_logits, dim=0)
    long_ret = (long_w * valid_returns[long_mask]).sum()
    short_ret = (short_w * valid_returns[short_mask]).sum()
    return long_ret - short_ret


def compute_msrr_loss(output, targets, valid_masks, config):
    device_ = output["scores_6m"].device
    target = targets["target_6m"]
    valid = valid_masks["target_6m"]
    if valid.sum() < 10:
        zero = torch.tensor(0.0, device=device_, requires_grad=True)
        return zero, 0.0, 0.0
    r_pf = _differentiable_long_short_return(output["scores_6m"], target, valid)
    r_base = _differentiable_long_short_return(output["base_6m"], target, valid)
    r_path2 = _differentiable_long_short_return(output["path2_6m"], target, valid)
    main_loss = (1.0 - r_pf).pow(2)
    aux_loss = (1.0 - r_base).pow(2)
    aux2_loss = (1.0 - r_path2).pow(2)
    total = main_loss + config.lambda_aux * aux_loss + config.lambda_aux2 * aux2_loss
    return total, main_loss.item(), aux_loss.item()


def compute_rank_correlation(scores, targets, valid_mask):
    valid = valid_mask
    if valid.sum() < 10:
        return 0.0

    pred = scores[valid]
    true = targets[valid]

    def _rank(t):
        order = t.argsort()
        ranks = torch.zeros_like(t)
        ranks[order] = torch.arange(len(t), device=t.device, dtype=torch.float32)
        return ranks

    rank_pred = _rank(pred)
    rank_true = _rank(true)
    mean_p = rank_pred.mean()
    mean_t = rank_true.mean()
    cov = ((rank_pred - mean_p) * (rank_true - mean_t)).sum()
    std_p = ((rank_pred - mean_p) ** 2).sum().sqrt()
    std_t = ((rank_true - mean_t) ** 2).sum().sqrt()
    if std_p * std_t < 1e-8:
        return 0.0
    return (cov / (std_p * std_t)).item()


def train_one_epoch(model, dataset, optimizer, config, scaler):
    model.train()
    epoch_loss = 0.0
    epoch_main = 0.0
    epoch_aux = 0.0
    epoch_grad_norm = 0.0
    n_months = 0
    indices = np.random.permutation(len(dataset))
    for idx in indices:
        batch = dataset[idx]
        k0 = batch["k0"].to(device, non_blocking=True)
        k1 = batch["k1"].to(device, non_blocking=True)
        k0_miss = batch["k0_miss"].to(device, non_blocking=True)
        k1_miss = batch["k1_miss"].to(device, non_blocking=True)
        cids = batch["country_ids"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
        valid_masks = {k: v.to(device, non_blocking=True) for k, v in batch["valid_masks"].items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type):
            output = model(k0, k1, k0_miss, k1_miss, cids)
            loss, main_val, aux_val = compute_msrr_loss(output, targets, valid_masks, config)

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_grad_norm += grad_norm.item()

        epoch_loss += loss.item()
        epoch_main += main_val
        epoch_aux += aux_val
        n_months += 1

    n = max(n_months, 1)
    return epoch_loss / n, epoch_main / n, epoch_aux / n, epoch_grad_norm / n


@torch.no_grad()
def evaluate(model, dataset, config):
    model.eval()
    total_loss = 0.0
    total_corr_6m = 0.0
    total_corr_base_6m = 0.0
    total_corr_path2_6m = 0.0
    n_months = 0
    for idx in range(len(dataset)):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        valid_masks = {k: v.to(device) for k, v in batch["valid_masks"].items()}

        output = model(k0, k1, k0_miss, k1_miss, cids)
        loss, _, _ = compute_msrr_loss(output, targets, valid_masks, config)
        total_loss += loss.item()

        total_corr_6m += compute_rank_correlation(output["scores_6m"], targets["target_6m"], valid_masks["target_6m"])
        total_corr_base_6m += compute_rank_correlation(output["base_6m"], targets["target_6m"], valid_masks["target_6m"])
        total_corr_path2_6m += compute_rank_correlation(output["path2_6m"], targets["target_6m"], valid_masks["target_6m"])
        n_months += 1

    n = max(n_months, 1)
    return {"loss": total_loss / n, "rank_corr_6m": total_corr_6m / n, "rank_corr_base_6m": total_corr_base_6m / n, "rank_corr_path2_6m": total_corr_path2_6m / n}


def _model_parameter_breakdown(model):
    counts = {}
    seen_params = set()
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        for p in module.parameters():
            seen_params.add(id(p))
        counts[name] = int(n)
    extra = 0
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen_params and "." not in name:
            extra += p.numel()
    if extra > 0:
        counts["_top_level_parameters"] = int(extra)
    counts["_total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def _capture_tensor_shapes(model, config):
    n_k0 = len(k0_chars)
    n_k1 = len(k1_chars)
    dummy_k0 = torch.zeros(2, n_k0, device=device)
    dummy_k1 = torch.zeros(2, n_k1, device=device)
    dummy_k0_miss = torch.zeros(2, n_k0, device=device)
    dummy_k1_miss = torch.zeros(2, n_k1, device=device)
    dummy_cids = torch.zeros(2, dtype=torch.long, device=device)

    trace = []
    handles = []

    def _shape(t):
        if isinstance(t, torch.Tensor):
            return list(t.shape)
        if isinstance(t, (tuple, list)) and len(t) > 0:
            return _shape(t[0])
        return None

    def _make_hook(name, description):
        def hook(module, inputs, output):
            trace.append({"stage": name, "module": module.__class__.__name__, "input_shape": _shape(inputs), "output_shape": _shape(output), "description": description})

        return hook

    submodule_descriptions = {
        "k0_encoder": "K0 characteristic encoding to R^d",
        "k1_encoder": "K1 characteristic encoding to R^d",
        "k0_agg": "K0 attention-weighted aggregation to firm token",
        "k1_agg": "K1 attention-weighted aggregation to firm token",
        "base_head_6m": "Per firm base score head, 6 month horizon",
    }
    for name, module in model.named_children():
        desc = submodule_descriptions.get(name, "")
        if name == "path2_net":
            for i, block in enumerate(module.blocks):
                handles.append(block.register_forward_hook(_make_hook(f"path2_net.blocks.{i}", "Path 2 cross sectional Transformer block")))
        elif desc:
            handles.append(module.register_forward_hook(_make_hook(name, desc)))

    original_min_firms = model.min_firms
    model.min_firms = 1
    model.eval()
    try:
        with torch.no_grad():
            model(dummy_k0, dummy_k1, dummy_k0_miss, dummy_k1_miss, dummy_cids)
    finally:
        for h in handles:
            h.remove()
        model.min_firms = original_min_firms

    return trace


def _config_to_dict(config):
    d = asdict(config)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def train_variant(config, seed_idx=0):
    variant = config.encoding_variant
    seed_tag = f"seed{seed_idx}"
    w = 104
    bar = "=" * w
    sep = "-" * w

    def _block(lines):
        for line in lines:
            print(line)

    _block(
        [
            bar,
            f"Variant {variant}  Seed {seed_idx}",
            f"Path 1 d_model={config.d_model}  d_ff={config.d_ff}  n_mlp_layers={config.n_mlp_layers}  dropout={config.dropout}",
            f"Path 2 path2_n_layers={config.path2_n_layers}  path2_n_heads={config.path2_n_heads}  path2_d_ff={config.path2_d_ff}  min_firms_attention={config.min_firms_attention}",
            f"Optimiser lr={config.learning_rate:.2e}  wd={config.weight_decay:.2e}  " f"grad_clip={config.grad_clip}  patience={config.patience}",
            f"Loss MSRR (1 - r_pf)^2 + lambda_aux (1 - r_base)^2 + lambda_aux2 (1 - r_path2)^2  lambda_aux={config.lambda_aux}  lambda_aux2={config.lambda_aux2}",
        ]
    )

    train_ds = load_dataset(config.train_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df)
    val_ds = load_dataset(config.val_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df)

    model = DualPathTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_counts = _model_parameter_breakdown(model)
    tensor_shapes = _capture_tensor_shapes(model, config)

    _block([f" Data train_months={len(train_ds)}  val_months={len(val_ds)}", f" Model {n_params:,} trainable parameters", bar])

    col_header = f"{'Epoch':>5}{'TrnTotal':>9}{'TrnMain':>9}{'TrnAux':>9}  " f"{'ValLoss':>9}{'Corr6m':>8}{'Base6m':>8}{'Path26m':>8}  " f"{'LR':>9}{'GNorm':>8}"
    print(col_header)
    print(sep)
    sys.stdout.flush()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_loss = float("inf")
    best_epoch = 1
    best_val_metrics = None
    patience_counter = 0
    history = {"train_loss": [], "train_main": [], "train_aux": [], "val_loss": [], "val_corr_6m": [], "val_base_corr_6m": [], "val_path2_corr_6m": []}
    weights_path = config.results_dir / f"weights_{variant}_{seed_tag}.safetensors"
    scaler = torch.GradScaler(device.type)

    training_start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        if epoch <= config.warmup_epochs:
            warmup_lr = config.learning_rate * epoch / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss, train_main, train_aux, grad_norm = train_one_epoch(model, train_ds, optimizer, config, scaler)
        val_metrics = evaluate(model, val_ds, config)
        val_loss = val_metrics["loss"]
        val_corr_6m = val_metrics["rank_corr_6m"]
        val_base_corr_6m = val_metrics["rank_corr_base_6m"]
        val_path2_corr_6m = val_metrics["rank_corr_path2_6m"]

        if epoch > config.warmup_epochs:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_main"].append(train_main)
        history["train_aux"].append(train_aux)
        history["val_loss"].append(val_loss)
        history["val_corr_6m"].append(val_corr_6m)
        history["val_base_corr_6m"].append(val_base_corr_6m)
        history["val_path2_corr_6m"].append(val_path2_corr_6m)

        current_lr = optimizer.param_groups[0]["lr"]
        is_best = epoch > config.warmup_epochs and val_loss < best_val_loss - 1e-5
        marker = "  *" if is_best else ""

        row = (
            f"{epoch:>5}{train_loss:>9.6f}{train_main:>9.6f}{train_aux:>9.6f}  "
            f"{val_loss:>9.6f}{val_corr_6m:>8.4f}{val_base_corr_6m:>8.4f}{val_path2_corr_6m:>8.4f}  "
            f"{current_lr:>9.2e}{grad_norm:>8.4f}{marker}"
        )
        print(row)
        sys.stdout.flush()

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience_counter = 0
            safetensors_save(model.state_dict(), str(weights_path))
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(sep)
                print(f"Early stopping at epoch {epoch}  " f"(patience={config.patience}  best_epoch={best_epoch}  " f"best_val_loss={best_val_loss:.6f})")
                break

    training_time_seconds = time.time() - training_start_time

    del train_ds, val_ds
    gc.collect()

    final_epoch = len(history["train_loss"])
    model.load_state_dict(safetensors_load(str(weights_path)))
    test_ds = load_dataset(config.test_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df)
    test_metrics = evaluate(model, test_ds, config)
    del test_ds

    _block(
        [
            bar,
            f" Test Results  ({variant}, {seed_tag})",
            sep,
            f"{'Test MSRR loss':<22} {test_metrics['loss']:>10.6f}",
            f"{'Test rank corr 6m':<22}{test_metrics['rank_corr_6m']:>10.4f}",
            f"{'Test base corr 6m':<22}{test_metrics['rank_corr_base_6m']:>10.4f}",
            f"{'Test path2 corr 6m':<22}{test_metrics['rank_corr_path2_6m']:>10.4f}",
            sep,
            f"Best val loss {best_val_loss:.6f}  (epoch {best_epoch})",
            f"Stopped epoch {final_epoch}",
            f"Training time {training_time_seconds:.1f} seconds",
            sep,
            f"Weights {weights_path}",
        ]
    )

    model_architecture = {
        "encoding_variant": variant,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "dropout": config.dropout,
        "n_mlp_layers": config.n_mlp_layers,
        "path2_n_layers": config.path2_n_layers,
        "path2_n_heads": config.path2_n_heads,
        "path2_d_ff": config.path2_d_ff,
        "lambda_aux": config.lambda_aux,
        "lambda_aux2": config.lambda_aux2,
        "min_firms_attention": config.min_firms_attention,
        "n_k0_characteristics": len(k0_chars),
        "n_k1_characteristics": len(k1_chars),
        "n_countries": len(country_codes),
        "parameter_counts": parameter_counts,
        "tensor_shapes": tensor_shapes,
    }

    results_path = config.results_dir / f"metrics_{variant}_{seed_tag}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "variant": variant,
                "seed_idx": seed_idx,
                "seed_value": config.seed,
                "n_params": n_params,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "stopped_epoch": final_epoch,
                "training_time_seconds": training_time_seconds,
                "history": history,
                "val_metrics": best_val_metrics,
                "test_metrics": test_metrics,
                "config": _config_to_dict(config),
                "feature_columns": {"k0": k0_chars, "k1": k1_chars},
                "countries": {"country_to_id": col_meta.get("country_to_id", {}), "country_codes": list(country_codes)},
                "model_architecture": model_architecture,
            },
            f,
            indent=2,
        )

    _block([f"Metrics {results_path}", bar, ""])

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


hpt_dir = cfg.results_dir / "transformer-hpt"
base_seed = cfg.seed

for variant_name in ["linear", "per_feature", "ple", "periodic", "fourier"]:
    cfg.encoding_variant = variant_name

    hpt_path = hpt_dir / f"best_params_{variant_name}.json"
    if hpt_path.exists():
        with open(hpt_path, "r") as f:
            best_params = json.load(f)
        cfg.d_model = best_params["d_model"]
        cfg.d_ff = best_params["d_ff_derived"]
        cfg.dropout = best_params["dropout"]
        cfg.n_mlp_layers = best_params["n_mlp_layers"]
        cfg.path2_n_layers = best_params["path2_n_layers"]
        cfg.path2_n_heads = best_params["path2_n_heads"]
        cfg.path2_d_ff = best_params["path2_d_ff"]
        cfg.lambda_aux = best_params["lambda_aux"]
        cfg.lambda_aux2 = best_params["lambda_aux2"]
        cfg.learning_rate = best_params["lr"]
        cfg.weight_decay = best_params["weight_decay"]
        cfg.grad_clip = best_params["grad_clip"]
        if "periodic_num_freq" in best_params:
            cfg.periodic_num_freq = best_params["periodic_num_freq"]
        if "ple_num_bins" in best_params:
            cfg.ple_num_bins = best_params["ple_num_bins"]
        if "min_firms_attention" in best_params:
            cfg.min_firms_attention = best_params["min_firms_attention"]
        if "warmup_epochs" in best_params:
            cfg.warmup_epochs = best_params["warmup_epochs"]
        print(f"applied tuned hyperparameters for {variant_name} from {hpt_path}")
    else:
        print(f"no tuned hyperparameters found at {hpt_path}, using defaults")

    for seed_idx in range(cfg.n_seeds):
        cfg.seed = base_seed + seed_idx
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        train_variant(cfg, seed_idx=seed_idx)

print("All variants and seeds trained successfully")
