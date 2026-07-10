import gc
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
import typing

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tuning configuration

@dataclass
class TuningConfig:
    # Paths to processed data produced by the preprocessing pipeline
    train_path: Path = Path("data/processed/train.parquet")
    val_path: Path = Path("data/processed/val.parquet")
    col_metadata_path: Path = Path("data/processed/column_metadata.json")
    country_lookup_path: Path = Path("data/processed/country_lookup.parquet")
    results_dir: Path = Path("results/transformer/transformer-hpt")

    # Optuna study settings
    n_trials: int = 50
    n_startup_trials: int = 10
    n_warmup_steps: int = 6
    seed: int = 42
    db_name: str = "hpt_dual_path.db"

    # Training budget per trial (shorter than main training to allow more trials)
    max_epochs: int = 50
    patience: int = 8

    # Fixed architecture parameters not included in the search space
    min_firms_attention: int = 10

    # Encoding variants to tune
    variants: tuple = ("linear", "per_feature", "ple", "periodic", "fourier")


tcfg = TuningConfig()
tcfg.results_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(tcfg.seed)
np.random.seed(tcfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(tcfg.seed)


# Column setup from saved metadata

with open(tcfg.col_metadata_path, "r") as f:
    col_meta = json.load(f)

k0_chars = col_meta["retained_k0"]
k1_chars = col_meta["retained_k1"]

parquet_schema_cols = set(pq.read_schema(tcfg.train_path).names)

target_cols = ["target_6m"]

k0_feature_cols = [c for c in k0_chars if c in parquet_schema_cols]
k1_feature_cols = [c for c in k1_chars if c in parquet_schema_cols]
k0_miss_cols = [f"{c}_miss" for c in k0_chars if f"{c}_miss" in parquet_schema_cols]
k1_miss_cols = [f"{c}_miss" for c in k1_chars if f"{c}_miss" in parquet_schema_cols]

country_lookup_df = pd.read_parquet(tcfg.country_lookup_path)
country_lookup_df["eom"] = pd.to_datetime(country_lookup_df["eom"])

print(f"K0 characteristics, {len(k0_chars)}")
print(f"K1 characteristics, {len(k1_chars)}")
print(f"Countries, {len(col_meta['country_codes'])}")
print(f"Device, {device}")


# Dataset
class CrossSectionalDataset(Dataset):
    """Stores one tensor batch per calendar month containing the K0 and K1
    characteristic tensors, binary missingness flags, integer country identifiers,
    continuous return targets, and valid-observation masks."""

    def __init__(self, df, k0_cols, k1_cols, k0_miss, k1_miss,
                 target_col_list, country_lookup):
        dates = sorted(df["eom"].unique())
        self.monthly_data = []
        df = df.merge(country_lookup, on=["id", "eom"], how="left")
        df["country_id"] = df["country_id"].fillna(-1).astype(np.int16)
        for date in dates:
            group = df[df["eom"] == date]
            k0 = torch.tensor(group[k0_cols].values, dtype=torch.float32)
            k1 = torch.tensor(group[k1_cols].values, dtype=torch.float32)
            k0_m = torch.tensor(group[k0_miss].values, dtype=torch.float32)
            k1_m = torch.tensor(group[k1_miss].values, dtype=torch.float32)
            cids = torch.tensor(group["country_id"].values, dtype=torch.long)
            firm_ids = torch.tensor(group["id"].values, dtype=torch.long)
            targets = {}
            valid_masks = {}
            for tc in target_col_list:
                vals = group[tc].values.copy().astype(np.float32)
                valid_mask = ~np.isnan(vals)
                vals[~valid_mask] = 0.0
                targets[tc] = torch.tensor(vals, dtype=torch.float32)
                valid_masks[tc] = torch.tensor(valid_mask, dtype=torch.bool)
            self.monthly_data.append({
                "k0": k0, "k1": k1, "k0_miss": k0_m, "k1_miss": k1_m,
                "country_ids": cids, "firm_ids": firm_ids,
                "targets": targets, "valid_masks": valid_masks,
            })
        del df
        gc.collect()

    def __len__(self):
        return len(self.monthly_data)

    def __getitem__(self, idx):
        return self.monthly_data[idx]


def load_dataset(path, k0_cols, k1_cols, k0_miss, k1_miss, target_col_list, country_lookup):
    """Load a processed parquet split and return a CrossSectionalDataset."""
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
    return CrossSectionalDataset(
        df, k0_cols, k1_cols, k0_miss, k1_miss,
        target_col_list, country_lookup
    )


# Architecture
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
        value, gate = self.fc2(h).chunk(2, dim=-1)
        return self.layer_norm(residual + value * torch.sigmoid(gate))


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
    def __init__(self, n_features, d_model, num_bins=16):
        super().__init__()
        self.register_buffer("boundaries", torch.linspace(-0.5, 0.5, num_bins + 1))
        self.feature_weights = nn.Parameter(torch.randn(n_features, num_bins, d_model) * 0.02)

    def forward(self, x):
        boundaries = typing.cast(torch.Tensor, self.boundaries)
        t_lower = boundaries[:-1]
        t_upper = boundaries[1:]
        activations = torch.clamp(
            (x.unsqueeze(-1) - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0
        )
        return torch.einsum("bnk,nkd->bnd", activations, self.feature_weights)


class PeriodicEncoder(nn.Module):
    def __init__(self, n_features, d_model, num_freq=32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.phi = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.proj = nn.Linear(num_freq, d_model)

    def forward(self, x):
        return self.proj(torch.sin(x.unsqueeze(-1) * self.omega.unsqueeze(0) + self.phi.unsqueeze(0)))


class FourierEncoder(nn.Module):
    def __init__(self, n_features, d_model, num_freq=32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.proj = nn.Linear(num_freq * 2, d_model)

    def forward(self, x):
        scaled = x.unsqueeze(-1) * self.omega.unsqueeze(0)
        return self.proj(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1))


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


class MultiHeadAttention(nn.Module):
    """Kelly, Kuznetsov, Malamud, and Xu (2025), equation 13. Bilinear
    per-head attention sigma(Y W_h Y') Y V_h, summed across heads, dense
    softmax with no top-k truncation. No dropout, matching the paper.
    No 1/sqrt(d_model) scaling; a diagnostic on a real trained checkpoint
    showed combining scaling with cross-sectional rank normalisation of the
    input (see _cross_sectional_soft_norm) undid most of the normalisation's
    benefit. The normalisation is what fixes the attention collapse; see
    nonlinear_dualapproach.py for the full rationale."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
            for _ in range(n_heads)
        ])
        self.v = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_heads)
        ])

    def forward(self, y):
        head_outputs = []
        head_weights = []
        for h in range(self.n_heads):
            scores = y @ self.w[h] @ y.t()
            weights = F.softmax(scores, dim=-1)
            head_outputs.append(weights @ self.v[h](y))
            head_weights.append(weights)
        out = torch.stack(head_outputs, dim=0).sum(dim=0)
        return out, torch.stack(head_weights, dim=0)


class TransformerBlock(nn.Module):
    """Kelly, Kuznetsov, Malamud, and Xu (2025), equations 13-17. Bare
    residual connections with no LayerNorm, single hidden layer ReLU
    feed-forward: T(Y) = F^R(A^R(Y)). No dropout, matching the paper."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        attn_out, attn_w = self.attention(x)
        a_r = attn_out + x
        f = self.w2(F.relu(self.w1(a_r)))
        f_r = f + a_r
        return f_r, attn_w


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
        return (encoded * weights.unsqueeze(-1)).sum(dim=1), weights


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


def _cross_sectional_soft_norm(z_c, temperature=2.0):
    """Normalises each of z_c's d_model columns across the firm axis, then
    bounds the result to the open interval (-0.5, 0.5) via a scaled
    hyperbolic tangent, so gradient from Path 2's contribution to the loss
    can reach the shared encoder and aggregation layers, unlike the earlier
    hard rank based normalisation. See nonlinear_dualapproach.py for the
    full rationale."""
    n = z_c.shape[0]
    if n <= 1:
        return torch.zeros_like(z_c)
    mean = z_c.mean(dim=0, keepdim=True)
    std = z_c.std(dim=0, keepdim=True, unbiased=False)
    z_score = (z_c - mean) / (std + 1e-6)
    return 0.5 * torch.tanh(z_score / temperature)


class DualPathTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_k0 = len(k0_chars)
        n_k1 = len(k1_chars)
        self.k0_encoder = build_encoder(
            config.encoding_variant, n_k0, config.d_model,
            ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq,
        )
        self.k1_encoder = build_encoder(
            config.encoding_variant, n_k1, config.d_model,
            ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq,
        )
        self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)
        self.k1_static_emb = nn.Parameter(torch.randn(n_k1, config.d_model) * 0.02)
        self.k0_agg = AttentiveAggregation(config.d_model)
        self.k1_agg = AttentiveAggregation(config.d_model)
        self.base_head_6m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)
        # z is cross-sectionally rank-normalised within each country group
        # before Path 2's attention, via _cross_sectional_soft_norm; see
        # nonlinear_dualapproach.py for the full rationale, including why a
        # z-score based normalisation was tried first and rejected.
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff,
                             config.dropout)
            for _ in range(config.n_layers)
        ])
        self.adj_head_6m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.min_firms = config.min_firms_attention

    def _encode_firms(self, k0, k1, k0_miss, k1_miss):
        k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
        k0_token, _ = self.k0_agg(k0_encoded, k0_miss)
        k1_encoded = self.k1_encoder(k1) + self.k1_static_emb.unsqueeze(0)
        k1_token, _ = self.k1_agg(k1_encoded, k1_miss)
        return k0_token + k1_token

    def forward(self, k0, k1, k0_miss, k1_miss, country_ids):
        z = self._encode_firms(k0, k1, k0_miss, k1_miss)
        base_6m = self.base_head_6m(z)
        adj_6m = torch.zeros_like(base_6m)
        for cid in country_ids.unique():
            mask = country_ids == cid
            if mask.sum() < self.min_firms:
                continue
            z_c = z[mask]
            z_c = _cross_sectional_soft_norm(z_c)
            for block in self.blocks:
                z_c, _ = block(z_c)
            adj_6m[mask] = self.adj_head_6m(z_c).squeeze(-1)
        return {
            "scores_6m": base_6m + adj_6m,
            "base_6m": base_6m,
            "adj_6m": adj_6m,
        }


# Training components

def _differentiable_long_short_return(scores, returns, valid_mask):
    """Differentiable per cross-section long short portfolio return. Firms
    with a valid forward return are split into long and short legs by the
    cross sectional mean of the score. Within each leg the weights are
    given by a softmax over the demeaned score, so each leg sums to unity
    and the portfolio is dollar neutral by construction."""
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
    """Maximum Sharpe Ratio Regression loss for the six month long short
    portfolio. Following Kelly, Kuznetsov, Malamud, and Xu (2025), the
    main loss is the squared deviation of the realised portfolio return
    from a unit target. The auxiliary loss applies the same objective to
    the Path 1 base scores so that the per firm encoder remains
    independently predictive."""
    device_ = output["scores_6m"].device
    target = targets["target_6m"]
    valid = valid_masks["target_6m"]
    if valid.sum() < 10:
        return torch.tensor(0.0, device=device_, requires_grad=True)
    r_pf = _differentiable_long_short_return(output["scores_6m"], target, valid)
    r_base = _differentiable_long_short_return(output["base_6m"], target, valid)
    main_loss = (1.0 - r_pf).pow(2)
    aux_loss = (1.0 - r_base).pow(2)
    return main_loss + config.lambda_aux * aux_loss


def compute_rank_correlation(scores, targets, valid_mask):
    """Spearman rank correlation between predicted scores and realised returns."""
    if valid_mask.sum() < 10:
        return 0.0
    pred = scores[valid_mask]
    true = targets[valid_mask]

    def _rank(t):
        order = t.argsort()
        ranks = torch.zeros_like(t)
        ranks[order] = torch.arange(len(t), device=t.device, dtype=torch.float32)
        return ranks

    rp, rt = _rank(pred), _rank(true)
    mp, mt = rp.mean(), rt.mean()
    cov = ((rp - mp) * (rt - mt)).sum()
    sp = ((rp - mp) ** 2).sum().sqrt()
    st = ((rt - mt) ** 2).sum().sqrt()
    if sp * st < 1e-8:
        return 0.0
    return (cov / (sp * st)).item()


def train_one_epoch(model, dataset, optimizer, config, scaler):
    model.train()
    total_loss = 0.0
    n = 0
    for idx in np.random.permutation(len(dataset)):
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
            loss = compute_msrr_loss(output, targets, valid_masks, config)
        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, dataset, config):
    """Evaluate the model on a dataset and return the validation msrr
    loss together with the 6 month combined, base, and adjustment score
    rank correlations. The base score correlation quantifies the
    predictive content of the per firm encoder alone, before the cross
    sectional adjustment is added, and the adjustment correlation
    quantifies the predictive content of path 2 in isolation, which
    together with the base correlation shows whether the cross sectional
    block is contributing meaningful information beyond path 1 rather
    than merely reproducing it."""
    model.eval()
    total_loss = 0.0
    total_corr_6m = 0.0
    total_corr_base_6m = 0.0
    total_corr_adj_6m = 0.0
    n = 0
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
        total_loss += compute_msrr_loss(output, targets, valid_masks, config).item()
        total_corr_6m += compute_rank_correlation(
            output["scores_6m"], targets["target_6m"], valid_masks["target_6m"]
        )
        total_corr_base_6m += compute_rank_correlation(
            output["base_6m"], targets["target_6m"], valid_masks["target_6m"]
        )
        total_corr_adj_6m += compute_rank_correlation(
            output["adj_6m"], targets["target_6m"], valid_masks["target_6m"]
        )
        n += 1
    m = max(n, 1)
    return {
        "loss": total_loss / m,
        "rank_corr_6m": total_corr_6m / m,
        "rank_corr_base_6m": total_corr_base_6m / m,
        "rank_corr_adj_6m": total_corr_adj_6m / m,
    }


# Hyperparameter search space

def sample_hyperparameters(trial, variant):

    d_model = trial.suggest_categorical("d_model", [64, 96, 128])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    d_ff_mult = trial.suggest_categorical("d_ff_mult", [2, 4])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.01, 0.4)
    # top_k_attention removed from the search space: MultiHeadAttention
    # (Kelly, Kuznetsov, Malamud, and Xu, 2025) uses dense softmax attention
    # with no truncation, so sampling this would waste trial diversity on a
    # parameter the model no longer reads.
    n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 3)
    lambda_aux = trial.suggest_float("lambda_aux", 0.1, 0.5)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    grad_clip = trial.suggest_float("grad_clip", 0.1, 5.0)

    # Variant-specific encoding parameters
    ple_num_bins = trial.suggest_categorical("ple_num_bins", [8, 16, 32]) if variant == "ple" else 16
    periodic_num_freq = (
        trial.suggest_categorical("periodic_num_freq", [16, 32, 64])
        if variant in ("periodic", "fourier") else 32
    )

    return {
        "d_model": d_model, "n_heads": n_heads,
        "d_ff": d_model * d_ff_mult, "n_layers": n_layers,
        "dropout": dropout,
        "n_mlp_layers": n_mlp_layers, "lambda_aux": lambda_aux,
        "learning_rate": lr, "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "ple_num_bins": ple_num_bins, "periodic_num_freq": periodic_num_freq,
    }


# Config used inside the objective function

@dataclass
class TrialConfig:
    encoding_variant: str = "linear"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    top_k_attention: int = 50
    n_mlp_layers: int = 2
    lambda_aux: float = 0.3
    min_firms_attention: int = 10
    # linear warmup over the first warmup_epochs epochs, mirroring
    # nonlinear_dualapproach.py. Without this, every trial trained its
    # first epoch at the full sampled lr, which ranges up to 5e-3, with no
    # ramp, exposing fresh random weights to the same instability the
    # warmup was introduced to prevent in the main training script.
    warmup_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    ple_num_bins: int = 16
    periodic_num_freq: int = 32
    max_epochs: int = 50
    patience: int = 8
    target_vol: float = 0.10
    vol_lookback: int = 6
    max_leverage_long_short: float = 3.0
    max_position_weight: float = 0.05


@torch.no_grad()
def _validation_long_short_sharpe(model, dataset, config, rebalance_freq=6, tc_bps=25, periods_per_year=2):
    """Run a long short mean split portfolio simulation against the
    validation dataset and return its annualised Sharpe ratio. The leg
    construction mirrors _differentiable_long_short_return exactly: firms
    with a valid forward return are split at the cross sectional mean
    score, not a fixed quantile, so the quantity Optuna optimises is
    computed on the same portfolio definition the model is trained
    against and the same construction eval_dual_path.py reports at
    test time."""
    model.eval()
    raw_ls_returns = []
    portfolio_returns = []
    prev_long_ids = None
    prev_short_ids = None

    def _capped_softmax(scores, max_w):
        n = scores.shape[0]
        if n == 0:
            return scores.new_zeros(0)
        if max_w <= 1.0 / n + 1e-12:
            return scores.new_full((n,), 1.0 / n)
        weights = F.softmax(scores, dim=0)
        for _ in range(20):
            over = weights > max_w
            if not over.any():
                break
            excess = (weights[over] - max_w).sum()
            weights = torch.where(over, torch.full_like(weights, max_w), weights)
            residual = ~over
            residual_total = weights[residual].sum()
            if residual_total <= 1e-12:
                break
            weights = torch.where(
                residual,
                weights * (1.0 + excess / residual_total),
                weights,
            )
        return weights

    def _turnover(prev_ids, curr_ids):
        if prev_ids is None:
            return 0.0
        prev = set(prev_ids.tolist())
        curr = set(curr_ids.tolist())
        if not curr:
            return 0.0
        new_in = len(curr - prev)
        exited = len(prev - curr)
        return (new_in + exited) / max(len(curr), 1)

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        firm_ids = batch["firm_ids"]
        out = model(k0, k1, k0_miss, k1_miss, cids)
        scores = out["scores_6m"]
        raw_returns = batch["targets"]["target_6m"].to(device)
        valid = batch["valid_masks"]["target_6m"].to(device)

        valid_scores = scores[valid]
        valid_returns = raw_returns[valid]
        n_valid = valid_scores.shape[0]
        if n_valid < 10:
            continue
        mean_score = valid_scores.mean()
        long_mask = valid_scores > mean_score
        short_mask = ~long_mask
        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue

        long_w = _capped_softmax(valid_scores[long_mask] - mean_score, config.max_position_weight)
        short_w = _capped_softmax(mean_score - valid_scores[short_mask], config.max_position_weight)
        long_ret = (long_w * valid_returns[long_mask]).sum().item()
        short_ret = (short_w * valid_returns[short_mask]).sum().item()
        ls_ret = long_ret - short_ret

        valid_firm_ids = firm_ids[valid.cpu()]
        long_ids = valid_firm_ids[long_mask.cpu()].numpy()
        short_ids = valid_firm_ids[short_mask.cpu()].numpy()
        lt = _turnover(prev_long_ids, long_ids)
        st = _turnover(prev_short_ids, short_ids)
        base_turnover = lt + st
        if len(raw_ls_returns) >= config.vol_lookback:
            recent = np.array(raw_ls_returns[-config.vol_lookback:])
            realised_vol = recent.std(ddof=1) * np.sqrt(periods_per_year)
            leverage = config.target_vol / max(realised_vol, 1e-6)
            leverage = float(np.clip(
                leverage,
                1.0 / config.max_leverage_long_short,
                config.max_leverage_long_short,
            ))
        else:
            leverage = 1.0
        tc = leverage * base_turnover * tc_bps / 10000.0
        portfolio_returns.append(leverage * ls_ret - tc)
        raw_ls_returns.append(ls_ret)
        prev_long_ids = long_ids
        prev_short_ids = short_ids

    if not portfolio_returns:
        return 0.0
    returns_arr = np.array(portfolio_returns, dtype=float)
    ann_ret = float(returns_arr.mean() * periods_per_year)
    ann_vol = float(returns_arr.std(ddof=1) * np.sqrt(periods_per_year))
    return float(ann_ret / max(ann_vol, 1e-8))


# Objective function

def objective(trial, variant, train_ds, val_ds, tuning_config):
    params = sample_hyperparameters(trial, variant)

    cfg = TrialConfig()
    cfg.encoding_variant = variant
    cfg.min_firms_attention = tuning_config.min_firms_attention
    cfg.max_epochs = tuning_config.max_epochs
    cfg.patience = tuning_config.patience
    for key, val in params.items():
        setattr(cfg, key, val)

    model = DualPathTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    # Within trial early stopping is driven by the validation MSRR loss,
    # which aligns the model selection signal with the training objective.
    # ReduceLROnPlateau therefore runs in min mode.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = torch.GradScaler(device.type)

    best_val_loss = float("inf")
    best_epoch_metrics = {}
    best_state_dict = None
    patience_counter = 0
    epoch_log_path = tuning_config.results_dir / f"epoch_history_{variant}.csv"

    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        if epoch <= cfg.warmup_epochs:
            warmup_lr = cfg.learning_rate * epoch / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss = train_one_epoch(model, train_ds, optimizer, cfg, scaler)
        val_metrics = evaluate(model, val_ds, cfg)
        val_loss = val_metrics["loss"]
        val_corr_6m = val_metrics["rank_corr_6m"]
        val_base_corr_6m = val_metrics["rank_corr_base_6m"]
        val_adj_corr_6m = val_metrics["rank_corr_adj_6m"]

        if epoch > cfg.warmup_epochs:
            scheduler.step(val_loss)

        epoch_row = {
            "variant": variant,
            "trial_number": trial.number + 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "corr_6m": val_corr_6m,
            "base_corr_6m": val_base_corr_6m,
            "adj_corr_6m": val_adj_corr_6m,
        }
        pd.DataFrame([epoch_row]).to_csv(
            epoch_log_path, mode="a", header=not epoch_log_path.exists(), index=False
        )

        # Report the validation rank correlation to the pruner at every
        # epoch. The pruning signal remains rank correlation rather than
        # the MSRR loss because the pruner expects a maximise direction
        # signal, and the rank correlation is bounded in [-1, 1] which
        # makes the median comparison well behaved across trials. The
        # within trial best epoch is still selected on the MSRR loss.
        trial.report(val_corr_6m, epoch)
        if trial.should_prune():
            for key, val in epoch_row.items():
                if key not in ("variant", "trial_number"):
                    trial.set_user_attr(f"last_{key}", val)
            trial.set_user_attr("n_epochs_trained", epoch)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        # epoch > cfg.warmup_epochs excludes the ramp-up window from ever
        # becoming the saved checkpoint. Without this, an epoch trained at a
        # still-low warmup lr could register a deceptively low val_loss before
        # the model has done any meaningful learning, and that checkpoint
        # would then be reloaded below for the Sharpe ratio Optuna optimises.
        if epoch > cfg.warmup_epochs and val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch_metrics = dict(epoch_row)
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    for key, val in best_epoch_metrics.items():
        if key not in ("variant", "trial_number"):
            trial.set_user_attr(f"best_{key}", val)
    trial.set_user_attr("n_epochs_trained", epoch)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    val_sharpe = _validation_long_short_sharpe(model, val_ds, cfg)
    trial.set_user_attr("val_sharpe_ls", val_sharpe)
    trial.set_user_attr("best_val_loss", best_val_loss)

    del model
    del best_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_sharpe


# Terminal output helpers

def _print_trial_row(trial, study, w):
    """Print one summary line after a trial completes or is pruned, 
    including the validation long short Sharpe ratio, the base 
    f_firm 6 month rank correlation, and the running best Sharpe ratio"""
    pruned = trial.state == optuna.trial.TrialState.PRUNED
    value = float("nan") if pruned else trial.value
    try:
        best = study.best_value
        best_t = study.best_trial.number + 1
    except Exception:
        best, best_t = float("nan"), 0
    status = "PRUNED" if pruned else "DONE"
    marker = "  *" if (not pruned and trial.number == study.best_trial.number) else ""
    p = trial.params
    d_model = p.get("d_model", "?")
    n_heads = p.get("n_heads", "?")
    lr = p.get("lr", float("nan"))
    n_layers = p.get("n_layers", "?")
    ua = trial.user_attrs
    base_corr_6m = ua.get("best_base_corr_6m", ua.get("last_base_corr_6m", float("nan")))
    adj_corr_6m = ua.get("best_adj_corr_6m", ua.get("last_adj_corr_6m", float("nan")))
    print(
        f"{trial.number + 1:>5}{value:>8.4f}{base_corr_6m:>8.4f}{adj_corr_6m:>8.4f}{best:>8.4f}  "
        f"d={d_model:<3} h={n_heads:<2} L={n_layers} lr={lr:.2e}{status:<6} {marker}"
    )
    sys.stdout.flush()


def _print_best_params(study, variant, results_dir):
    try:
        best = study.best_trial
    except Exception:
        print("No completed trials")
        return

    w = 80
    bar = "=" * w
    sep = "-" * w
    print(bar)
    print(f"Best trial for variant: {variant}")
    print(f"Trial number: {best.number + 1}  |  Val Sharpe LS: {best.value:.6f}")
    print(sep)
    for key, val in best.params.items():
        if isinstance(val, float):
            print(f"{key:<25} {val:.6g}")
        else:
            print(f"{key:<25} {val}")
    print(sep)
    print(f"{'d_ff (derived)':<25} {best.params['d_model'] * best.params['d_ff_mult']}")

    # Combined and base score rank correlations at the best epoch, recorded
    # as user attributes by the objective function. The base value quantifies
    # the f_firm contribution alone, before the cross sectional adjustment.
    ua = best.user_attrs
    print(sep)
    print(f"{'best epoch':<25} {ua.get('best_epoch', 'n/a')}")
    print(f"{'n epochs trained':<25} {ua.get('n_epochs_trained', 'n/a')}")
    print(f"{'train loss (best ep)':<25} {ua.get('best_train_loss', float('nan')):.6f}")
    print(f"{'val MSRR (best ep)':<25} {ua.get('best_val_loss', float('nan')):.6f}")
    print(f"{'corr 6m, combined':<25} {ua.get('best_corr_6m', float('nan')):.6f}")
    print(f"{'corr 6m, base f_firm':<25} {ua.get('best_base_corr_6m', float('nan')):.6f}")
    print(f"{'corr 6m, adjustment':<25} {ua.get('best_adj_corr_6m', float('nan')):.6f}")
    print(f"{'val Sharpe LS':<25} {ua.get('val_sharpe_ls', float('nan')):.6f}")

    # Study statistics
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(sep)
    print(f"   Completed trials: {n_complete}  |  Pruned trials: {n_pruned}")

    values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if values:
        print(f"   Val Sharpe LS  mean={np.mean(values):.4f}  std={np.std(values):.4f}  "
              f"min={np.min(values):.4f}  max={np.max(values):.4f}")
    print(bar)

    out_path = results_dir / f"best_params_{variant}.json"
    payload = dict(best.params)
    payload["d_ff_derived"] = best.params["d_model"] * best.params["d_ff_mult"]
    payload["best_val_sharpe_ls"] = best.value
    payload["trial_number"] = best.number + 1
    for key in [
        "best_epoch", "n_epochs_trained",
        "best_train_loss", "best_val_loss",
        "best_corr_6m", "best_base_corr_6m", "best_adj_corr_6m",
        "val_sharpe_ls",
    ]:
        if key in ua:
            payload[key] = ua[key]
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Best params saved, {out_path}")
    


def _save_trials_dataframe(study, variant, results_dir):
    """Export the full trial history for one variant to a csv file. The
    dataframe returned by optuna includes the optuna objective value (the
    combined 6 month rank correlation), every sampled hyperparameter such as
    d_model, and the user attributes recorded by the objective function,
    which cover the base f_firm rank correlations, the validation loss, the
    training loss, and the epoch at which the best value was reached. This
    table is the primary input for any cross trial analysis, such as relating
    the embedding dimension to the gap between the combined score and the
    base score rank correlations."""
    df = study.trials_dataframe()
    if df.empty:
        print("  No trials to export.")
        return df
    df.insert(0, "variant", variant)
    out_path = results_dir / f"trials_{variant}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Trial history saved, {out_path}")
    
    return df


def run_study(variant, train_ds, val_ds, tuning_config):
    """Create or resume an Optuna study for one encoding variant and run
    n_trials trials. The study is persisted to SQLite at results/hpt_dual_path.db
    so that completed trials are not lost if the process is interrupted.
    n_jobs is set to 1 to prevent threading race conditions in the single-GPU
    forward pass and dataset access."""
    db_path = tuning_config.results_dir / tuning_config.db_name
    storage = f"sqlite:///{db_path}"
    study_name = f"dual_path_{variant}"

    pruner = MedianPruner(
        n_startup_trials=tuning_config.n_startup_trials,
        n_warmup_steps=tuning_config.n_warmup_steps,
        interval_steps=1,
    )
    sampler = TPESampler(seed=tuning_config.seed)

    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        storage=storage, load_if_exists=True,
        pruner=pruner, sampler=sampler,
    )

    n_existing = len(study.trials)
    n_remaining = max(0, tuning_config.n_trials - n_existing)

    w = 80
    bar = "=" * w
    sep = "-" * w
    print(bar)
    print(f"Study, {study_name}")
    print(f"Target, maximise validation long short Sharpe ratio")
    print(f"Total trials, {tuning_config.n_trials}")
    print(f"max_epochs per trial, {tuning_config.max_epochs}  |  patience, {tuning_config.patience}")
    print(f"n_startup_trials, {tuning_config.n_startup_trials}  |  n_warmup_steps, {tuning_config.n_warmup_steps}")
    print(f"Database, {db_path}")
    print(sep)
    print(f"{'Trial':>5}{'Sharpe':>8}{'Base6m':>8}{'Adj6m':>8}{'Best':>8}{'Params (key)':>30}  Status")
    print(sep)

    if n_remaining == 0:
        print("All trials already completed")
    else:
        def _callback(study, trial):
            _print_trial_row(trial, study, w)

        study.optimize(
            lambda trial: objective(trial, variant, train_ds, val_ds, tuning_config),
            n_trials=n_remaining,
            n_jobs=1,
            callbacks=[_callback],
            gc_after_trial=True,
        )

    _print_best_params(study, variant, tuning_config.results_dir)
    _save_trials_dataframe(study, variant, tuning_config.results_dir)
    return study


# Main execution

print("Loading training and validation datasets")
print("(datasets are loaded once and shared across all trials for each variant)")

train_ds = load_dataset(
    tcfg.train_path, k0_feature_cols, k1_feature_cols,
    k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df,
)
val_ds = load_dataset(
    tcfg.val_path, k0_feature_cols, k1_feature_cols,
    k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df,
)
print(f"Train months, {len(train_ds)}  |  Val months, {len(val_ds)}")


all_studies = {}
for variant in tcfg.variants:
    study = run_study(variant, train_ds, val_ds, tcfg)
    all_studies[variant] = study

# Final cross-variant summary

w = 90
bar = "=" * w
sep = "-" * w
print(bar)
print(" Cross-variant summary")
print(sep)
print(f"{'Variant':<14}{'Sharpe':>10}{'Best Sharpe':>13}{'AdjCorr':>10}{'Trial':>6}{'d_model':>7}{'n_heads':>7}{'lr':>10}{'n_layers':>8}")
print(sep)
for variant, study in all_studies.items():
    try:
        bt = study.best_trial
        p = bt.params
        ua = bt.user_attrs
        base_corr_6m = ua.get("best_base_corr_6m", float("nan"))
        adj_corr_6m = ua.get("best_adj_corr_6m", float("nan"))
        print(
            f"  {variant:<14}  {bt.value:>10.6f}  {base_corr_6m:>13.6f}  {adj_corr_6m:>10.6f}  {bt.number + 1:>6}  "
            f"{p.get('d_model', '?'):>7}  {p.get('n_heads', '?'):>7}  "
            f"{p.get('lr', float('nan')):>10.2e}  {p.get('n_layers', '?'):>8}"
        )
    except Exception:
        print(f"  {variant:<14}  {'no completed trials':>40}")
print(bar)

# Master trial history across all variants. Each row corresponds to one
# trial and carries its sampled hyperparameters (including d_model), the
# combined 6 month rank correlation returned to optuna, and the base f_firm
# rank correlations for all three horizons recorded as user attributes. This
# table supports a direct comparison of how the embedding dimension relates
# to the combined score and to the base score alone.
all_trials_frames = []
for variant, study in all_studies.items():
    df = study.trials_dataframe()
    if df.empty:
        continue
    df.insert(0, "variant", variant)
    all_trials_frames.append(df)

if all_trials_frames:
    master_df = pd.concat(all_trials_frames, ignore_index=True)
    master_path = tcfg.results_dir / "all_trials_summary.csv"
    master_df.to_csv(master_path, index=False)
    print(f"Master trial summary saved, {master_path}")
    print(f"Columns, {list(master_df.columns)}")
    print(bar)
