## Portfolio Transformer with Four Feature Encoding Variants

import gc
import json
import math
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

### Configuration

@dataclass
class Config:
	train_path: Path = Path("../data/processed/train.parquet")
	val_path: Path = Path("../data/processed/val.parquet")
	test_path: Path = Path("../data/processed/test.parquet")
	results_dir: Path = Path("../results")

	d_model: int = 64
	n_heads: int = 4
	n_layers: int = 2
	d_ff: int = 128
	dropout: float = 0.1

	top_k_attention: int = 50
	time2vec_dim: int = 64
	ple_num_bins: int = 16
	periodic_num_freq: int = 32

	learning_rate: float = 1e-4
	weight_decay: float = 1e-5
	max_epochs: int = 100
	patience: int = 15
	grad_clip: float = 1.0

	lambda_3m: float = 0.2
	lambda_6m: float = 0.5
	lambda_12m: float = 0.3

	encoding_variant: str = "linear"
	seed: int = 24

cfg = Config()
cfg.results_dir.mkdir(parents = True, exist_ok = True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

### Column Classification

with open("../jsons/train_columns.json", "r") as f:
	all_columns = json.load(f)

miss_flags = [c for c in all_columns if c.endswith("_miss")]
miss_bases = [c.replace("_miss", "") for c in miss_flags]
non_miss = [c for c in all_columns if not c.endswith("_miss")]

lag12_cols = [c for c in non_miss if c.endswith("_lag12")]
lag12_bases = [c.replace("_lag12", "") for c in lag12_cols]

K1_CHARS = sorted([c for c in lag12_bases if c in non_miss])
all_chars = sorted([c for c in miss_bases if c in non_miss])
K0_CHARS = sorted([c for c in all_chars if c not in K1_CHARS])

LAG_SUFFIXES = ["", "_lag12", "_lag24", "_lag36", "_lag48", "_lag60"]
LAG_POSITIONS = [0, 12, 24, 36, 48, 60]

k0_feature_cols = K0_CHARS.copy()
k1_feature_cols = []
for char in K1_CHARS:
	for suffix in LAG_SUFFIXES:
		k1_feature_cols.append(char + suffix)

target_cols = ["target_3m", "target_6m", "target_12m"]

print(f"K0 characteristics (current only): {len(K0_CHARS)}")
print(f"K1 characteristics (with lags): {len(K1_CHARS)}")
print(f"K1 feature columns (K1 x 6 lags): {len(k1_feature_cols)}")
print(f"Missingness flags: {len(miss_flags)}")
print(f"Total model input features: {len(k0_feature_cols) + len(k1_feature_cols) + len(miss_flags)}")

### Dataset and Data Loading

class CrossSectionalDataset(Dataset):
	"""
	Dataset that groups firm observations by month for cross-sectional attention.
	Each item returned is an entire cross-section (all firms in one month).
	Targets are continuous returns for regression.
	"""

	def __init__(self, df, k0_cols, k1_cols, miss_cols, target_cols):
		self.target_col_names = target_cols

		dates = sorted(df["eom"].unique())
		self.monthly_data = []

		n_k1 = len(K1_CHARS)
		for date in dates:
			group = df[df["eom"] == date]

			k0 = torch.tensor(group[k0_cols].values, dtype = torch.float32)
			k1_raw = group[k1_cols].values.astype(np.float32)
			k1 = torch.tensor(k1_raw.reshape(len(group), n_k1, 6), dtype = torch.float32)
			miss = torch.tensor(group[miss_cols].values, dtype = torch.float32)

			targets = {}
			valid_masks = {}
			for tc in target_cols:
				vals = group[tc].values.copy().astype(np.float32)
				valid_mask = ~np.isnan(vals)
				vals[~valid_mask] = 0.0
				targets[tc] = torch.tensor(vals, dtype = torch.float32)
				valid_masks[tc] = torch.tensor(valid_mask, dtype = torch.bool)

			self.monthly_data.append({
				"k0": k0,
				"k1": k1,
				"miss": miss,
				"targets": targets,
				"valid_masks": valid_masks,
				"n_firms": len(group),
			})

		del df
		gc.collect()

	def __len__(self):
		return len(self.monthly_data)

	def __getitem__(self, idx):
		return self.monthly_data[idx]


def load_split(path, k0_cols, k1_cols, miss_cols, target_cols):
	required = k0_cols + k1_cols + miss_cols + target_cols + ["eom"]
	df = pd.read_parquet(path, columns = required)

	for col in k0_cols + k1_cols + miss_cols:
		df[col] = df[col].fillna(0.0)

	return CrossSectionalDataset(df, k0_cols, k1_cols, miss_cols, target_cols)

## Architecture Components

### Time2Vec Temporal Encoding

class Time2Vec(nn.Module):
	def __init__(self, d_out):
		super().__init__()
		self.d_out = d_out
		self.omega = nn.Parameter(torch.randn(d_out))
		self.phi = nn.Parameter(torch.randn(d_out))

	def forward(self, lag_position):
		lag = lag_position.float().unsqueeze(-1)
		raw = self.omega * lag + self.phi
		out = torch.zeros_like(raw)
		out[..., 0] = raw[..., 0]
		out[..., 1:] = torch.sin(raw[..., 1:])
		return out

### Gated Residual Network

class GRN(nn.Module):
	def __init__(self, d_model, d_ff, dropout = 0.1):
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
		value, gate = gated.chunk(2, dim = -1)
		h = value * torch.sigmoid(gate)
		return self.layer_norm(residual + h)

### Feature Encoding Variants

class LinearEncoder(nn.Module):
	def __init__(self, n_features, d_model):
		super().__init__()
		self.weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
		self.biases = nn.Parameter(torch.zeros(n_features, d_model))

	def forward(self, x):
		return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class PerFeatureTokeniser(nn.Module):
	def __init__(self, n_features, d_model):
		super().__init__()
		self.projections = nn.Parameter(torch.randn(n_features, 1, d_model) * 0.02)
		self.biases = nn.Parameter(torch.zeros(n_features, d_model))

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		proj = self.projections.squeeze(1).unsqueeze(0)
		return x_exp * proj + self.biases.unsqueeze(0)


class PiecewiseLinearEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_bins = 16):
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
		out = torch.einsum("bnk,nkd->bnd", bin_act, self.feature_weights)
		return out


class PeriodicEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq = 32):
		super().__init__()
		self.num_freq = num_freq
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.phi = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq, d_model)

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		sinusoidal = torch.sin(x_exp * self.omega.unsqueeze(0) + self.phi.unsqueeze(0))
		out = self.proj(sinusoidal)
		return out
	
class FourierEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq = 32):
		super().__init__()
		self.num_freq = num_freq
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq * 2, d_model)

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		scaled = x_exp * self.omega.unsqueeze(0)
		features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim = -1)
		out = self.proj(features)
		return out


def build_encoder(variant, n_features, d_model, ple_bins = 16, periodic_freq = 32):
	if variant == "linear":
		return LinearEncoder(n_features, d_model)
	elif variant == "per_feature":
		return PerFeatureTokeniser(n_features, d_model)
	elif variant == "ple":
		return PiecewiseLinearEncoder(n_features, d_model, num_bins = ple_bins)
	elif variant == "periodic":
		return PeriodicEncoder(n_features, d_model, num_freq = periodic_freq)
	elif variant == "fourier":
		return FourierEncoder(n_features, d_model, num_freq = periodic_freq)
	else:
		raise ValueError(f"Unknown encoding variant: {variant}")

### Multi-Head Sparse Attention

class SparseMultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, top_k, dropout = 0.1):
		super().__init__()
		assert d_model % n_heads == 0
		self.d_model = d_model
		self.n_heads = n_heads
		self.d_k = d_model // n_heads
		self.top_k = top_k

		self.W_q = nn.Linear(d_model, d_model)
		self.W_k = nn.Linear(d_model, d_model)
		self.W_v = nn.Linear(d_model, d_model)
		self.W_o = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		n_firms = x.shape[0]
		x = x.unsqueeze(0)

		Q = self.W_q(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		K = self.W_k(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		V = self.W_v(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)

		scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

		k = min(self.top_k, n_firms)
		topk_vals, _ = scores.topk(k, dim = -1)
		threshold = topk_vals[..., -1:].detach()
		mask = scores < threshold
		scores = scores.masked_fill(mask, float("-inf"))

		attn_weights = F.softmax(scores, dim = -1)
		attn_weights = self.dropout(attn_weights)

		context = torch.matmul(attn_weights, V)
		context = context.transpose(1, 2).contiguous().view(1, n_firms, self.d_model)
		out = self.W_o(context).squeeze(0)

		return out, attn_weights.squeeze(0)

### Transformer Encoder Block

class TransformerBlock(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, top_k, dropout = 0.1):
		super().__init__()
		self.norm1 = nn.LayerNorm(d_model)
		self.attention = SparseMultiHeadAttention(d_model, n_heads, top_k, dropout)
		self.grn = GRN(d_model, d_ff, dropout)

	def forward(self, x):
		normed = self.norm1(x)
		attn_out, attn_weights = self.attention(normed)
		x = x + attn_out
		x = self.grn(x)
		return x, attn_weights

### Portfolio Transformer

class PortfolioTransformer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		n_k0 = len(K0_CHARS)
		n_k1 = len(K1_CHARS)
		n_miss = len(miss_flags)

		self.k0_encoder = build_encoder(
			config.encoding_variant, n_k0, config.d_model,
			ple_bins = config.ple_num_bins,
			periodic_freq = config.periodic_num_freq
		)
		self.k1_encoder = build_encoder(
			config.encoding_variant, n_k1, config.d_model,
			ple_bins = config.ple_num_bins,
			periodic_freq = config.periodic_num_freq
		)

		self.time2vec = Time2Vec(config.d_model)
		self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)
		self.miss_proj = nn.Linear(n_miss, config.d_model)

		self.blocks = nn.ModuleList([
			TransformerBlock(
				config.d_model, config.n_heads, config.d_ff,
				config.top_k_attention, config.dropout
			)
			for _ in range(config.n_layers)
		])

		# Regression heads: single scalar output per horizon
		self.head_3m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.head_6m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.head_12m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))

		self.register_buffer("lag_positions", torch.tensor(LAG_POSITIONS, dtype = torch.float32))

	def _encode_firm_token(self, k0, k1, miss):
		n_firms = k0.shape[0]

		k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
		k0_token = k0_encoded.sum(dim = 1)

		k1_flat = k1.permute(0, 2, 1).reshape(n_firms * 6, -1)
		k1_encoded = self.k1_encoder(k1_flat)
		k1_encoded = k1_encoded.view(n_firms, 6, len(K1_CHARS), self.config.d_model)

		t2v_all = self.time2vec(self.lag_positions).unsqueeze(0).unsqueeze(2)
		k1_encoded = k1_encoded + t2v_all

		k1_token = k1_encoded.sum(dim = (1, 2))

		miss_token = self.miss_proj(miss)
		return k0_token + k1_token + miss_token

	def forward(self, k0, k1, miss):
		z = self._encode_firm_token(k0, k1, miss)

		all_attn = []
		for block in self.blocks:
			z, attn_w = block(z)
			all_attn.append(attn_w)

		# Each head outputs (n_firms, 1), squeeze to (n_firms,)
		return self.head_3m(z).squeeze(-1), self.head_6m(z).squeeze(-1), self.head_12m(z).squeeze(-1), all_attn

## Training Utilities

def compute_multitask_loss(scores_3m, scores_6m, scores_12m, targets, valid_masks, config):
	"""Multi-horizon Huber regression loss, masked to valid observations."""
	total_loss = torch.tensor(0.0, device = scores_3m.device)
	horizon_losses = {}

	for horizon, scores, weight in [
		("target_3m", scores_3m, config.lambda_3m),
		("target_6m", scores_6m, config.lambda_6m),
		("target_12m", scores_12m, config.lambda_12m),
	]:
		valid = valid_masks[horizon]
		if valid.sum() > 0:
			loss = F.huber_loss(scores[valid], targets[horizon][valid], delta = 1.0)
			total_loss = total_loss + weight * loss
			horizon_losses[horizon] = loss.item()

	return total_loss, horizon_losses


def compute_rank_correlation(scores, targets, valid_mask):
	"""Spearman rank correlation between predicted scores and continuous returns."""
	valid = valid_mask
	if valid.sum() < 10:
		return 0.0

	pred = scores[valid]
	true = targets[valid]

	def _rank(t):
		order = t.argsort()
		ranks = torch.zeros_like(t)
		ranks[order] = torch.arange(len(t), device = t.device, dtype = torch.float32)
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

### 6. Training and Persistence

def train_one_epoch(model, dataset, optimizer, config, scaler):
	model.train()
	epoch_loss = 0.0
	n_months = 0

	indices = np.random.permutation(len(dataset))
	for idx in indices:
		batch = dataset[idx]
		k0 = batch["k0"].to(device, non_blocking = True)
		k1 = batch["k1"].to(device, non_blocking = True)
		miss = batch["miss"].to(device, non_blocking = True)
		targets = {k: v.to(device, non_blocking = True) for k, v in batch["targets"].items()}
		valid_masks = {k: v.to(device, non_blocking = True) for k, v in batch["valid_masks"].items()}

		optimizer.zero_grad(set_to_none = True)
		with torch.autocast("cuda"):
			scores_3m, scores_6m, scores_12m, _ = model(k0, k1, miss)
			loss, _ = compute_multitask_loss(scores_3m, scores_6m, scores_12m, targets, valid_masks, config)

		if loss.requires_grad:
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
			scaler.step(optimizer)
			scaler.update()

		epoch_loss += loss.item()
		n_months += 1

	return epoch_loss / max(n_months, 1)


@torch.no_grad()
def evaluate(model, dataset, config):
	model.eval()
	total_loss = 0.0
	total_corr = {"target_3m": 0.0, "target_6m": 0.0, "target_12m": 0.0}
	n_months = 0

	for idx in range(len(dataset)):
		batch = dataset[idx]
		k0 = batch["k0"].to(device)
		k1 = batch["k1"].to(device)
		miss = batch["miss"].to(device)
		targets = {k: v.to(device) for k, v in batch["targets"].items()}
		valid_masks = {k: v.to(device) for k, v in batch["valid_masks"].items()}

		scores_3m, scores_6m, scores_12m, _ = model(k0, k1, miss)
		loss, _ = compute_multitask_loss(scores_3m, scores_6m, scores_12m, targets, valid_masks, config)
		total_loss += loss.item()

		for horizon, scores in [("target_3m", scores_3m), ("target_6m", scores_6m), ("target_12m", scores_12m)]:
			total_corr[horizon] += compute_rank_correlation(scores, targets[horizon], valid_masks[horizon])

		n_months += 1

	n = max(n_months, 1)
	return {
		"loss": total_loss / n,
		"rank_corr": {k: v / n for k, v in total_corr.items()},
	}


def train_variant(config):
	"""
	Train a single encoding variant, evaluate on test, save weights and metrics to disk,
	then free all memory.
	"""
	variant = config.encoding_variant
	print(f"Encoding variant: {variant}")
	print(f"d_model: {config.d_model}, n_heads: {config.n_heads}, n_layers: {config.n_layers}")
	print(f"Horizon weights: 3m={config.lambda_3m}, 6m={config.lambda_6m}, 12m={config.lambda_12m}")
	print()

	train_ds = load_split(config.train_path, k0_feature_cols, k1_feature_cols, miss_flags, target_cols
	)

	val_ds = load_split(config.val_path, k0_feature_cols, k1_feature_cols, miss_flags, target_cols
	)
	print(f"Train months: {len(train_ds)}, Val months: {len(val_ds)}")

	model = PortfolioTransformer(config).to(device)
	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters: {n_params:,}")
	print()

	optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate,
		weight_decay = config.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode = "min", factor = 0.5, patience = 10
	)

	best_val_loss = float("inf")
	patience_counter = 0
	history = {"train_loss": [], "val_loss": [], "val_corr_6m": []}
	weights_path = config.results_dir / f"weights_{variant}.pt"

	scaler = torch.GradScaler("cuda")

	for epoch in range(1, config.max_epochs + 1):
		train_loss = train_one_epoch(model, train_ds, optimizer, config, scaler)
		val_metrics = evaluate(model, val_ds, config)
		val_loss = val_metrics["loss"]
		scheduler.step(val_loss)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["val_corr_6m"].append(val_metrics["rank_corr"]["target_6m"])

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch:3d} | "
			f"Train Loss:{train_loss:.6f} | "
			f"Val Loss:{val_loss:.6f} | "
			f"Val Corr 6m:{val_metrics['rank_corr']['target_6m']:.4f} | "
			f"LR:{current_lr:.2e}"
		)
		sys.stdout.flush()

		if val_loss < best_val_loss - 1e-5:
			best_val_loss = val_loss
			patience_counter = 0
			torch.save(model.state_dict(), weights_path)
		else:
			patience_counter += 1
			if patience_counter >= config.patience:
				print(f"Early stopping at epoch {epoch}")
				break

	del train_ds, val_ds
	gc.collect()

	model.load_state_dict(torch.load(weights_path, weights_only = True))
	test_ds = load_split(
		config.test_path, k0_feature_cols, k1_feature_cols, miss_flags, target_cols
	)
	test_metrics = evaluate(model, test_ds, config)
	del test_ds

	print(f"Test Loss: {test_metrics['loss']:.6f}")
	for h in ["target_3m", "target_6m", "target_12m"]:
		print(f"{h} | Corr:{test_metrics['rank_corr'][h]:.4f}")

	results_path = config.results_dir / f"metrics_{variant}.json"
	results_payload = {
		"variant": variant,
		"n_params": n_params,
		"best_val_loss": best_val_loss,
		"stopped_epoch": len(history["train_loss"]),
		"history": history,
		"test_metrics": test_metrics,
	}
	with open(results_path, "w") as f:
		json.dump(results_payload, f, indent = 2)

	print(f"Weights saved to: {weights_path}")
	print(f"Metrics saved to: {results_path}")

	del model
	gc.collect()
	torch.cuda.empty_cache()

## Train Each Variant 

#### Linear Projection

cfg.encoding_variant = "linear"
train_variant(cfg)

### Piecewise Linear Encoding

cfg.encoding_variant = "ple"
train_variant(cfg)

### Per-Feature Tokenisation

cfg.encoding_variant = "per_feature"
train_variant(cfg)

### Periodic Encoding

cfg.encoding_variant = "periodic"
train_variant(cfg)

### Fourier Encoding

cfg.encoding_variant = "fourier"
train_variant(cfg)

### Load Results and Compare Variants

def load_all_results(results_dir):
	variants = ["linear", "per_feature", "ple", "periodic", "fourier"]
	all_results = {}

	for variant in variants:
		metrics_path = results_dir / f"metrics_{variant}.json"
		if metrics_path.exists():
			with open(metrics_path, "r") as f:
				all_results[variant] = json.load(f)
			print(f"Loaded: {variant} (stopped at epoch {all_results[variant]['stopped_epoch']})")
		else:
			print(f"Missing: {variant} (not yet trained)")

	return all_results


all_results = load_all_results(cfg.results_dir)

if all_results:
	print()
	print(f"{'Variant':<20s} {'Params':>10s} {'Epoch':>6s} {'Test Loss':>10s} {'Corr 3m':>8s} {'Corr 6m':>8s} {'Corr 12m':>9s}")
	print("=" * 75)
	for v, res in all_results.items():
		tm = res["test_metrics"]
		print(
			f"{v:<20s} {res['n_params']:>10,} {res['stopped_epoch']:>6d}"
			f" {tm['loss']:>10.5f}"
			f" {tm['rank_corr']['target_3m']:>8.5f}"
			f" {tm['rank_corr']['target_6m']:>8.5f}"
			f" {tm['rank_corr']['target_12m']:>9.5f}"
		)
	print()
	best_variant = max(all_results, key = lambda v: all_results[v]["test_metrics"]["rank_corr"]["target_6m"])
	print(f"Best variant by 6m rank correlation: {best_variant}")

### Results Summary

def print_results_table(all_results):
	print(f"{'Variant':<18} {'Params':>10} {'Epoch':>6} {'Test Loss':>10} {'Corr 3m':>8} {'Corr 6m':>8} {'Corr 12m':>9}")
	print("=" * 88)

	for variant, res in all_results.items():
		tm = res["test_metrics"]
		print(
			f"{variant:<18} {res['n_params']:>10,} "
			f"{res['stopped_epoch']:>6} {tm['loss']:>10.5f} "
			f"{tm['rank_corr']['target_3m']:>8.5f} {tm['rank_corr']['target_6m']:>8.5f} "
			f"{tm['rank_corr']['target_12m']:>9.5f}"
		)

	# Identify best variant
	best = max(all_results, key = lambda v: all_results[v]["test_metrics"]["rank_corr"]["target_6m"])
	print(f"\nBest variant by 6m rank correlation: {best}")


print_results_table(all_results)

### Training Summary

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11


def plot_training_curves(all_results):
	n_variants = len(all_results)
	fig, axes = plt.subplots(n_variants, 2, figsize = (13, 4.5 * n_variants))
	if n_variants == 1:
		axes = axes.reshape(1, -1)

	for row, (variant, res) in enumerate(all_results.items()):
		history = res["history"]

		axes[row, 0].plot(history["train_loss"], label = "Train", linewidth = 1.5)
		axes[row, 0].plot(history["val_loss"], label = "Validation", linewidth = 1.5)
		axes[row, 0].set_xlabel("Epoch")
		axes[row, 0].set_ylabel("Loss")
		axes[row, 0].set_title(f"Multi-task Loss: {variant}")
		axes[row, 0].legend()
		axes[row, 0].grid(alpha = 0.3)

		axes[row, 1].plot(history["val_corr_6m"], linewidth = 1.5, color = "tab:green")
		axes[row, 1].set_xlabel("Epoch")
		axes[row, 1].set_ylabel("Spearman Correlation")
		axes[row, 1].set_title(f"Val Rank Correlation (6m): {variant}")
		axes[row, 1].grid(alpha = 0.3)

	plt.tight_layout()
	plt.show()


def plot_variant_comparison(all_results):
	variants = list(all_results.keys())
	corr_6m = [all_results[v]["test_metrics"]["rank_corr"]["target_6m"] for v in variants]
	losses = [all_results[v]["test_metrics"]["loss"] for v in variants]

	fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
	labels = [v.replace("_", " ").title() for v in variants]

	axes[0].bar(labels, corr_6m)
	axes[0].set_ylabel("Rank Correlation")
	axes[0].set_title("6-Month Rank Correlation (Test)")
	axes[0].grid(axis = "y", alpha = 0.3)

	axes[1].bar(labels, losses)
	axes[1].set_ylabel("Loss")
	axes[1].set_title("Multi-task Loss (Test)")
	axes[1].grid(axis = "y", alpha = 0.3)

	plt.tight_layout()
	plt.show()


plot_training_curves(all_results)
plot_variant_comparison(all_results)

### Portfolio Simulation

@torch.no_grad()
def portfolio_simulation(model, dataset, config, rebalance_freq = 6, transaction_cost_bps = 25):
	"""Long-only portfolio: top quintile by 6-month predicted score, equal weighted."""
	model.eval()
	portfolio_returns = []
	prev_holdings = set()

	for idx in range(0, len(dataset), rebalance_freq):
		batch = dataset[idx]
		k0 = batch["k0"].to(device, non_blocking = True)
		k1 = batch["k1"].to(device, non_blocking = True)
		miss = batch["miss"].to(device, non_blocking = True)

		_, scores_6m, _, _ = model(k0, k1, miss)

		n_firms = scores_6m.shape[0]
		n_quintile = max(int(0.2 * n_firms), 1)

		_, top_indices = scores_6m.topk(n_quintile)
		top_set = set(top_indices.cpu().numpy().tolist())

		if len(top_set) == 0:
			portfolio_returns.append(0.0)
			prev_holdings = top_set
			continue

		new_holdings = top_set - prev_holdings
		exited_holdings = prev_holdings - top_set
		turnover = (len(new_holdings) + len(exited_holdings)) / max(len(top_set), 1)
		tc = turnover * transaction_cost_bps / 10000.0

		raw_returns = batch["targets"]["target_6m"]
		valid = batch["valid_masks"]["target_6m"]

		# Equal-weighted return across valid firms in top quintile
		valid_returns = []
		for firm_idx in top_indices.cpu().numpy():
			if valid[firm_idx]:
				valid_returns.append(raw_returns[firm_idx].item())

		if len(valid_returns) > 0:
			mean_return = sum(valid_returns) / len(valid_returns)
		else:
			mean_return = 0.0

		portfolio_returns.append(mean_return - tc)
		prev_holdings = top_set

	return np.array(portfolio_returns)


@torch.no_grad()
def portfolio_simulation_long_short(model, dataset, config, rebalance_freq = 6, transaction_cost_bps = 25):
	"""Long-short portfolio: long top quintile, short bottom quintile, score-proportional weights."""
	model.eval()
	portfolio_returns = []
	prev_long = set()
	prev_short = set()

	for idx in range(0, len(dataset), rebalance_freq):
		batch = dataset[idx]
		k0 = batch["k0"].to(device, non_blocking = True)
		k1 = batch["k1"].to(device, non_blocking = True)
		miss = batch["miss"].to(device, non_blocking = True)

		_, scores_6m, _, _ = model(k0, k1, miss)

		n_firms = scores_6m.shape[0]
		n_quintile = max(int(0.2 * n_firms), 1)

		_, long_indices = scores_6m.topk(n_quintile)
		_, short_indices = scores_6m.topk(n_quintile, largest = False)

		long_set = set(long_indices.cpu().numpy().tolist())
		short_set = set(short_indices.cpu().numpy().tolist())

		long_turnover = len(long_set - prev_long) + len(prev_long - long_set)
		short_turnover = len(short_set - prev_short) + len(prev_short - short_set)
		total_turnover = (long_turnover + short_turnover) / max(n_quintile, 1)
		tc = total_turnover * transaction_cost_bps / 10000.0

		raw_returns = batch["targets"]["target_6m"]
		valid = batch["valid_masks"]["target_6m"]

		# Score-proportional long leg
		long_scores = scores_6m[long_indices]
		long_weights = F.softmax(long_scores, dim = 0)
		long_return = 0.0
		for i, firm_idx in enumerate(long_indices.cpu().numpy()):
			if valid[firm_idx]:
				long_return += long_weights[i].item() * raw_returns[firm_idx].item()

		# Score-proportional short leg (inverted for weighting)
		short_scores = -scores_6m[short_indices]
		short_weights = F.softmax(short_scores, dim = 0)
		short_return = 0.0
		for i, firm_idx in enumerate(short_indices.cpu().numpy()):
			if valid[firm_idx]:
				short_return += short_weights[i].item() * raw_returns[firm_idx].item()

		ls_return = long_return - short_return - tc
		portfolio_returns.append(ls_return)

		prev_long = long_set
		prev_short = short_set

	return np.array(portfolio_returns)


def compute_portfolio_metrics(returns, periods_per_year = 2):
	cum_return = (1 + returns).prod() - 1
	annualised_return = (1 + cum_return) ** (periods_per_year / max(len(returns), 1)) - 1
	annualised_vol = returns.std() * np.sqrt(periods_per_year)
	sharpe = annualised_return / max(annualised_vol, 1e-8)

	cum_wealth = np.cumprod(1 + returns)
	peak = np.maximum.accumulate(cum_wealth)
	drawdown = (peak - cum_wealth) / peak
	max_dd = drawdown.max()

	return {
		"cumulative_return": cum_return,
		"annualised_return": annualised_return,
		"annualised_vol": annualised_vol,
		"sharpe_ratio": sharpe,
		"max_drawdown": max_dd,
		"n_rebalances": len(returns),
	}


best_variant = max(all_results, key = lambda v: all_results[v]["test_metrics"]["rank_corr"]["target_6m"])
print(f"Best variant: {best_variant}")
print()

cfg.encoding_variant = best_variant
best_model = PortfolioTransformer(cfg).to(device)
best_model.load_state_dict(torch.load(cfg.results_dir / f"weights_{best_variant}.pt", weights_only = True))

test_ds = load_split(
	cfg.test_path, k0_feature_cols, k1_feature_cols, miss_flags, target_cols
)

lo_returns = portfolio_simulation(best_model, test_ds, cfg)
ls_returns = portfolio_simulation_long_short(best_model, test_ds, cfg)

print("Long-Only Portfolio:")
for k, v in compute_portfolio_metrics(lo_returns).items():
	print(f"{k}: {v:.4f}")

print()
print("Long-Short Portfolio:")
for k, v in compute_portfolio_metrics(ls_returns).items():
	print(f"{k}: {v:.4f}")

del best_model, test_ds
gc.collect()
torch.cuda.empty_cache()

### Model Dimension Check

test_cfg = Config()

print("Parameter counts by variant")
for variant in ["linear", "per_feature", "ple", "periodic", "fourier"]:
	test_cfg.encoding_variant = variant
	m = PortfolioTransformer(test_cfg).to(device)
	n_test = 100
	k0_t = torch.randn(n_test, len(K0_CHARS), device = device)
	k1_t = torch.randn(n_test, len(K1_CHARS), 6, device = device)
	miss_t = torch.zeros(n_test, len(miss_flags), device = device)
	s3, s6, s12, attn = m(k0_t, k1_t, miss_t)
	n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
	print(f"{variant:15s} | params:{n_p:>10,} | score shape:{s6.shape} | attn:{attn[0].shape}")

	del m

gc.collect()
torch.cuda.empty_cache()
print()
print("All encoding variants produce correct output dimensions.")


