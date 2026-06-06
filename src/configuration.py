from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
	time2vec_dim: int = 16
	ple_num_bins: int = 16
	periodic_num_freq: int = 32

	learning_rate: float = 1e-4
	weight_decay: float = 1e-5
	max_epochs: int = 50
	patience: int = 7
	grad_clip: float = 1.0

	lambda_3m: float = 0.2
	lambda_6m: float = 0.5
	lambda_12m: float = 0.3

	n_classes: int = 5
	encoding_variant: str = "linear"
	max_firms: int = 5000
	seed: int = 24

cfg = Config()
cfg.results_dir.mkdir(parents = True, exist_ok = True)


## Column Classfication

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

## Dataset and Data Loading

class CrossSectionalDataset(Dataset):
	"""
	Dataset that groups firm observations by month for cross-sectional attention
	Each item returned is an entire cross-section (all firms in one month)
	"""

	def __init__(self, df, k0_cols, k1_cols, miss_cols, target_cols, n_classes, max_firms):
		self.n_classes = n_classes
		self.max_firms = max_firms
		self.target_col_names = target_cols

		dates = sorted(df["eom"].unique())
		self.monthly_data = []

		n_k1 = len(K1_CHARS)
		for date in dates:
			group = df[df["eom"] == date]
			if len(group) > max_firms:
				group = group.sample(n = max_firms, random_state = 42)

			k0 = torch.tensor(group[k0_cols].values, dtype = torch.float32)
			k1_raw = group[k1_cols].values.astype(np.float32)
			k1 = torch.tensor(k1_raw.reshape(len(group), n_k1, 6), dtype = torch.float32)
			miss = torch.tensor(group[miss_cols].values, dtype = torch.float32)

			# Discretised quintile labels for training
			targets = {}
			# Raw continuous returns for portfolio simulation
			raw_targets = {}
			for tc in target_cols:
				vals = group[tc].values.copy()
				valid_mask = ~np.isnan(vals)

				labels = np.full(len(vals), -1, dtype = np.int64)
				if valid_mask.sum() > n_classes:
					breaks = np.quantile(
						vals[valid_mask],
						np.linspace(0, 1, n_classes + 1)[1:-1]
					)
					labels[valid_mask] = np.digitize(vals[valid_mask], breaks)
				targets[tc] = torch.tensor(labels, dtype = torch.long)

				raw_vals = np.copy(vals)
				raw_vals[~valid_mask] = 0.0
				raw_targets[tc] = torch.tensor(raw_vals, dtype = torch.float32)

			self.monthly_data.append({
				"k0": k0,
				"k1": k1,
				"miss": miss,
				"targets": targets,
				"raw_targets": raw_targets,
				"n_firms": len(group),
			})

		del df
		gc.collect()

	def __len__(self):
		return len(self.monthly_data)

	def __getitem__(self, idx):
		return self.monthly_data[idx]


def load_split(path, k0_cols, k1_cols, miss_cols, target_cols, n_classes, max_firms):
	required = k0_cols + k1_cols + miss_cols + target_cols + ["eom"]
	df = pd.read_parquet(path, columns = required)

	for col in k0_cols + k1_cols + miss_cols:
		df[col] = df[col].fillna(0.0)

	return CrossSectionalDataset(df, k0_cols, k1_cols, miss_cols, target_cols, n_classes, max_firms)