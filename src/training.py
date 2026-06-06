import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from portfolio_transformer import PortfolioTransformer
from configuration import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
	print(f"GPU: {torch.cuda.get_device_name(0)}")
	print(f"CUDA version: {torch.version.cuda}")

## Training Utilities

def compute_multitask_loss(logits_3m, logits_6m, logits_12m, targets, config):
	total_loss = torch.tensor(0.0, device = logits_3m.device)
	horizon_losses = {}

	for horizon, logits, weight in [
		("target_3m", logits_3m, config.lambda_3m),
		("target_6m", logits_6m, config.lambda_6m),
		("target_12m", logits_12m, config.lambda_12m),
	]:
		labels = targets[horizon]
		valid = labels >= 0
		if valid.sum() > 0:
			loss = F.cross_entropy(logits[valid], labels[valid])
			total_loss = total_loss + weight * loss
			horizon_losses[horizon] = loss.item()

	return total_loss, horizon_losses


def compute_accuracy(logits, labels):
	valid = labels >= 0
	if valid.sum() == 0:
		return 0.0
	preds = logits[valid].argmax(dim = -1)
	return (preds == labels[valid]).float().mean().item()


def compute_rank_correlation(logits, labels):
	valid = labels >= 0
	if valid.sum() < 10:
		return 0.0
	probs = F.softmax(logits[valid], dim = -1)
	quintile_idx = torch.arange(probs.shape[1], device = probs.device, dtype = torch.float32)
	expected_score = (probs * quintile_idx.unsqueeze(0)).sum(dim = -1)

	def _rank(t):
		order = t.argsort()
		ranks = torch.zeros_like(t)
		ranks[order] = torch.arange(len(t), device = t.device, dtype = torch.float32)
		return ranks

	rank_pred = _rank(expected_score)
	rank_true = _rank(labels[valid].float())
	mean_p = rank_pred.mean()
	mean_t = rank_true.mean()
	cov = ((rank_pred - mean_p) * (rank_true - mean_t)).sum()
	std_p = ((rank_pred - mean_p) ** 2).sum().sqrt()
	std_t = ((rank_true - mean_t) ** 2).sum().sqrt()
	if std_p * std_t < 1e-8:
		return 0.0
	return (cov / (std_p * std_t)).item()

## Training and Persistence

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

		optimizer.zero_grad(set_to_none = True)
		with torch.amp.autocast("cuda"):
			logits_3m, logits_6m, logits_12m, _ = model(k0, k1, miss)
			loss, _ = compute_multitask_loss(logits_3m, logits_6m, logits_12m, targets, config)

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
	total_acc = {"target_3m": 0.0, "target_6m": 0.0, "target_12m": 0.0}
	n_months = 0

	for idx in range(len(dataset)):
		batch = dataset[idx]
		k0 = batch["k0"].to(device)
		k1 = batch["k1"].to(device)
		miss = batch["miss"].to(device)
		targets = {k: v.to(device) for k, v in batch["targets"].items()}

		logits_3m, logits_6m, logits_12m, _ = model(k0, k1, miss)
		loss, _ = compute_multitask_loss(logits_3m, logits_6m, logits_12m, targets, config)
		total_loss += loss.item()

		for horizon, logits in [("target_3m", logits_3m), ("target_6m", logits_6m), ("target_12m", logits_12m)]:
			labels = targets[horizon]
			total_corr[horizon] += compute_rank_correlation(logits, labels)
			total_acc[horizon] += compute_accuracy(logits, labels)

		n_months += 1

	n = max(n_months, 1)
	return {
		"loss": total_loss / n,
		"rank_corr": {k: v / n for k, v in total_corr.items()},
		"accuracy": {k: v / n for k, v in total_acc.items()},
	}


def train_variant(config):
	"""
	Train a single encoding variant, evaluate on test, save weights and metrics to disk,
	then free all memory
	"""
	variant = config.encoding_variant
	print(f"Encoding variant: {variant}")
	print(f"d_model: {config.d_model}, n_heads: {config.n_heads}, n_layers: {config.n_layers}")
	print(f"Horizon weights: 3m={config.lambda_3m}, 6m={config.lambda_6m}, 12m={config.lambda_12m}")
	print()

	# Load training and validation data
	train_ds = load_split(config.train_path, k0_feature_cols, k1_feature_cols,
		miss_flags, target_cols, config.n_classes, config.max_firms
	)

	val_ds = load_split(config.val_path, k0_feature_cols, k1_feature_cols,
		miss_flags, target_cols, config.n_classes, config.max_firms
	)
	print(f"Train months: {len(train_ds)}, Val months: {len(val_ds)}")

	# Initialise model
	model = PortfolioTransformer(config).to(device)
	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters: {n_params:,}")
	print()

	optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate,
		weight_decay = config.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode = "min", factor = 0.5, patience = 3
	)

	best_val_loss = float("inf")
	patience_counter = 0
	history = {"train_loss": [], "val_loss": [], "val_corr_6m": []}
	weights_path = config.results_dir / f"weights_{variant}.pt"

	scaler = torch.amp.GradScaler("cuda")
	
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
			f"Train Loss:{train_loss:.4f} | "
			f"Val Loss:{val_loss:.4f} | "
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

	# Free training data
	del train_ds, val_ds
	gc.collect()

	# Load best weights and evaluate on test set
	model.load_state_dict(torch.load(weights_path, weights_only = True))
	test_ds = load_split(
		config.test_path, k0_feature_cols, k1_feature_cols,
		miss_flags, target_cols, config.n_classes, config.max_firms
	)
	test_metrics = evaluate(model, test_ds, config)
	del test_ds

	print(f"\nTest Loss: {test_metrics['loss']:.4f}")
	for h in ["target_3m", "target_6m", "target_12m"]:
		print(f"{h} | Corr:{test_metrics['rank_corr'][h]:.4f} | Acc:{test_metrics['accuracy'][h]:.4f}")

	# Save history and test metrics to JSON
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

	print(f"\nWeights saved to: {weights_path}")
	print(f"Metrics saved to: {results_path}")

	# Free everything
	del model
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()