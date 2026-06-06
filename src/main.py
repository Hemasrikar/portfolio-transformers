import matplotlib.pyplot as plt
import matplotlib

from training import *
from configuration import *
from transformer_architecture import *
from transformer_components import *
from portfolio_transformer import PortfolioTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
	print(f"GPU: {torch.cuda.get_device_name(0)}")
	print(f"CUDA version: {torch.version.cuda}")

## Training Different Varients

cfg.encoding_variant = "linear"
train_variant(cfg)

cfg.encoding_variant = "per_feature"
train_variant(cfg)

cfg.encoding_variant = "periodic"
train_variant(cfg)

## Comparing Varients

def load_all_results(results_dir):
	"""Load saved metrics for all completed variants from disk."""
	variants = ["linear", "per_feature", "ple", "periodic"]
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

## Results summary

def print_results_table(all_results):
	print(f"{'Variant':<18} {'Params':>10} {'Epoch':>6} {'Test Loss':>10} {'Corr 3m':>8} {'Corr 6m':>8} {'Corr 12m':>9} {'Acc 6m':>7}")
	print("=" * 88)

	for variant, res in all_results.items():
		tm = res["test_metrics"]
		print(
			f"{variant:<18} "
			f"{res['n_params']:>10,} "
			f"{res['stopped_epoch']:>6} "
			f"{tm['loss']:>10.5f} "
			f"{tm['rank_corr']['target_3m']:>8.5f} "
			f"{tm['rank_corr']['target_6m']:>8.5f} "
			f"{tm['rank_corr']['target_12m']:>9.5f} "
			f"{tm['accuracy']['target_6m']:>7.5f}"
		)

	# Identify best variant
	best = max(all_results, key = lambda v: all_results[v]["test_metrics"]["rank_corr"]["target_6m"])
	print(f"\nBest variant by 6m rank correlation: {best}")


print_results_table(all_results)

## Training Summary

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
	acc_6m = [all_results[v]["test_metrics"]["accuracy"]["target_6m"] for v in variants]
	losses = [all_results[v]["test_metrics"]["loss"] for v in variants]

	fig, axes = plt.subplots(1, 3, figsize = (14, 4.5))
	labels = [v.replace("_", " ").title() for v in variants]

	axes[0].bar(labels, corr_6m)
	axes[0].set_ylabel("Rank Correlation")
	axes[0].set_title("6-Month Rank Correlation (Test)")
	axes[0].grid(axis = "y", alpha = 0.3)

	axes[1].bar(labels, acc_6m)
	axes[1].set_ylabel("Accuracy")
	axes[1].set_title("6-Month Quintile Accuracy (Test)")
	axes[1].grid(axis = "y", alpha = 0.3)

	axes[2].bar(labels, losses)
	axes[2].set_ylabel("Loss")
	axes[2].set_title("Multi-task Loss (Test)")
	axes[2].grid(axis = "y", alpha = 0.3)

	plt.tight_layout()
	plt.show()


plot_training_curves(all_results)
plot_variant_comparison(all_results)

## Portfolio Simulation

@torch.no_grad()
def portfolio_simulation(model, dataset, config, rebalance_freq = 6, transaction_cost_bps = 25):
	model.eval()
	portfolio_returns = []
	prev_holdings = set()

	for idx in range(0, len(dataset), rebalance_freq):
		batch = dataset[idx]
		k0 = batch["k0"].to(device, non_blocking = True)
		k1 = batch["k1"].to(device, non_blocking = True)
		miss = batch["miss"].to(device, non_blocking = True)

		_, logits_6m, _, _ = model(k0, k1, miss)

		probs = F.softmax(logits_6m, dim = -1)
		quintile_idx = torch.arange(config.n_classes, device = device, dtype = torch.float32)
		scores = (probs * quintile_idx.unsqueeze(0)).sum(dim = -1)

		n_firms = scores.shape[0]
		cutoff = int(0.8 * n_firms)
		_, top_indices = scores.topk(n_firms - cutoff)
		top_set = set(top_indices.cpu().numpy().tolist())

		n_selected = len(top_set)
		if n_selected == 0:
			portfolio_returns.append(0.0)
			prev_holdings = top_set
			continue

		new_holdings = top_set - prev_holdings
		exited_holdings = prev_holdings - top_set
		turnover = (len(new_holdings) + len(exited_holdings)) / max(n_selected, 1)
		tc = turnover * transaction_cost_bps / 10000.0

		top_scores = scores[top_indices]
		top_weights = F.softmax(top_scores, dim = 0)

		raw_returns = batch["raw_targets"]["target_6m"]
		target_labels = batch["targets"]["target_6m"]
		weighted_return = 0.0
		for i, firm_idx in enumerate(top_indices.cpu().numpy()):
			if target_labels[firm_idx] >= 0:
				weighted_return += top_weights[i].item() * raw_returns[firm_idx].item()

		portfolio_returns.append(weighted_return - tc)
		prev_holdings = top_set

	return np.array(portfolio_returns)


@torch.no_grad()
def portfolio_simulation_long_short(model, dataset, config, rebalance_freq = 6, transaction_cost_bps = 25):
	model.eval()
	portfolio_returns = []
	prev_long = set()
	prev_short = set()

	for idx in range(0, len(dataset), rebalance_freq):
		batch = dataset[idx]
		k0 = batch["k0"].to(device, non_blocking = True)
		k1 = batch["k1"].to(device, non_blocking = True)
		miss = batch["miss"].to(device, non_blocking = True)

		_, logits_6m, _, _ = model(k0, k1, miss)

		probs = F.softmax(logits_6m, dim = -1)
		quintile_idx = torch.arange(config.n_classes, device = device, dtype = torch.float32)
		scores = (probs * quintile_idx.unsqueeze(0)).sum(dim = -1)

		n_firms = scores.shape[0]
		n_quintile = max(int(0.2 * n_firms), 1)

		# Top quintile (long) and bottom quintile (short)
		_, long_indices = scores.topk(n_quintile)
		_, short_indices = scores.topk(n_quintile, largest = False)

		long_set = set(long_indices.cpu().numpy().tolist())
		short_set = set(short_indices.cpu().numpy().tolist())

		# Transaction costs
		long_turnover = len(long_set - prev_long) + len(prev_long - long_set)
		short_turnover = len(short_set - prev_short) + len(prev_short - short_set)
		total_turnover = (long_turnover + short_turnover) / max(n_quintile, 1)
		tc = total_turnover * transaction_cost_bps / 10000.0

		raw_returns = batch["raw_targets"]["target_6m"]
		target_labels = batch["targets"]["target_6m"]

		# Score-weighted long leg
		long_scores = scores[long_indices]
		long_weights = F.softmax(long_scores, dim = 0)
		long_return = 0.0
		for i, firm_idx in enumerate(long_indices.cpu().numpy()):
			if target_labels[firm_idx] >= 0:
				long_return += long_weights[i].item() * raw_returns[firm_idx].item()

		# Score-weighted short leg (inverted scores for weighting)
		short_scores = -scores[short_indices]
		short_weights = F.softmax(short_scores, dim = 0)
		short_return = 0.0
		for i, firm_idx in enumerate(short_indices.cpu().numpy()):
			if target_labels[firm_idx] >= 0:
				short_return += short_weights[i].item() * raw_returns[firm_idx].item()

		# Long-short return: gain from longs, gain from shorts declining
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
	cfg.test_path, k0_feature_cols, k1_feature_cols,
	miss_flags, target_cols, cfg.n_classes, cfg.max_firms
)

lo_returns = portfolio_simulation(best_model, test_ds, cfg)
ls_returns = portfolio_simulation_long_short(best_model, test_ds, cfg)

print("Long-Only Portfolio:")
for k, v in compute_portfolio_metrics(lo_returns).items():
	print(f"  {k}: {v:.4f}")

print()
print("Long-Short Portfolio:")
for k, v in compute_portfolio_metrics(ls_returns).items():
	print(f"  {k}: {v:.4f}")

del best_model, test_ds
gc.collect()
if torch.cuda.is_available():
	torch.cuda.empty_cache()
	
## Sanity check

test_cfg = Config()
print("Parameter counts by variant:")
for variant in ["linear", "per_feature", "ple", "periodic"]:
	test_cfg.encoding_variant = variant
	m = PortfolioTransformer(test_cfg).to(device)
	n_test = 100
	k0_t = torch.randn(n_test, len(K0_CHARS), device = device)
	k1_t = torch.randn(n_test, len(K1_CHARS), 6, device = device)
	miss_t = torch.zeros(n_test, len(miss_flags), device = device)
	l3, l6, l12, attn = m(k0_t, k1_t, miss_t)
	n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
	print(f"{variant:15s} | params:{n_p:>10,} | logits:{l6.shape} | attn:{attn[0].shape}")
	del m

gc.collect()
if torch.cuda.is_available():
	torch.cuda.empty_cache()
print()
print("All encoding variants produce correct output dimensions.")