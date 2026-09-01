import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

seed_metrics_pattern = re.compile(r"metrics_(?P<variant>.+)_seed(?P<seed>\d+)\.json$")


def discover_seed_metrics(results_dir):
	paths = sorted(glob.glob(str(results_dir / "metrics_*_seed*.json")))
	records = []
	for p in paths:
		m = seed_metrics_pattern.search(Path(p).name)
		if m is None:
			continue
		with open(p, "r") as f:
			data = json.load(f)
		records.append({"path": p, "variant": m.group("variant"), "seed": int(m.group("seed")), "data": data})
	if not records:
		resolved = results_dir.resolve()
		print(f"no metrics_<variant>_seed<n>.json files found under, {resolved}")
	return records


def build_learning_curve_table(records):
	rows = []
	for rec in records:
		variant = rec["variant"]
		seed = rec["seed"]
		history = rec["data"].get("history", {})
		n_epochs = len(history.get("train_loss", []))
		best_epoch = rec["data"].get("best_epoch")
		stopped_epoch = rec["data"].get("stopped_epoch")
		for e in range(n_epochs):
			rows.append({
				"variant": variant,
				"seed": seed,
				"epoch": e + 1,
				"train_loss": history["train_loss"][e],
				"train_main": history["train_main"][e],
				"train_aux": history["train_aux"][e],
				"val_loss": history["val_loss"][e],
				"val_corr_6m": history["val_corr_6m"][e],
				"val_base_corr_6m": history["val_base_corr_6m"][e],
				"val_path2_corr_6m": history["val_path2_corr_6m"][e],
				"best_epoch": best_epoch,
				"stopped_epoch": stopped_epoch,
			})
	df = pd.DataFrame(rows)
	if not df.empty:
		df["gap_loss"] = df["val_loss"] - df["train_loss"]
	return df


def build_val_test_table(records):
	rows = []
	for rec in records:
		data = rec["data"]
		val_metrics = data.get("val_metrics") or {}
		test_metrics = data.get("test_metrics") or {}
		if not val_metrics or not test_metrics:
			continue
		rows.append({
			"variant": rec["variant"],
			"seed": rec["seed"],
			"best_epoch": data.get("best_epoch"),
			"stopped_epoch": data.get("stopped_epoch"),
			"val_corr_6m": val_metrics.get("rank_corr_6m"),
			"val_base_corr_6m": val_metrics.get("rank_corr_base_6m"),
			"val_path2_corr_6m": val_metrics.get("rank_corr_path2_6m"),
			"val_loss": val_metrics.get("loss"),
			"test_corr_6m": test_metrics.get("rank_corr_6m"),
			"test_base_corr_6m": test_metrics.get("rank_corr_base_6m"),
			"test_path2_corr_6m": test_metrics.get("rank_corr_path2_6m"),
			"test_loss": test_metrics.get("loss"),
		})
	df = pd.DataFrame(rows)
	if not df.empty:
		df["gap_corr_6m"] = df["test_corr_6m"] - df["val_corr_6m"]
		df["gap_base_corr_6m"] = df["test_base_corr_6m"] - df["val_base_corr_6m"]
		df["gap_path2_corr_6m"] = df["test_path2_corr_6m"] - df["val_path2_corr_6m"]
		df["gap_loss"] = df["test_loss"] - df["val_loss"]
	return df


def build_combined_vs_base_gap_table(val_test_df):
	# reproduces, at the pre ensemble, individual seed level, the same comparison
	# stage2_path2_orthogonality_analysis.md reports at the ensemble level in
	# section 4.7, combined score minus base score, on validation and on test
	if val_test_df.empty:
		return pd.DataFrame(columns=["variant", "seed", "val_combined_minus_base", "test_combined_minus_base"])
	df = val_test_df.copy()
	df["val_combined_minus_base"] = df["val_corr_6m"] - df["val_base_corr_6m"]
	df["test_combined_minus_base"] = df["test_corr_6m"] - df["test_base_corr_6m"]
	cols = ["variant", "seed", "val_combined_minus_base", "test_combined_minus_base"]
	return df[cols]


def paired_gap_test(val_values, test_values):
	val_arr = np.asarray(val_values, dtype=float)
	test_arr = np.asarray(test_values, dtype=float)
	n = len(val_arr)
	if n < 3:
		return {"n": n, "mean_gap": float(np.mean(test_arr - val_arr)) if n > 0 else float("nan"), "t_stat": float("nan"), "p_value": float("nan"), "wilcoxon_p": float("nan")}
	t_stat, p_value = stats.ttest_rel(test_arr, val_arr)
	try:
		_, wilcoxon_p = stats.wilcoxon(test_arr, val_arr)
	except ValueError:
		wilcoxon_p = float("nan")
	return {"n": n, "mean_gap": float(np.mean(test_arr - val_arr)), "t_stat": float(t_stat), "p_value": float(p_value), "wilcoxon_p": float(wilcoxon_p)}


def plot_learning_curves(curve_df, out_dir):
	# the naive approach, averaging train and val loss across all seeds at every
	# epoch, is misleading once seeds start dropping out at their own patience
	# triggered stop, since the mean at a late epoch can end up computed over a
	# single remaining seed and reads as a trend when it is really n equal to 1
	# noise, so the bold mean line here is drawn only over the epoch range where
	# every seed in that variant is still contributing, marked full_coverage_epoch,
	# and past that point only the thin per seed lines continue, with a shaded
	# band and a label stating how many seeds remain
	if curve_df.empty:
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	for variant, group in curve_df.groupby("variant"):
		n_seeds_total = group["seed"].nunique()
		full_coverage_epoch = group.groupby("seed")["epoch"].max().min()

		fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
		ax_loss, ax_corr = axes
		for seed, seed_group in group.groupby("seed"):
			seed_group = seed_group.sort_values("epoch")
			ax_loss.plot(seed_group["epoch"], seed_group["train_loss"], color="tab:blue", alpha=0.30, linewidth=1)
			ax_loss.plot(seed_group["epoch"], seed_group["val_loss"], color="tab:orange", alpha=0.30, linewidth=1)
			ax_corr.plot(seed_group["epoch"], seed_group["val_corr_6m"], color="tab:green", alpha=0.30, linewidth=1)
			seed_best = seed_group.loc[seed_group["epoch"] == seed_group["best_epoch"].iloc[0]]
			if not seed_best.empty:
				ax_loss.scatter(seed_best["epoch"], seed_best["val_loss"], color="black", s=18, zorder=5)
				ax_corr.scatter(seed_best["epoch"], seed_best["val_corr_6m"], color="black", s=18, zorder=5)

		full_group = group[group["epoch"] <= full_coverage_epoch]
		mean_by_epoch = full_group.groupby("epoch")[["train_loss", "val_loss", "val_corr_6m"]].mean()
		ax_loss.plot(mean_by_epoch.index, mean_by_epoch["train_loss"], color="tab:blue", linewidth=2.2, label=f"train loss, mean of {n_seeds_total} seeds")
		ax_loss.plot(mean_by_epoch.index, mean_by_epoch["val_loss"], color="tab:orange", linewidth=2.2, label=f"val loss, mean of {n_seeds_total} seeds")
		ax_corr.plot(mean_by_epoch.index, mean_by_epoch["val_corr_6m"], color="tab:green", linewidth=2.2, label=f"val rank corr, mean of {n_seeds_total} seeds")

		max_epoch = group["epoch"].max()
		if max_epoch > full_coverage_epoch:
			for ax in (ax_loss, ax_corr):
				ax.axvspan(full_coverage_epoch, max_epoch, color="grey", alpha=0.08)
			mid_shaded = (full_coverage_epoch + max_epoch) / 2
			ax_corr.text(
				mid_shaded, ax_corr.get_ylim()[0] + 0.03 * (ax_corr.get_ylim()[1] - ax_corr.get_ylim()[0]),
				f"fewer than {n_seeds_total}\nseeds",
				fontsize=7, color="dimgrey", ha="center", va="bottom",
			)

		ax_loss.axvline(full_coverage_epoch, color="grey", linestyle=":", linewidth=1)
		ax_corr.axvline(full_coverage_epoch, color="grey", linestyle=":", linewidth=1)
		ax_loss.set_ylabel("msrr loss")
		ax_loss.set_title(f"learning curve, {variant} variant , black dots mark best epoch for each seed")
		ax_loss.legend(loc="upper right", fontsize=8)
		ax_corr.set_ylabel("val rank corr 6m")
		ax_corr.set_xlabel("epoch")
		ax_corr.legend(loc="upper right", fontsize=8)
		fig.tight_layout()
		fig.savefig(out_dir / f"learning_curve_{variant}.png", dpi=140)
		plt.close(fig)


def plot_best_epoch_gap(curve_df, out_dir):
	# the cleanest single check for a train versus val gap, no cross seed
	# averaging at all, just the train loss and val loss each seed actually had
	# at the epoch its own checkpoint was taken from, one bar pair per seed,
	# grouped by variant, so the comparison across variants is direct
	if curve_df.empty:
		return
	rows = []
	for (variant, seed), group in curve_df.groupby(["variant", "seed"]):
		best_epoch = group["best_epoch"].iloc[0]
		best_row = group.loc[group["epoch"] == best_epoch]
		if best_row.empty:
			continue
		rows.append({
			"variant": variant,
			"seed": seed,
			"best_epoch": best_epoch,
			"train_loss": best_row["train_loss"].iloc[0],
			"val_loss": best_row["val_loss"].iloc[0],
		})
	best_df = pd.DataFrame(rows)
	if best_df.empty:
		return
	best_df["gap_at_best_epoch"] = best_df["val_loss"] - best_df["train_loss"]
	out_dir.mkdir(parents=True, exist_ok=True)
	best_df.to_csv(out_dir / "gap_at_best_epoch_by_seed.csv", index=False)

	variants = sorted(best_df["variant"].unique())
	fig, axes = plt.subplots(1, len(variants), figsize=(3.2 * len(variants), 5), sharey=True)
	if len(variants) == 1:
		axes = [axes]
	for ax, variant in zip(axes, variants):
		sub = best_df[best_df["variant"] == variant].sort_values("seed")
		x_positions = np.arange(len(sub))
		width = 0.35
		ax.bar(x_positions - width / 2, sub["train_loss"], width, color="tab:blue", label="train loss")
		ax.bar(x_positions + width / 2, sub["val_loss"], width, color="tab:orange", label="val loss")
		ax.set_xticks(x_positions)
		ax.set_xticklabels([f"seed {s}" for s in sub["seed"]], fontsize=8)
		ax.set_title(variant, fontsize=10)
		if ax is axes[0]:
			ax.set_ylabel("msrr loss, at each seed's own best epoch")
			ax.legend(fontsize=8)
	fig.suptitle("train vs validation loss, per seed", fontsize=10)
	fig.tight_layout(rect=[0, 0, 1, 0.94])
	fig.savefig(out_dir / "gap_at_best_epoch_by_variant.png", dpi=140)
	plt.close(fig)


def plot_val_test_gap(val_test_df, out_dir):
	if val_test_df.empty:
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	variants = sorted(val_test_df["variant"].unique())
	fig, ax = plt.subplots(figsize=(9, 5))
	x_positions = np.arange(len(variants))
	width = 0.35
	val_means = [val_test_df.loc[val_test_df["variant"] == v, "val_corr_6m"].mean() for v in variants]
	test_means = [val_test_df.loc[val_test_df["variant"] == v, "test_corr_6m"].mean() for v in variants]
	val_stds = [val_test_df.loc[val_test_df["variant"] == v, "val_corr_6m"].std() for v in variants]
	test_stds = [val_test_df.loc[val_test_df["variant"] == v, "test_corr_6m"].std() for v in variants]
	ax.bar(x_positions - width / 2, val_means, width, yerr=val_stds, label="validation, combined score", color="tab:blue", capsize=4)
	ax.bar(x_positions + width / 2, test_means, width, yerr=test_stds, label="test, combined score", color="tab:orange", capsize=4)
	ax.set_xticks(x_positions)
	ax.set_xticklabels(variants)
	ax.set_ylabel("rank correlation, 6m target")
	ax.set_title("validation vs test rank correlation, per seed mean and std")
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_dir / "val_vs_test_rank_corr.png", dpi=140)
	plt.close(fig)


def plot_val_test_scatter(val_test_df, out_dir):
	# the single most direct picture of a generalization gap, one point per
	# seed, validation rank corr on x, test rank corr on y, against the y equal
	# to x line, a variant with no overfitting problem should hug that line,
	# a variant that overfits the validation window should sit below it
	if val_test_df.empty:
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	variants = sorted(val_test_df["variant"].unique())
	colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
	fig, ax = plt.subplots(figsize=(6.5, 6.5))
	lo = min(val_test_df["val_corr_6m"].min(), val_test_df["test_corr_6m"].min())
	hi = max(val_test_df["val_corr_6m"].max(), val_test_df["test_corr_6m"].max())
	pad = 0.02
	ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, linestyle="--", label="y equal to x, no gap")
	for variant, color in zip(variants, colors):
		sub = val_test_df[val_test_df["variant"] == variant]
		ax.scatter(sub["val_corr_6m"], sub["test_corr_6m"], color=color, label=variant, s=45, edgecolor="white", linewidth=0.5)
	ax.set_xlim(lo - pad, hi + pad)
	ax.set_ylim(lo - pad, hi + pad)
	ax.set_xlabel("validation rank corr 6m, combined score, at best epoch")
	ax.set_ylabel("test rank corr 6m, combined score")
	ax.set_title("validation vs test rank correlation, one point per seed")
	ax.legend(fontsize=8)
	ax.set_aspect("equal", adjustable="box")
	fig.tight_layout()
	fig.savefig(out_dir / "val_vs_test_scatter.png", dpi=140)
	plt.close(fig)


def analyze_hpt_epoch_history(hpt_dir, out_dir):
	paths = sorted(glob.glob(str(hpt_dir / "epoch_history_*.csv")))
	if not paths:
		return None
	frames = [pd.read_csv(p) for p in paths]
	df = pd.concat(frames, ignore_index=True)
	df["gap_loss"] = df["val_loss"] - df["train_loss"]
	summary_rows = []
	for (variant, trial_number), group in df.groupby(["variant", "trial_number"]):
		group = group.sort_values("epoch")
		n_epochs = len(group)
		best_row_idx = group["val_loss"].idxmin()
		best_epoch = group.loc[best_row_idx, "epoch"]
		final_gap = group["gap_loss"].iloc[-1]
		mean_gap_last_quarter = group["gap_loss"].iloc[max(0, n_epochs - max(1, n_epochs // 4)):].mean()
		summary_rows.append({
			"variant": variant,
			"trial_number": trial_number,
			"n_epochs_trained": n_epochs,
			"best_epoch": best_epoch,
			"best_epoch_fraction": best_epoch / n_epochs if n_epochs > 0 else float("nan"),
			"final_gap_loss": final_gap,
			"mean_gap_loss_last_quarter": mean_gap_last_quarter,
		})
	summary_df = pd.DataFrame(summary_rows)
	out_dir.mkdir(parents=True, exist_ok=True)
	summary_df.to_csv(out_dir / "hpt_epoch_overfit_summary.csv", index=False)
	return summary_df


def analyze_hpt_trials(hpt_dir, epoch_summary_df, out_dir):
	candidate_paths = sorted(glob.glob(str(hpt_dir / "trials_*.csv")))
	master_path = hpt_dir / "all_trials_summary.csv"
	if master_path.exists():
		candidate_paths = [str(master_path)]
	if not candidate_paths:
		return
	frames = [pd.read_csv(p) for p in candidate_paths]
	trials_df = pd.concat(frames, ignore_index=True)
	if "number" not in trials_df.columns or "value" not in trials_df.columns:
		return
	trials_df = trials_df.rename(columns={"number": "trial_number_zero_based"})
	trials_df["trial_number"] = trials_df["trial_number_zero_based"] + 1
	if epoch_summary_df is None or epoch_summary_df.empty:
		return
	merged = trials_df.merge(epoch_summary_df, on=["variant", "trial_number"], how="inner")
	if merged.empty:
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(8, 6))
	completed = merged[merged["state"] == "COMPLETE"] if "state" in merged.columns else merged
	pruned = merged[merged["state"] == "PRUNED"] if "state" in merged.columns else merged.iloc[0:0]
	ax.scatter(completed["final_gap_loss"], completed["value"], c="tab:blue", label="completed trials", alpha=0.7)
	if not pruned.empty:
		ax.scatter(pruned["final_gap_loss"], pruned["value"], c="tab:red", label="pruned trials", alpha=0.5, marker="x")
	ax.set_xlabel("val loss minus train loss, final logged epoch")
	ax.set_ylabel("trial objective, validation long short sharpe")
	ax.set_title("hpt trial outcome against the train vs val loss gap")
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_dir / "hpt_trial_gap_vs_objective.png", dpi=140)
	plt.close(fig)

	completed_with_value = completed.dropna(subset=["final_gap_loss", "value"])
	if len(completed_with_value) >= 3:
		corr = completed_with_value["final_gap_loss"].corr(completed_with_value["value"])
		print(f"hpt trial gap loss vs val sharpe correlation, {corr:.4f}")

	merged.to_csv(out_dir / "hpt_trial_overfit_joined.csv", index=False)


def print_val_test_summary(val_test_df):
	if val_test_df.empty:
		return
	print("")
	print("validation vs test rank correlation, per variant, paired across seeds")
	for variant, group in val_test_df.groupby("variant"):
		result = paired_gap_test(group["val_corr_6m"], group["test_corr_6m"])
		n_seeds_declined = int((group["test_corr_6m"] < group["val_corr_6m"]).sum())
		print(
			f"variant {variant}, n seeds {result['n']}, mean test minus val gap {result['mean_gap']:.4f}, "
			f"seeds where test below val {n_seeds_declined} of {result['n']}, "
			f"paired t p value {result['p_value']:.4f}, wilcoxon p value {result['wilcoxon_p']:.4f}"
		)


def print_combined_vs_base_summary(gap_df):
	if gap_df.empty:
		return
	print("")
	print("combined score minus base score, validation vs test, per seed, per variant")
	for variant, group in gap_df.groupby("variant"):
		val_mean = group["val_combined_minus_base"].mean()
		test_mean = group["test_combined_minus_base"].mean()
		n_seeds_flip = int(((group["val_combined_minus_base"] > 0) & (group["test_combined_minus_base"] < 0)).sum())
		print(f"variant {variant}, val mean {val_mean:.4f}, test mean {test_mean:.4f}, seeds flipping sign {n_seeds_flip} of {len(group)}")


def print_base_only_summary(val_test_df):
	# isolates whether the validation to test decline is a whole model, whole
	# regime effect, which would show up in base_6m alone as much as in the
	# combined score, or whether it is concentrated in path 2's contribution,
	# which would show base_6m holding up on test while the combined score
	# does not, pointing at path 2 specifically rather than at the encoder or
	# at a shift between the validation and test windows
	if val_test_df.empty:
		return
	print("")
	print("base score only, validation vs test rank correlation, per variant, paired across seeds")
	for variant, group in val_test_df.groupby("variant"):
		result = paired_gap_test(group["val_base_corr_6m"], group["test_base_corr_6m"])
		n_seeds_declined = int((group["test_base_corr_6m"] < group["val_base_corr_6m"]).sum())
		print(
			f"variant {variant}, n seeds {result['n']}, mean test minus val gap {result['mean_gap']:.4f}, "
			f"seeds where test below val {n_seeds_declined} of {result['n']}, "
			f"paired t p value {result['p_value']:.4f}, wilcoxon p value {result['wilcoxon_p']:.4f}"
		)


def main():
	parser = argparse.ArgumentParser(description="overfitting diagnostics built from the data saved by hpt_dual_path.py and nonlinear_dualapproach.py")
	parser.add_argument("--results-dir", type=Path, default=Path("results/transformer"))
	parser.add_argument("--hpt-dir", type=Path, default=Path("results/transformer/transformer-hpt"))
	parser.add_argument("--out-dir", type=Path, default=Path("results/transformer/overfit_diagnostics"))
	args = parser.parse_args()

	records = discover_seed_metrics(args.results_dir)

	curve_df = build_learning_curve_table(records)
	val_test_df = build_val_test_table(records)
	combined_vs_base_df = build_combined_vs_base_gap_table(val_test_df)

	args.out_dir.mkdir(parents=True, exist_ok=True)
	if not curve_df.empty:
		curve_df.to_csv(args.out_dir / "learning_curves_long.csv", index=False)
	if not val_test_df.empty:
		val_test_df.to_csv(args.out_dir / "val_vs_test_by_seed.csv", index=False)
	if not combined_vs_base_df.empty:
		combined_vs_base_df.to_csv(args.out_dir / "combined_vs_base_by_seed.csv", index=False)

	plot_learning_curves(curve_df, args.out_dir)
	plot_best_epoch_gap(curve_df, args.out_dir)
	plot_val_test_gap(val_test_df, args.out_dir)
	plot_val_test_scatter(val_test_df, args.out_dir)
	print_val_test_summary(val_test_df)
	print_base_only_summary(val_test_df)
	print_combined_vs_base_summary(combined_vs_base_df)

	epoch_summary_df = analyze_hpt_epoch_history(args.hpt_dir, args.out_dir)
	analyze_hpt_trials(args.hpt_dir, epoch_summary_df, args.out_dir)

	print("")
	print(f"outputs written under, {args.out_dir.resolve()}")


if __name__ == "__main__":
	main()
