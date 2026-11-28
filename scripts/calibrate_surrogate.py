"""
calibration check. verifies that ensemble std (uncertainty) correlates with
actual prediction error. if it does, the acquisition function's exploration
term in phase 5 has real signal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.utils.seeding import seed_everything

seed_everything(0)

processed = Path("data/processed")
cache_dir = Path("cache/embeddings")
model_dir = Path("cache/models")
fig_dir = Path("results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)


# load embeddings
print("loading embeddings...")
with open(cache_dir / "antibody_heavy.pkl", "rb") as f: heavy_emb = pickle.load(f)
with open(cache_dir / "antibody_light.pkl", "rb") as f: light_emb = pickle.load(f)
with open(cache_dir / "virus_env.pkl", "rb") as f: virus_emb = pickle.load(f)

# load models
print("loading ensemble...")
with open(model_dir / "ensemble_meta.pkl", "rb") as f: meta = pickle.load(f)
n_models = meta["n_models"]
models = [joblib.load(model_dir / f"rf_{i:02d}.joblib") for i in range(n_models)]
print(f"  loaded {len(models)} models")


print("\nbuilding test_both features...")
test_both = pd.read_csv(processed / "test_pairs_both_heldout.csv")

X_list, y_list = [], []
for _, row in test_both.iterrows():
    ab = row["antibody_id"]
    v = row["virus_id"]
    if ab in heavy_emb and ab in light_emb and v in virus_emb:
        X_list.append(np.concatenate([heavy_emb[ab], light_emb[ab], virus_emb[v]]))
        y_list.append(row["pIC50"])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"  n = {len(X)}")

if len(X) < 100:
    print("too few samples to assess calibration meaningfully. aborting.")
    sys.exit(1)


# ensemble predictions and std
print("\npredicting with ensemble...")
preds = np.stack([m.predict(X) for m in models], axis=0)  # (n_models, n_samples)
y_mean = preds.mean(axis=0)
y_std = preds.std(axis=0)
abs_err = np.abs(y - y_mean)

print(f"\nensemble std distribution:")
print(f"  min={y_std.min():.4f}, median={np.median(y_std):.4f}, "
      f"mean={y_std.mean():.4f}, max={y_std.max():.4f}")


# correlation between std and |error|
print("\n" + "=" * 50)
print("CALIBRATION METRICS")
print("=" * 50)

pear_r = pearsonr(y_std, abs_err)[0]
spear_r = spearmanr(y_std, abs_err)[0]
print(f"  pearson(std, |error|):   {pear_r:.3f}")
print(f"  spearman(std, |error|):  {spear_r:.3f}")


# metric 2: quantile-binned calibration

n_bins = 5
bin_edges = np.quantile(y_std, np.linspace(0, 1, n_bins + 1))
bin_idx = np.digitize(y_std, bin_edges[1:-1])

print(f"\n  quantile-binned calibration (want mean|error| to INCREASE):")
print(f"  {'bin':>5} {'n':>5} {'std_mean':>10} {'|err|_mean':>12}")
bin_std = []
bin_err = []
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    s = y_std[mask].mean()
    e = abs_err[mask].mean()
    bin_std.append(s)
    bin_err.append(e)
    print(f"  {b+1:>5} {mask.sum():>5} {s:>10.4f} {e:>12.4f}")

bin_trend_corr = pearsonr(bin_std, bin_err)[0] if len(bin_std) > 2 else float("nan")
print(f"\n  bin-trend pearson: {bin_trend_corr:.3f} (want positive)")

# if the ensemble is well-calibrated, ~90% of true values should lie within
lower = y_mean - 1.645 * y_std
upper = y_mean + 1.645 * y_std
coverage = ((y >= lower) & (y <= upper)).mean()
print(f"\n  90% interval coverage: {coverage:.1%} (want close to 90%)")


# figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# scatter: std vs |error|
ax = axes[0]
ax.scatter(y_std, abs_err, alpha=0.3, s=8)
ax.set_xlabel("ensemble std (predicted uncertainty)")
ax.set_ylabel("|error| (actual)")
ax.set_title(f"Calibration scatter (pearson={pear_r:.3f}, spearman={spear_r:.3f})")
z = np.polyfit(y_std, abs_err, 1)
x_line = np.linspace(y_std.min(), y_std.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6, label="trend")
ax.legend()
ax.grid(alpha=0.3)

# binned bars
ax = axes[1]
ax.bar(range(1, len(bin_err) + 1), bin_err, color="steelblue")
ax.set_xlabel("std quintile (1=lowest, 5=highest)")
ax.set_ylabel("mean |error|")
ax.set_title("Binned calibration")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = fig_dir / "surrogate_calibration.png"
plt.savefig(out_path, dpi=150)
print(f"\nsaved figure to {out_path}")


print("VERDICT")

if spear_r > 0.30 and bin_trend_corr > 0.7:
    print(" WELL CALIBRATED")
    print(f"   spearman={spear_r:.3f} (threshold >0.30)")
    print(f"   bin-trend pearson={bin_trend_corr:.3f} (threshold >0.7)")
    print("   uncertainty term in phase 5 acquisition will be meaningful.")
elif spear_r > 0.15 and bin_trend_corr > 0.4:
    print("MARGINALLY CALIBRATED")
    print(f"   spearman={spear_r:.3f}, bin-trend={bin_trend_corr:.3f}")
    print("   uncertainty signal is weak but real. phase 5 lambda may need to be small (0.05-0.1).")
elif spear_r > 0:
    print("WEAK CALIBRATION")
    print(f"   spearman={spear_r:.3f}, bin-trend={bin_trend_corr:.3f}")
    print("   std correlates with error but weakly. using std only as a tie-breaker.")
    print("   phase 5 lambda should be small (~0.01) or zero (pure exploitation).")
else:
    print(f" NEGATIVE OR ZERO CORRELATION (spearman={spear_r:.3f})")
    print("   ensemble std does NOT track error. do not use uncertainty bonus in phase 5.")
    print("   set lambda=0. eHVI alone will be more informative than adding noise.")