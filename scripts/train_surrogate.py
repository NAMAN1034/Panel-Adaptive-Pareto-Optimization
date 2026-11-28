"""
train the surrogate ensemble. builds (antibody, virus) feature vectors from
cached esm-2 embeddings, trains 10 random forest regressors on bootstrap samples
of the training pairs, evaluates on the held-out test sets, saves models.
"""

import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from src.utils.seeding import seed_everything

seed_everything(0)

processed = Path("data/processed")
cache_dir = Path("cache/embeddings")
model_dir = Path("cache/models")
model_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path("results/tables")
results_dir.mkdir(parents=True, exist_ok=True)


# args
parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true",
                    help="train 1 model on 10%% of data, verify pipeline, exit")
parser.add_argument("--n-models", type=int, default=10,
                    help="ensemble size (default 10)")
parser.add_argument("--n-trees", type=int, default=100,
                    help="trees per random forest (default 100)")
parser.add_argument("--n-jobs", type=int, default=-1,
                    help="parallel jobs (-1 = all cores)")
args = parser.parse_args()



print("loading embedding caches...")
with open(cache_dir / "antibody_heavy.pkl", "rb") as f:
    heavy_emb = pickle.load(f)
with open(cache_dir / "antibody_light.pkl", "rb") as f:
    light_emb = pickle.load(f)
with open(cache_dir / "virus_env.pkl", "rb") as f:
    virus_emb = pickle.load(f)

print(f"  heavy: {len(heavy_emb)}")
print(f"  light: {len(light_emb)}")
print(f"  virus: {len(virus_emb)}")

sample_dim = next(iter(heavy_emb.values())).shape[0]
print(f"  embedding dim: {sample_dim}")
assert sample_dim == 1280, f"expected 1280-dim embeddings, got {sample_dim}"




print("\nloading split tables...")
train_pairs = pd.read_csv(processed / "train_pairs.csv")
test_both = pd.read_csv(processed / "test_pairs_both_heldout.csv")
test_ab = pd.read_csv(processed / "test_pairs_antibody_heldout.csv")
test_v = pd.read_csv(processed / "test_pairs_virus_heldout.csv")

print(f"  train pairs: {len(train_pairs)}")
print(f"  test_both: {len(test_both)}")
print(f"  test_ab_only: {len(test_ab)}")
print(f"  test_v_only: {len(test_v)}")


# feature vector = [heavy(1280), light(1280), virus(1280)] = 3840 dims
# any pair where heavy, light, OR virus embedding is missing gets dropped

def build_features(pairs_df, name):
    """returns X (n x 3840) and y (n,) for pairs with all three embeddings."""
    X_list, y_list, kept_ids = [], [], []
    dropped_no_heavy = dropped_no_light = dropped_no_virus = 0

    for _, row in pairs_df.iterrows():
        ab = row["antibody_id"]
        v = row["virus_id"]

        if ab not in heavy_emb:
            dropped_no_heavy += 1
            continue
        if ab not in light_emb:
            dropped_no_light += 1
            continue
        if v not in virus_emb:
            dropped_no_virus += 1
            continue

        h = heavy_emb[ab]
        l = light_emb[ab]
        vir = virus_emb[v]
        X_list.append(np.concatenate([h, l, vir]))
        y_list.append(row["pIC50"])
        kept_ids.append((ab, v))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"  {name}: kept {len(X)}/{len(pairs_df)} "
          f"(dropped: no_heavy={dropped_no_heavy}, "
          f"no_light={dropped_no_light}, no_virus={dropped_no_virus})")

    return X, y, kept_ids


print("\nbuilding feature matrices...")
X_train, y_train, _ = build_features(train_pairs, "train")
X_test_both, y_test_both, _ = build_features(test_both, "test_both")
X_test_ab, y_test_ab, _ = build_features(test_ab, "test_ab_only")
X_test_v, y_test_v, _ = build_features(test_v, "test_v_only")

if len(X_train) < 5000:
    print(f"\nWARNING: only {len(X_train)} training pairs. surrogate will struggle.")
    print("if this is unexpectedly low, check embedding id matching.")

print(f"\nfeature dim: {X_train.shape[1]}")
print(f"pIC50 range: {y_train.min():.2f} to {y_train.max():.2f} "
      f"(mean {y_train.mean():.2f}, std {y_train.std():.2f})")



def eval_model(model, X, y, name):
    """quick Pearson + RMSE + MAE on a test set."""
    y_pred = model.predict(X)
    r = pearsonr(y, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"    {name}: n={len(y)}, pearson={r:.3f}, rmse={rmse:.3f}, mae={mae:.3f}")
    return {"pearson": r, "rmse": rmse, "mae": mae, "n": len(y)}


if args.smoke_test:
    print("\n" + "=" * 50)
    print("SMOKE TEST MODE")
    print("=" * 50)
    print("training 1 small model on 10% of train data to verify pipeline...")

    n_smoke = max(1000, len(X_train) // 10)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_train), size=n_smoke, replace=False)
    X_smoke = X_train[idx]
    y_smoke = y_train[idx]

    print(f"  smoke train size: {n_smoke}")
    print(f"  smoke trees: 20")

    model = RandomForestRegressor(
        n_estimators=20, n_jobs=args.n_jobs, random_state=0, verbose=0
    )
    t0 = time.time()
    model.fit(X_smoke, y_smoke)
    print(f"  trained in {time.time() - t0:.1f}s")

    print("\n  eval on held-out sets:")
    m_both = eval_model(model, X_test_both, y_test_both, "test_both")
    m_ab   = eval_model(model, X_test_ab, y_test_ab, "test_ab_only")
    m_v    = eval_model(model, X_test_v, y_test_v, "test_v_only")

    print("\n" + "=" * 50)
    if m_both["pearson"] < 0.3:
        print(f"SMOKE TEST FAILED: pearson on test_both = {m_both['pearson']:.3f} < 0.3")
        print("possible causes: feature misalignment, wrong y values, embedding cache corrupt")
        print("I shouln't run full training until this is fixed")
        sys.exit(1)
    elif m_both["pearson"] < 0.5:
        print(f"SMOKE TEST MARGINAL: pearson on test_both = {m_both['pearson']:.3f}")
        print("pipeline works but weak. full training may improve with 100+ trees.")
        print("proceeding is OK, but check results carefully.")
    else:
        print(f"SMOKE TEST PASSED: pearson on test_both = {m_both['pearson']:.3f}")

    sys.exit(0)


print("\n" + "=" * 50)
print(f"TRAINING {args.n_models}-MODEL ENSEMBLE")
print(f"  trees per model: {args.n_trees}")
print(f"  train samples: {len(X_train)}")
print(f"  feature dim: {X_train.shape[1]}")
print("=" * 50)

# different random seed per model. each RF also does internal tree bootstrap.

models = []
train_metrics = []

for i in range(args.n_models):
    print(f"\nmodel {i+1}/{args.n_models}...")
    rng = np.random.RandomState(seed=1000 + i)
    idx = rng.choice(len(X_train), size=len(X_train), replace=True)

    X_boot = X_train[idx]
    y_boot = y_train[idx]

    rf = RandomForestRegressor(
        n_estimators=args.n_trees,
        n_jobs=args.n_jobs,
        random_state=1000 + i,  
        verbose=0,
    )

    t0 = time.time()
    rf.fit(X_boot, y_boot)
    train_time = time.time() - t0

    m_both = eval_model(rf, X_test_both, y_test_both, "test_both")
    m_both["model_id"] = i
    m_both["train_time_s"] = round(train_time, 1)
    train_metrics.append(m_both)

    models.append(rf)


# logbook
print("\n" + "=" * 50)
print("ENSEMBLE SUMMARY")
print("=" * 50)

pearsons = [m["pearson"] for m in train_metrics]
rmses = [m["rmse"] for m in train_metrics]

print(f"per-model pearson (test_both): mean={np.mean(pearsons):.3f}, "
      f"std={np.std(pearsons):.3f}, min={np.min(pearsons):.3f}, max={np.max(pearsons):.3f}")
print(f"per-model rmse (test_both):    mean={np.mean(rmses):.3f}, "
      f"std={np.std(rmses):.3f}")

# ensemble prediction: average across models
def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)  # (n_models, n_samples)
    return preds.mean(axis=0), preds.std(axis=0)

print("\nevaluating ENSEMBLE prediction on held-out sets:")
for name, Xt, yt in [
    ("test_both", X_test_both, y_test_both),
    ("test_ab_only", X_test_ab, y_test_ab),
    ("test_v_only", X_test_v, y_test_v),
]:
    y_mean, y_std = ensemble_predict(models, Xt)
    r = pearsonr(yt, y_mean)[0]
    rmse = np.sqrt(mean_squared_error(yt, y_mean))
    mae = mean_absolute_error(yt, y_mean)
    print(f"  {name}: n={len(yt)}, pearson={r:.3f}, rmse={rmse:.3f}, mae={mae:.3f}")




print("\nsaving models...")
for i, m in enumerate(models):
    joblib.dump(m, model_dir / f"rf_{i:02d}.joblib")

meta = {
    "n_models": args.n_models,
    "n_trees": args.n_trees,
    "feature_dim": X_train.shape[1],
    "n_train_samples": len(X_train),
    "embedding_paths": {
        "heavy": str(cache_dir / "antibody_heavy.pkl"),
        "light": str(cache_dir / "antibody_light.pkl"),
        "virus": str(cache_dir / "virus_env.pkl"),
    },
    "per_model_metrics": train_metrics,
    "y_train_stats": {
        "min": float(y_train.min()),
        "max": float(y_train.max()),
        "mean": float(y_train.mean()),
        "std": float(y_train.std()),
    },
}

with open(model_dir / "ensemble_meta.pkl", "wb") as f:
    pickle.dump(meta, f)

metrics_df = pd.DataFrame(train_metrics)
metrics_df.to_csv(results_dir / "surrogate_train_metrics.csv", index=False)

print(f"  models saved to: {model_dir / 'rf_*.joblib'}")
print(f"  metadata saved to: {model_dir / 'ensemble_meta.pkl'}")
print(f"  metrics saved to: {results_dir / 'surrogate_train_metrics.csv'}")

print(f"\nmodel dir size: {sum(f.stat().st_size for f in model_dir.glob('*.joblib')) / 1e6:.1f} MB")
print(f"\nmodule 3.2 done. run 3.3 next for calibration check.")