"""
DMS preprocessing. loads all DMS mutation-effect CSVs, standardizes columns,
applies functional effect filter, tiers by escape strength, maps to HXB2,
saves unified escape mutation table.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.seeding import seed_everything


seed_everything(0)

raw = Path("data/raw/dms")
out = Path("data/interim")
out.mkdir(parents=True, exist_ok=True)


datasets = [
    {
        "name": "TRO11_3BNC117",
        "mut_path": raw / "TRO11_Env_3BNC117_mut_effects_filtered.csv",
        "func_path": raw / "TRO11_Env_TZM-bl_entry_func_effects_filtered.csv",
        "ab_target": "3BNC117",
        "ref_name": "TRO11",
        "map_path": raw / "TRO11_to_HXB.csv",
    },
    {
        "name": "TRO11_10-1074",
        "mut_path": raw / "TRO11_Env_10-1074_mut_effects_filtered.csv",
        "func_path": raw / "TRO11_Env_TZM-bl_entry_func_effects_filtered.csv",
        "ab_target": "10-1074",
        "ref_name": "TRO11",
        "map_path": raw / "TRO11_to_HXB.csv",
    },
    {
        "name": "BF520_3BNC117",
        "mut_path": raw / "BF520_Env_3BNC117_mut_effects_filtered.csv",
        "func_path": raw / "BF520_Env_TZM-bl_entry_func_effects_filtered.csv",
        "ab_target": "3BNC117",
        "ref_name": "BF520",
        "map_path": raw / "BF520_to_HXB.csv",
    },
    {
        "name": "BF520_10-1074",
        "mut_path": raw / "BF520_Env_10-1074_mut_effects_filtered.csv",
        "func_path": raw / "BF520_Env_TZM-bl_entry_func_effects_filtered.csv",
        "ab_target": "10-1074",
        "ref_name": "BF520",
        "map_path": raw / "BF520_to_HXB.csv",
    },
]


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"couldnt find {label} col. tried {candidates}. actual cols: {list(df.columns)}")


all_dfs = []
loaded_names = []
skipped_names = []

for ds in datasets:
    print(f"\n{'='*50}")
    print(f"processing: {ds['name']}")
    print('='*50)

    if not ds["mut_path"].exists():
        print(f"  SKIP: missing mut file {ds['mut_path'].name}")
        skipped_names.append(ds["name"])
        continue
    if not ds["map_path"].exists():
        print(f"  SKIP: missing map file {ds['map_path'].name}")
        skipped_names.append(ds["name"])
        continue

    mut_df = pd.read_csv(ds["mut_path"])
    print(f"  mut shape: {mut_df.shape}")

    site_col = pick_col(mut_df, ["site", "position", "Site"], "site")
    wt_col = pick_col(mut_df, ["wildtype", "wild_type", "wt", "WT"], "wildtype")
    mut_col = pick_col(mut_df, ["mutant", "mut", "mutation"], "mutant")
    effect_col = pick_col(mut_df, ["mut_effect", "effect", "mut_escape", "escape", "escape_mean", "escape_median"], "mut_effect")
    print(f"  using: site='{site_col}' wt='{wt_col}' mut='{mut_col}' effect='{effect_col}'")

    core = mut_df[[site_col, wt_col, mut_col, effect_col]].copy()
    core.columns = ["ref_site", "wildtype", "mutant", "mut_effect"]

    core["ref_site"] = pd.to_numeric(core["ref_site"], errors="coerce")
    before = len(core)
    core = core[core["ref_site"].notna()].copy()
    print(f"  dropped {before - len(core)} rows with non-numeric site")
    core["ref_site"] = core["ref_site"].astype(int)

    for cand in ["n_models", "n_libs", "n_replicates"]:
        if cand in mut_df.columns:
            core["n_models"] = mut_df[cand]
            break
    else:
        core["n_models"] = np.nan

    # func effect
    if "func_effect" in mut_df.columns:
        core["func_effect"] = mut_df["func_effect"]
        print("  func_effect already in mut file")
    elif ds["func_path"] is not None and ds["func_path"].exists():
        print(f"  loading func from: {ds['func_path'].name}")
        func_df = pd.read_csv(ds["func_path"])
        try:
            f_site = pick_col(func_df, ["site", "position", "Site"], "func site")
            f_wt = pick_col(func_df, ["wildtype", "wild_type", "wt", "WT"], "func wt")
            f_mut = pick_col(func_df, ["mutant", "mut", "mutation"], "func mut")
            f_eff = pick_col(func_df, ["func_effect", "mut_func_effect", "function_effect", "effect"], "func effect")

            func_core = func_df[[f_site, f_wt, f_mut, f_eff]].copy()
            func_core.columns = ["ref_site", "wildtype", "mutant", "func_effect"]
            func_core["ref_site"] = pd.to_numeric(func_core["ref_site"], errors="coerce")
            func_core = func_core[func_core["ref_site"].notna()].copy()
            func_core["ref_site"] = func_core["ref_site"].astype(int)
            func_core["wildtype"] = func_core["wildtype"].astype(str)
            func_core["mutant"] = func_core["mutant"].astype(str)

            core["wildtype"] = core["wildtype"].astype(str)
            core["mutant"] = core["mutant"].astype(str)

            before = len(core)
            core = core.merge(func_core, on=["ref_site", "wildtype", "mutant"], how="left")
            print(f"  matched func_effect for {core['func_effect'].notna().sum()} / {before}")
        except KeyError as e:
            print(f"  couldnt parse func file: {e}")
            core["func_effect"] = np.nan
    else:
        print(f"  no func_effect file for {ds['name']}")
        core["func_effect"] = np.nan

    # quality filters
    before = len(core)
    core = core[core["mut_effect"].notna()].copy()
    print(f"  dropped {before - len(core)} rows with nan mut_effect")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    before = len(core)
    core = core[core["mutant"].isin(valid_aa)].copy()
    core = core[core["wildtype"].isin(valid_aa)].copy()
    print(f"  dropped {before - len(core)} rows with invalid amino acids")

    before = len(core)
    core = core[core["mutant"] != core["wildtype"]].copy()
    print(f"  dropped {before - len(core)} synonymous mutations")

    print(f"  loading site map: {ds['map_path'].name}")
    site_map = pd.read_csv(ds["map_path"])

    try:
        map_seq  = pick_col(site_map, ["sequential_site", "seq_site"], "map sequential")
        map_swt  = pick_col(site_map, ["sequential_wt", "seq_wt"], "map seq wt")
        map_hxb  = pick_col(site_map, ["reference_site", "hxb2_site"], "map HXB2")
        map_hwt  = pick_col(site_map, ["reference_wt", "hxb2_wt"], "map HXB2 wt")
    except KeyError as e:
        print(f"  SKIP {ds['name']}: {e}")
        skipped_names.append(ds["name"])
        continue

    seq_map = site_map[[map_seq, map_swt]].copy()
    seq_map.columns = ["seq_site", "seq_wt"]
    seq_map["seq_site"] = pd.to_numeric(seq_map["seq_site"], errors="coerce")
    seq_map = seq_map.dropna().drop_duplicates("seq_site")
    seq_map["seq_site"] = seq_map["seq_site"].astype(int)

    hxb_map = site_map[[map_hxb, map_hwt]].copy()
    hxb_map.columns = ["hxb2_site", "hxb2_wt"]
    hxb_map["hxb2_site"] = pd.to_numeric(hxb_map["hxb2_site"], errors="coerce")
    hxb_map = hxb_map.dropna(subset=["hxb2_site"]).drop_duplicates("hxb2_site")
    hxb_map["hxb2_site"] = hxb_map["hxb2_site"].astype(int)

    # mut.site is sequential (TRO11/BF520 numbering)
    test_a = core.merge(seq_map, left_on="ref_site", right_on="seq_site", how="inner")
    match_a = (test_a["wildtype"] == test_a["seq_wt"]).mean() if len(test_a) > 0 else 0

    # mut.site is already HXB2
    test_b = core.merge(hxb_map, left_on="ref_site", right_on="hxb2_site", how="inner")
    match_b = (test_b["wildtype"] == test_b["hxb2_wt"]).mean() if len(test_b) > 0 else 0

    print(f"  wt match if sequential numbering: {match_a:.1%} (n={len(test_a)})")
    print(f"  wt match if HXB2 numbering:       {match_b:.1%} (n={len(test_b)})")

    if match_b > match_a and match_b > 0.5:
        print(f"  -> mut.site is already HXB2. using it directly.")
        core["hxb2_site"] = core["ref_site"]
    else:
        print(f"  -> mut.site is sequential. mapping to HXB2.")
        # build seq 
        seq_to_hxb = site_map[[map_seq, map_hxb]].copy()
        seq_to_hxb.columns = ["seq_site", "hxb2_site"]
        seq_to_hxb["seq_site"]  = pd.to_numeric(seq_to_hxb["seq_site"],  errors="coerce")
        seq_to_hxb["hxb2_site"] = pd.to_numeric(seq_to_hxb["hxb2_site"], errors="coerce")
        seq_to_hxb = seq_to_hxb.dropna().drop_duplicates("seq_site")
        seq_to_hxb["seq_site"]  = seq_to_hxb["seq_site"].astype(int)
        seq_to_hxb["hxb2_site"] = seq_to_hxb["hxb2_site"].astype(int)

        core = core.merge(seq_to_hxb, left_on="ref_site", right_on="seq_site", how="left")
        core = core.drop(columns=["seq_site"], errors="ignore")

    before = len(core)
    core = core[core["hxb2_site"].notna()].copy()
    print(f"  dropped {before - len(core)} mutations with no HXB2 mapping")
    core["hxb2_site"] = core["hxb2_site"].astype(int)

    core["dms_source"] = ds["name"]
    core["antibody_target"] = ds["ab_target"]
    core["ref_name"] = ds["ref_name"]

    all_dfs.append(core)
    loaded_names.append(ds["name"])


if len(all_dfs) == 0:
    raise ValueError("no datasets loaded. check download state above.")

combined = pd.concat(all_dfs, ignore_index=True)
print(f"\n{'='*50}")
print(f"combined: {len(combined)} mutations from {len(all_dfs)} datasets")
print('='*50)


print("\nmut_effect distribution (per dataset):")
print(combined.groupby("dms_source")["mut_effect"].describe().round(3))


# func_effect filter 
before = len(combined)
combined = combined[combined["func_effect"].isna() | (combined["func_effect"] > -0.5)].copy()
print(f"\nfunc_effect filter (>-0.5 to keep): {before} -> {len(combined)}")


# top 5% = strong, next 15% = moderate, rest = weak
combined["escape_tier"] = "unknown"

for source in combined["dms_source"].unique():
    mask = combined["dms_source"] == source
    sub = combined.loc[mask, "mut_effect"]
    q95 = sub.quantile(0.95)
    q80 = sub.quantile(0.80)
    print(f"  {source}: q80={q80:.3f}, q95={q95:.3f}")

    combined.loc[mask & (sub >= q95), "escape_tier"] = "strong"
    combined.loc[mask & (sub >= q80) & (sub < q95), "escape_tier"] = "moderate"
    combined.loc[mask & (sub < q80) & sub.notna(), "escape_tier"] = "weak"

print("\nescape tier counts:")
print(combined["escape_tier"].value_counts())
print("\ntier breakdown per dataset:")
print(combined.groupby(["dms_source", "escape_tier"]).size().unstack(fill_value=0))


final_cols = [
    "antibody_target", "dms_source", "ref_name",
    "ref_site", "hxb2_site", "wildtype", "mutant",
    "mut_effect", "func_effect", "escape_tier", "n_models",
]
combined = combined[final_cols]

out_path = out / "dms_escape_mutations.csv"
combined.to_csv(out_path, index=False)
print(f"\nsaved to {out_path}")

print(f"\nloaded datasets: {loaded_names}")
if skipped_names:
    print(f"skipped datasets: {skipped_names}")

print(f"\nfinal summary:")
print(f"  total mutations: {len(combined)}")
print(f"  unique HXB2 sites: {combined['hxb2_site'].nunique()}")