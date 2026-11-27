"""
build escape mutation pools from the processed DMS data.
filter to VRC01-class antibodies (3BNC117 datasets only; 10-1074 is V3-glycan)
filter to VRC01 epitope region (HXB2 425-510)
split into strong and weak pools 
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.seeding import seed_everything

seed_everything(0)

interim = Path("data/interim")
processed = Path("data/processed")
processed.mkdir(parents=True, exist_ok=True)


# load dms table 
dms = pd.read_csv(interim / "dms_escape_mutations.csv")
print(f"loaded {len(dms)} mutations from {dms['dms_source'].nunique()} datasets")
print(f"  targets in data: {sorted(dms['antibody_target'].unique())}")


# filter to VRC01-class antibodies
vrc01_targets = ["3BNC117"]

dms_vrc01 = dms[dms["antibody_target"].isin(vrc01_targets)].copy()
print(f"\nafter VRC01-class filter (targets={vrc01_targets}): {len(dms_vrc01)} mutations")
print(f"  sources kept: {sorted(dms_vrc01['dms_source'].unique())}")


# filter to VRC01 epitope region (HXB2 425-510)
before = len(dms_vrc01)
dms_vrc01 = dms_vrc01[
    (dms_vrc01["hxb2_site"] >= 425) & (dms_vrc01["hxb2_site"] <= 510)
].copy()
print(f"\nafter epitope filter (HXB2 425-510): {len(dms_vrc01)} mutations "
      f"(dropped {before - len(dms_vrc01)})")


# split into pools
strong_pool = dms_vrc01[dms_vrc01["escape_tier"] == "strong"].copy()
moderate_pool = dms_vrc01[dms_vrc01["escape_tier"] == "moderate"].copy()
weak_pool = dms_vrc01[dms_vrc01["escape_tier"] == "weak"].copy()

print(f"\nraw pool sizes (before dedup):")
print(f"  strong:   {len(strong_pool)}")
print(f"  moderate: {len(moderate_pool)}")
print(f"  weak:     {len(weak_pool)}")


# deduplicate
def dedupe_pool(pool, tier_name):
    if len(pool) == 0:
        return pool
    out = (
        pool.groupby(["hxb2_site", "mutant"], as_index=False)
        .agg(
            wildtype=("wildtype", "first"),
            mut_effect=("mut_effect", "mean"),
            n_sources=("dms_source", "nunique"),
        )
    )
    print(f"  {tier_name}: {len(pool)} -> {len(out)} unique (site, mutant) pairs")
    return out


print(f"\nafter deduplication:")
strong_pool = dedupe_pool(strong_pool, "strong")
moderate_pool = dedupe_pool(moderate_pool, "moderate")
weak_pool = dedupe_pool(weak_pool, "weak")


strong_pool.to_csv(processed / "escape_pool_strong.csv", index=False)
weak_pool.to_csv(processed / "escape_pool_weak.csv", index=False)
moderate_pool.to_csv(processed / "escape_pool_moderate.csv", index=False)

print(f"\nsaved to:")
print(f"  {processed / 'escape_pool_strong.csv'}")
print(f"  {processed / 'escape_pool_weak.csv'}")
print(f"  {processed / 'escape_pool_moderate.csv'}")


# for logbook
print(f"\n{'='*50}")
print("STRONG POOL SUMMARY")
print('='*50)
print(f"unique mutations: {len(strong_pool)}")
print(f"unique sites:     {strong_pool['hxb2_site'].nunique()}")
print(f"unique mutants:   {strong_pool['mutant'].nunique()}")
print(f"site range:       {strong_pool['hxb2_site'].min()}-{strong_pool['hxb2_site'].max()}")

print(f"\nsites per HXB2 sub-region:")
for lo, hi, name in [(425, 440, "425-440"), (441, 460, "441-460"),
                      (461, 480, "461-480"), (481, 510, "481-510")]:
    n = ((strong_pool["hxb2_site"] >= lo) & (strong_pool["hxb2_site"] <= hi)).sum()
    print(f"  {name}: {n} mutations")

print(f"\nfirst 5 strong mutations:")
print(strong_pool.head(5).to_string(index=False))


print(f"\nmodule 2.7 done.")