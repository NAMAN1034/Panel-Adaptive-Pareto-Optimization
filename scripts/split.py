"""
train/test split by virus + antibody.
20% of viruses held out, stratified by clade
20% of antibodies held out, stratified by class (vrc01_class vs other)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.seeding import seed_everything


seed_everything(0)

interim = Path("data/interim")
processed = Path("data/processed")
processed.mkdir(parents=True, exist_ok=True)


# load the clean catnap table from module 2.3
clean = pd.read_csv(interim / "catnap_clean.csv")
print("clean table shape:", clean.shape)
print("unique antibodies:", clean["antibody_id"].nunique())
print("unique viruses:", clean["virus_id"].nunique())

if len(clean) == 0:
    raise ValueError("clean table is empty, go back and check module 2.3")


# ---------------------------------------------------------------
# step 1: classify antibodies so we can stratify the split
# ---------------------------------------------------------------
"""
the clean table doesn't have class info. i'm hardcoding the VRC01-class
members because that's what my optimizer will design against. anything
else goes into "other". the surrogate still learns from other-class
antibodies (good for diversity) but held-out gives me both types to test.
"""
vrc01_class_patterns = [
    # core VRC01 alphabet (VRC01-VRC08, VRC13, VRC16 are all VRC01-class)
    "VRC01", "VRC02", "VRC03", "VRC04", "VRC05", "VRC06", "VRC07", "VRC08",
    "VRC13", "VRC16",
    # CH31 / PG / FL families (hyphens stripped in normalization)
    "VRCCH31", "VRCPG04", "VRCPG20",
    "VRCFL06", "VRCFL08", "VRCFL09", "VRCFL10", "VRCFL11", "VRCFL12",
    # 3BNC / 12A / NIH45-46 families
    "3BNC117", "3BNC60", "3BNC62",
    "12A12", "12A21", "12A30", "12A58", "12A70", "12A80",
    "NIH4546",
    # PCIN / SBNC
    "SBNC", "PCIN63", "PCIN76",
]


def classify_antibody(ab_id):
    raw = str(ab_id).strip()
    ab_upper = raw.upper().replace(" ", "").replace("-", "").replace("_", "")

    # bispecifics and cocktails (contain + or / in the raw name) are NOT
    # pure single-antibody VRC01-class. exclude them so held-out eval
    if "+" in raw or "/" in raw:
        return "other"

    # N60P* lineage is NOT the N6 lineage, despite the shared prefix
    if ab_upper.startswith("N60P"):
        return "other"

    # n6 and its variants need strict matching 
    if ab_upper == "N6" or ab_upper.startswith("N6LS") or ab_upper.startswith("N6FR3"):
        return "vrc01_class"

    for pattern in vrc01_class_patterns:
        if ab_upper.startswith(pattern):
            return "vrc01_class"

    return "other"


clean["ab_class"] = clean["antibody_id"].apply(classify_antibody)

unique_abs = clean.drop_duplicates("antibody_id")[["antibody_id", "ab_class"]]
print("\nantibody class breakdown:")
print(unique_abs["ab_class"].value_counts())


# get unique viruses with their clade
virus_table = clean.drop_duplicates("virus_id")[["virus_id", "clade"]].copy()

print("\nvirus clade breakdown (top 10):")
print(virus_table["clade"].value_counts().head(10))

# can't stratify a clade that appears fewer than 5 times
clade_counts = virus_table["clade"].value_counts()
rare_clades = clade_counts[clade_counts < 5].index.tolist()
if rare_clades:
    print(f"\nmerging {len(rare_clades)} rare clades into 'rare': {rare_clades[:5]}...")
    virus_table.loc[virus_table["clade"].isin(rare_clades), "clade"] = "rare"

virus_train, virus_test = train_test_split(
    virus_table["virus_id"],
    test_size=0.2,
    stratify=virus_table["clade"],
    random_state=0,
)

print(f"\nvirus split: {len(virus_train)} train / {len(virus_test)} test")

# the clade proportions should be similar in both
train_clade = virus_table[virus_table["virus_id"].isin(virus_train)]["clade"].value_counts(normalize=True)
test_clade = virus_table[virus_table["virus_id"].isin(virus_test)]["clade"].value_counts(normalize=True)
print("\nclade proportions train vs test (should be close):")
print(pd.DataFrame({"train": train_clade, "test": test_clade}).round(3))


# only stratify if every class has >= 5 examples. otherwise stratify fails
min_class_count = unique_abs["ab_class"].value_counts().min()
if min_class_count < 5:
    print(f"\nsome classes have < 5 antibodies (min={min_class_count}), doing unstratified split")
    stratify_col = None
else:
    stratify_col = unique_abs["ab_class"]

ab_train, ab_test = train_test_split(
    unique_abs["antibody_id"],
    test_size=0.2,
    stratify=stratify_col,
    random_state=0,
)

print(f"\nantibody split: {len(ab_train)} train / {len(ab_test)} test")

# check class proportions
train_class = unique_abs[unique_abs["antibody_id"].isin(ab_train)]["ab_class"].value_counts(normalize=True)
test_class = unique_abs[unique_abs["antibody_id"].isin(ab_test)]["ab_class"].value_counts(normalize=True)
print("\nantibody class proportions train vs test:")
print(pd.DataFrame({"train": train_class, "test": test_class}).round(3))

# warn if held-out vrc01_class is tiny
n_test_vrc = (unique_abs[unique_abs["antibody_id"].isin(ab_test)]["ab_class"] == "vrc01_class").sum()
if n_test_vrc < 5:
    print(f"\nWARNING: only {n_test_vrc} vrc01_class antibodies held out.")
    print("surrogate validation on novel vrc01-class may be noisy with this few.")


# four possible combinations:
# train:      train antibody x train virus   <- what the surrogate trains on
# test_ab:    test antibody  x train virus   <- does surrogate generalize to new ab against seen virus?
# test_v:     train antibody x test virus    <- does surrogate generalize to new virus against seen ab?
# test_both:  test antibody  x test virus    <- hardest test, novel ab AND novel virus

train_pairs = clean[
    clean["antibody_id"].isin(ab_train) &
    clean["virus_id"].isin(virus_train)
].copy()

test_ab_only = clean[
    clean["antibody_id"].isin(ab_test) &
    clean["virus_id"].isin(virus_train)
].copy()

test_v_only = clean[
    clean["antibody_id"].isin(ab_train) &
    clean["virus_id"].isin(virus_test)
].copy()

test_both = clean[
    clean["antibody_id"].isin(ab_test) &
    clean["virus_id"].isin(virus_test)
].copy()

print("\nPAIR SPLITS")
print(f"train (train_ab x train_v):       {len(train_pairs)}")
print(f"test_ab_only (test_ab x train_v):  {len(test_ab_only)}")
print(f"test_v_only (train_ab x test_v):   {len(test_v_only)}")
print(f"test_both (test_ab x test_v):      {len(test_both)}")


# no antibody appears in both splits
assert set(ab_train).isdisjoint(set(ab_test)), "antibody ids leak between train and test"

# no virus appears in both splits
assert set(virus_train).isdisjoint(set(virus_test)), "virus ids leak between train and test"

# no (antibody, virus) pair appears in both train and test
train_keys = set(zip(train_pairs["antibody_id"], train_pairs["virus_id"]))
test_keys = set(zip(test_both["antibody_id"], test_both["virus_id"]))
assert train_keys.isdisjoint(test_keys), "same pair appears in train and test_both"

# training set shouldn't be tiny
if len(train_pairs) < 1000:
    print(f"\nWARNING: only {len(train_pairs)} training pairs. the surrogate will struggle.")



# pair tables
train_pairs.to_csv(processed / "train_pairs.csv", index=False)
test_both.to_csv(processed / "test_pairs_both_heldout.csv", index=False)
test_ab_only.to_csv(processed / "test_pairs_antibody_heldout.csv", index=False)
test_v_only.to_csv(processed / "test_pairs_virus_heldout.csv", index=False)

# id lists - these are the ground truth going forward
ab_train.to_frame(name="antibody_id").to_csv(processed / "train_antibody_ids.csv", index=False)
ab_test.to_frame(name="antibody_id").to_csv(processed / "test_antibody_ids.csv", index=False)
virus_train.to_frame(name="virus_id").to_csv(processed / "train_virus_ids.csv", index=False)
virus_test.to_frame(name="virus_id").to_csv(processed / "test_virus_ids.csv", index=False)

# metadata tables for later strat sampling in phase 5 and held-out eval in phase 7
virus_table.to_csv(processed / "virus_table_with_clade.csv", index=False)
unique_abs.to_csv(processed / "antibody_table_with_class.csv", index=False)

print("\nsaved split files to", processed)


# for logbook

print("split summary")
print(f"total clean pairs: {len(clean)}")
print(f"train pairs: {len(train_pairs)} ({len(train_pairs)/len(clean):.1%})")
print(f"held-out pairs (any held): {len(clean) - len(train_pairs)}")
print(f"  both held out:  {len(test_both)}")
print(f"  antibody held:  {len(test_ab_only)}")
print(f"  virus held:     {len(test_v_only)}")
print(f"")
print(f"train antibodies: {len(ab_train)}")
print(f"held-out antibodies: {len(ab_test)}")
print(f"train viruses: {len(virus_train)}")
print(f"held-out viruses: {len(virus_test)}")

#ChatGPT was used to help write the antibody classification function in this script. The function was written based on the VRC01-class antibody patterns and lineage information provided in the comments.