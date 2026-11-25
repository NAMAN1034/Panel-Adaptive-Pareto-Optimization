"""
catnap preprocessing. takes the raw assay + virus files and turns them
into one clean table with antibody_id, virus_id, pIC50, censored, clade, year. also saves a separate table with all the raw observations for reference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.seeding import seed_everything


seed_everything(0)

raw = Path("data/raw")
out = Path("data/interim")
out.mkdir(parents=True, exist_ok=True)


def find_file(stub):
    p = raw / (stub + ".txt")
    if p.exists():
        return p
    raise FileNotFoundError(f"couldnt find {stub}.txt in {raw}")


assay_path = find_file("catnap_neutralization")
virus_path = find_file("catnap_viruses")
print("assay file:", assay_path)
print("virus file:", virus_path)


# catnap files are tab delimited not comma, lowkey almost missed this
assay = pd.read_csv(assay_path, sep="\t", low_memory=False)
viruses = pd.read_csv(virus_path, sep="\t", low_memory=False)

print("\nassay shape:", assay.shape)
print("assay columns:", list(assay.columns))
print("\nvirus shape:", viruses.shape)
print("virus columns:", list(viruses.columns))


# column names in catnap are not super consistent between snapshots so just
# grab the first one that matches instead of hardcoding and praying
def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"couldnt find {label} col. tried {candidates}. actual cols: {list(df.columns)}")


ab_col = pick_col(assay, ["Antibody", "antibody", "ab_name", "antibody_name", "Ab"], "antibody")
virus_col = pick_col(assay, ["Virus", "virus", "virus_name", "virusname"], "virus")
ic50_col = pick_col(assay, ["IC50", "ic50", "IC50_ug_ml", "IC50 (ug/ml)"], "IC50")

print(f"\nusing -> antibody='{ab_col}' virus='{virus_col}' ic50='{ic50_col}'")


# catnap marks a bunch of values as ">10" or "<0.02" instead of a number.
# need to remember which ones were censored before i strip the symbols off
raw_ic50 = assay[ic50_col].astype(str).str.strip()

assay["ic50_original"] = raw_ic50

assay["censor_direction"] = np.select(
    [
        raw_ic50.str.startswith("<"),
        raw_ic50.str.startswith(">")
    ],
    ["left", "right"],
    default="none"
)

assay["is_exact"] = assay["censor_direction"] == "none"

assay["ic50_boundary"] = pd.to_numeric(
    raw_ic50
    .str.replace(r"^[<>]", "", regex=True)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

# remove missing, zero, and negative IC50 values
assay = assay[
    assay["ic50_boundary"].notna() &
    (assay["ic50_boundary"] > 0)
].copy()

# only calculate exact pIC50 values
assay["pIC50"] = np.nan

exact = assay["is_exact"]

assay.loc[exact, "pIC50"] = (
    8.176 - np.log10(assay.loc[exact, "ic50_boundary"])
)

# classify measurements relative to the 10 ug/mL cutoff
assay["cutoff_status"] = "uncertain"

assay.loc[
    exact & (assay["ic50_boundary"] <= 10),
    "cutoff_status"
] = "included_exact"

assay.loc[
    exact & (assay["ic50_boundary"] > 10),
    "cutoff_status"
] = "above_cutoff"

assay.loc[
    (assay["censor_direction"] == "left") &
    (assay["ic50_boundary"] <= 10),
    "cutoff_status"
] = "below_cutoff_censored"

assay.loc[
    (assay["censor_direction"] == "right") &
    (assay["ic50_boundary"] >= 10),
    "cutoff_status"
] = "above_cutoff"

print("\nIC50 BREAKDOWN")
print(assay["censor_direction"].value_counts())

print("\n CUTOFF BREAKDOWN")
print(assay["cutoff_status"].value_counts())

print("\nexact pIC50 range:",
      round(assay.loc[exact, "pIC50"].min(), 2),
      "to",
      round(assay.loc[exact, "pIC50"].max(), 2))

print("exact pIC50 median:",
      round(assay.loc[exact, "pIC50"].median(), 2))


# now pull clade + year from the virus metadata file.
# catnap sometimes calls them clade/subtype and year/collection_year
virus_key = pick_col(viruses, ["---Virus name", "Virus", "virus", "virus_name", "virusname"], "virus key")
subtype_col = pick_col(viruses, ["Subtype", "subtype", "clade"], "subtype")
year_col = pick_col(viruses, ["Year", "year", "collection_year"], "year")

print(f"\nvirus meta cols -> key='{virus_key}' subtype='{subtype_col}' year='{year_col}'")

virus_meta = viruses[[virus_key, subtype_col, year_col]].copy()
virus_meta.columns = ["virus", "subtype", "year"]

# preserve the original subtype labels
virus_meta["subtype"] = virus_meta["subtype"].astype("string").str.strip()
virus_meta["year"] = pd.to_numeric(virus_meta["year"], errors="coerce")

virus_meta = virus_meta.drop_duplicates(subset="virus")

# attach virus metadata to every measurement
observations = assay.merge(
    virus_meta,
    left_on=virus_col,
    right_on="virus",
    how="left"
)

observations = observations.rename(columns={
    ab_col: "antibody_id",
    virus_col: "virus_id"
})

# keep the original reference information too
keep_cols = ["antibody_id","virus_id","IC50","ic50_original","ic50_boundary","censor_direction","is_exact","pIC50","cutoff_status","subtype","year"]

for col in ["Reference", "Pubmed ID"]:
    if col in observations.columns:
        keep_cols.append(col)

observations = observations[keep_cols]

print("\n Observation dataset")
print("shape:", observations.shape)
print("missing subtype:", observations["subtype"].isna().sum())
print("missing year:", observations["year"].isna().sum())

# save every measurement before aggregation
observations.to_csv(
    out / "catnap_observations.csv",
    index=False
)

print("\nsaved observations to", out / "catnap_observations.csv")

# use exact measurements for the initial training dataset
exact_data = observations[observations["is_exact"]].copy()

clean = (
    exact_data
    .groupby(["antibody_id", "virus_id"], as_index=False)
    .agg(
        pIC50=("pIC50", "mean"),
        subtype=("subtype", "first"),
        year=("year", "first"),
        n_exact=("pIC50", "size")
    )
)

print("\nExact only dataset")
print("collapsed", len(exact_data), "measurements into", len(clean), "unique antibody-virus pairs")

clean.to_csv(
    out / "catnap_clean.csv",
    index=False
)

print("\nsaved exact-only dataset to", out / "catnap_clean.csv")

print("\nunique antibodies:", clean["antibody_id"].nunique())
print("unique viruses:", clean["virus_id"].nunique())

print("\nsubtype distribution:")
print(clean.drop_duplicates("virus_id")["subtype"].value_counts().head(20))

print("\nexact pIC50 statistics:")
print(clean["pIC50"].describe())