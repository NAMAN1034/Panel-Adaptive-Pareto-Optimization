"""
lanl env preprocessing. parses the gp160 alignment, keeps subtypes B and C,
extracts the vrc01 epitope region (hxb2 425-510), drops gappy sequences,
and builds a position weight matrix for each subtype.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from Bio import AlignIO

from src.utils.seeding import seed_everything


seed_everything(0)

raw = Path("data/raw")
out = Path("data/interim")
out.mkdir(parents=True, exist_ok=True)


# find the alignment file. i saved it as .fasta yesterday
aln_path = raw / "lanl_gp160_alignment.fasta"
if not aln_path.exists():
    raise FileNotFoundError(f"couldnt find {aln_path}. did you download the lanl alignment?")

print("loading alignment... (might take a sec if its big)")
aln = AlignIO.read(aln_path, "fasta")
print("alignment shape:", len(aln), "sequences x", aln.get_alignment_length(), "columns")


# find hxb2 in the alignment so i can map column -> hxb2 position

# the alignment is supposed to be aligned to hxb2 but i dont trust that without checking. look for a sequence with hxb2 in the id
hxb2_seq = None
hxb2_idx = None

for i, rec in enumerate(aln):
    name = rec.id.upper()
    if "HXB2" in name or "K03455" in name:
        hxb2_seq = str(rec.seq)
        hxb2_idx = i
        print(f"found hxb2 reference: {rec.id} (row {i})")
        break

if hxb2_seq is None:
    raise ValueError(
        "no hxb2 sequence found in alignment. "
        "either download the alignment with hxb2 included, or "
        "manually add the hxb2 gp160 sequence before running this."
    )

# map alignment column -> hxb2 residue number. hxb2 has gaps in the alignment too, so i only count non-gap positions
hxb2_map = {}  # col_index -> hxb2_number
hxb2_pos = 0

for col in range(len(hxb2_seq)):
    aa = hxb2_seq[col]
    if aa != "-":
        hxb2_pos += 1
        hxb2_map[col] = hxb2_pos

print(f"hxb2 ungapped length: {hxb2_pos} residues")

if not (800 <= hxb2_pos <= 900):
    print(f"WARNING: hxb2 ungapped length is {hxb2_pos}, expected around 856. check the alignment.")


# figure out which columns correspond to hxb2 425-510
epitope_cols = [col for col, pos in hxb2_map.items() if 425 <= pos <= 510]
epitope_cols.sort()

print(f"epitope columns (hxb2 425-510): {len(epitope_cols)} alignment columns")
print(f"first few: {epitope_cols[:5]} ... last few: {epitope_cols[-5:]}")


# filter to subtypes B and C

# lanl fasta headers usually look like: >B.US.1998.SOMENAME or >C.ZA.2003 so the subtype is the first letter after the >
def get_subtype(seq_id):
    first = seq_id.split(".")[0] if "." in seq_id else seq_id
    # sometimes its like "B" or "B_123" so just grab the first alpha char
    for ch in first:
        if ch.isalpha():
            return ch.upper()
    return "?"

subtypes = [get_subtype(rec.id) for rec in aln]
subtype_counts = pd.Series(subtypes).value_counts()
print("\nsubtype counts (top 15):")
print(subtype_counts.head(15))

# keep only B and C. proposal says theyre the majority of infections
keep_idx = [i for i, st in enumerate(subtypes) if st in ("B", "C")]
print(f"\nkeeping {len(keep_idx)} / {len(aln)} sequences (B and C only)")

if len(keep_idx) == 0:
    raise ValueError("no B or C sequences found. check the header format.")


# extract the epitope region and filter gappy sequences

records = []  # list of dicts with id, subtype, epitope_seq

for i in keep_idx:
    rec = aln[i]
    full = str(rec.seq)
    epi = "".join(full[col] for col in epitope_cols)

    # count gaps in the epitope region
    n_gaps = epi.count("-")
    gap_frac = n_gaps / len(epi) if len(epi) > 0 else 1.0

    # also drop sequences with stop codons or weird chars
    bad_chars = set("X*BZJ")
    has_bad = any(ch in bad_chars for ch in epi)

    records.append({
        "seq_id": rec.id,
        "subtype": get_subtype(rec.id),
        "epitope_seq": epi,
        "gap_frac": gap_frac,
        "has_bad": has_bad,
    })

epi_df = pd.DataFrame(records)
print(f"\nepitope extracted for {len(epi_df)} sequences")
print(f"gap fraction stats:")
print(epi_df["gap_frac"].describe())

# remove sequences with more than 5% gaps at epitope positions
before = len(epi_df)
epi_df = epi_df[epi_df["gap_frac"] <= 0.05].copy()
epi_df = epi_df[~epi_df["has_bad"]].copy()

print(f"\nafter gap + bad char filter: {len(epi_df)} / {before} sequences kept")
print("subtype breakdown after filter:")
print(epi_df["subtype"].value_counts())


if len(epi_df) == 0:
    raise ValueError(
        "all sequences got filtered out. the 5% gap cutoff might be too strict "
        "for this alignment. check the gap_frac describe() output above."
    )

# save the filtered epitope alignment so i can eyeball it later
fasta_out = out / "lanl_epitope_filtered.fasta"
with open(fasta_out, "w") as f:
    for _, row in epi_df.iterrows():
        f.write(f">{row['seq_id']}|{row['subtype']}\n{row['epitope_seq']}\n")

print(f"\nsaved filtered epitope alignment to {fasta_out}")


# build a pwm for each subtype

# standard 20 amino acids. order doesnt matter as long as its consistent
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# gapped positions become nan in the pwm. theyre not informative

def build_pwm(seqs):
    if len(seqs) == 0:
        return None

    L = len(seqs[0])
    # counts[pos][aa] = number of times aa appears at pos
    counts = np.zeros((L, len(AA_ORDER)), dtype=int)
    aa_to_idx = {aa: j for j, aa in enumerate(AA_ORDER)}

    for seq in seqs:
        if len(seq) != L:
            raise ValueError("sequences in a pwm group must all be the same length")
        for i, aa in enumerate(seq):
            if aa in aa_to_idx:  # skips gaps and anything weird
                counts[i, aa_to_idx[aa]] += 1

    # convert counts to frequencies. if a position is all gaps, freq = 0 everywhere
    row_sums = counts.sum(axis=1, keepdims=True)
    # avoid divide by zero. positions with 0 observations get all zeros
    freqs = np.divide(
        counts, row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums != 0
    )

    pwm = pd.DataFrame(freqs, columns=AA_ORDER)
    pwm.index = range(1, L + 1)  # 1-indexed positions within the epitope
    pwm.index.name = "epitope_pos"
    return pwm


# build one pwm per subtype
for st in ["B", "C"]:
    sub_seqs = epi_df[epi_df["subtype"] == st]["epitope_seq"].tolist()
    print(f"\nbuilding pwm for subtype {st} ({len(sub_seqs)} sequences)")

    if len(sub_seqs) < 5:
        print(f"WARNING: only {len(sub_seqs)} sequences for subtype {st}. pwm will be noisy.")

    pwm = build_pwm(sub_seqs)
    if pwm is not None:
        pwm_path = out / f"lanl_pwm_subtype_{st}.csv"
        pwm.to_csv(pwm_path)
        print(f"saved pwm to {pwm_path}  (shape: {pwm.shape})")

        # quick eyeball
        top_positions = [1, 10, 20, 40, 60, 86]  # arbitrary spots in the 86-aa region
        print("  most common aa at some positions:")
        for pos in top_positions:
            if pos in pwm.index:
                row = pwm.loc[pos]
                top_aa = row.idxmax()
                top_freq = row.max()
                print(f"    pos {pos}: {top_aa} ({top_freq:.2f})")


# save combined metadata table
meta_out = out / "lanl_epitope_metadata.csv"
epi_df[["seq_id", "subtype", "gap_frac"]].to_csv(meta_out, index=False)
print(f"\nsaved epitope metadata to {meta_out}")

print("\nmodule 2.4 done.")
print(f"epitope length per sequence: {len(epi_df.iloc[0]['epitope_seq'])} aa")
print(f"subtypes kept: {epi_df['subtype'].value_counts().to_dict()}")