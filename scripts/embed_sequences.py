"""
compute and cache esm-2 650M embeddings for all unique antibody heavy/light
chains and virus env sequences. first run downloads ~2.5GB model, then embeds
~4500 sequences on MPS (~30-50 min). 
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Bio import SeqIO

from src.utils.seeding import seed_everything

seed_everything(0)

raw = Path("data/raw")
processed = Path("data/processed")
cache_dir = Path("cache/embeddings")
cache_dir.mkdir(parents=True, exist_ok=True)


# which ids do we need to embed
ab_train = pd.read_csv(processed / "train_antibody_ids.csv")["antibody_id"].tolist()
ab_test = pd.read_csv(processed / "test_antibody_ids.csv")["antibody_id"].tolist()
all_abs = sorted(set(ab_train) | set(ab_test))

virus_train = pd.read_csv(processed / "train_virus_ids.csv")["virus_id"].tolist()
virus_test = pd.read_csv(processed / "test_virus_ids.csv")["virus_id"].tolist()
all_viruses = sorted(set(virus_train) | set(virus_test))

print(f"split antibodies: {len(all_abs)}")
print(f"split viruses: {len(all_viruses)}")


# load fasta sequences
def load_fasta(path):
    out = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        out[rec.id] = str(rec.seq)
    return out

print("\nloading fastas...")
heavy = load_fasta(raw / "catnap_antibody_heavy_aa.fasta")
light = load_fasta(raw / "catnap_antibody_light_aa.fasta")
virus = load_fasta(raw / "catnap_virus_aa_alignment.fasta")
print(f"  heavy: {len(heavy)}")
print(f"  light: {len(light)}")
print(f"  virus: {len(virus)}")


# id matching via format-aware strategies
print("ID MATCHING")
print("\nfirst 3 heavy fasta ids:")
for k in list(heavy.keys())[:3]:
    print(f"  {k}")
print("first 3 split antibody ids:")
for k in all_abs[:3]:
    print(f"  {k}")


# fasta id starts with split_id + '_' 
def match_abs(split_ids, fasta_dict):
    out = {}
    for sid in split_ids:
        prefix = sid + "_"
        for fid in fasta_dict:
            if fid == sid or fid.startswith(prefix):
                out[sid] = fid
                break
    return out


ab_heavy_match = match_abs(all_abs, heavy)
ab_light_match = match_abs(all_abs, light)
print(f"\nantibody heavy matches: {len(ab_heavy_match)}/{len(all_abs)} = {len(ab_heavy_match)/len(all_abs):.1%}")
print(f"antibody light matches: {len(ab_light_match)}/{len(all_abs)} = {len(ab_light_match)/len(all_abs):.1%}")

unmatched_abs = [a for a in all_abs if a not in ab_heavy_match][:10]
if unmatched_abs:
    print(f"first 10 unmatched antibodies: {unmatched_abs}")


# by accession number 
print("\nloading catnap_viruses.txt for accession mapping...")
viruses_meta = pd.read_csv(raw / "catnap_viruses.txt", sep="\t")

vid_col = next((c for c in viruses_meta.columns if "virus name" in c.lower()), None)
acc_col = next((c for c in viruses_meta.columns if "accession" in c.lower()), None)
print(f"  virus id column: '{vid_col}'")
print(f"  accession column: '{acc_col}'")

if vid_col is None or acc_col is None:
    print("couldnt find virus id or accession column. aborting.")
    sys.exit(1)

virus_id_to_acc = {}
for _, row in viruses_meta.iterrows():
    vid = str(row[vid_col]).strip()
    acc = str(row[acc_col]).strip()
    if vid and acc and acc.lower() != "nan":
        virus_id_to_acc[vid] = acc

print(f"  mapped {len(virus_id_to_acc)} virus ids to accessions")

# build accession 
acc_to_seq = {}
for fid, seq in virus.items():
    acc = fid.rsplit(".", 1)[-1] if "." in fid else fid
    acc_to_seq[acc] = seq

# match each split virus by accession
virus_seqs = {}
unmatched_viruses = []
for vid in all_viruses:
    acc = virus_id_to_acc.get(vid)
    if acc and acc in acc_to_seq:
        virus_seqs[vid] = acc_to_seq[acc]
    else:
        unmatched_viruses.append(vid)

virus_match_rate = len(virus_seqs) / len(all_viruses)
print(f"\nvirus matches (by accession): {len(virus_seqs)}/{len(all_viruses)} = {virus_match_rate:.1%}")
if unmatched_viruses:
    print(f"first 10 unmatched viruses: {unmatched_viruses[:10]}")


# build seq dicts for embedding 
ab_heavy_seqs = {sid: heavy[fid] for sid, fid in ab_heavy_match.items()}
ab_light_seqs = {sid: light[fid] for sid, fid in ab_light_match.items()}

print(f"\nfinal matched sequences:")
print(f"  antibody heavy: {len(ab_heavy_seqs)}")
print(f"  antibody light: {len(ab_light_seqs)}")
print(f"  virus: {len(virus_seqs)}")

if len(ab_heavy_seqs) / len(all_abs) < 0.5 or virus_match_rate < 0.5:
    print("\nMATCH RATE STILL TOO LOW. aborting.")
    sys.exit(1)


print(f"\nid match looks acceptable. proceeding to embed.")


# step 4: load esm-2 650M
print("\nloading esm-2 650M... (first run downloads ~2.5GB, be patient)")
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval()
batch_converter = alphabet.get_batch_converter()

# try mps, fall back to cpu
try:
    device = torch.device("mps")
    model = model.to(device)
    # quick smoke test
    _ = batch_converter([("test", "MKTAYIAKQRQ")])
    print(f"  model on device: mps (apple silicon gpu)")
except Exception as e:
    print(f"  mps failed ({e}), falling back to cpu")
    device = torch.device("cpu")
    model = model.to(device)


# embedding function
MAX_LEN = 1022  # esm-2 max context length including special tokens

def sanitize_seq(seq):
    safe = set("ACDEFGHIKLMNPQRSTVWYBXZUO.-")
    out = []
    for c in seq:
        if c in safe:
            out.append(c)
        else:
            out.append("X")
    return "".join(out)


def embed_one(seq, seq_id):
    seq = sanitize_seq(seq)

    still_bad = [c for c in seq if c not in "ACDEFGHIKLMNPQRSTVWYBXZUO.-"]
    if still_bad:
        raise ValueError(f"sanitize failed: {set(still_bad)} in {seq_id}")

    if len(seq) > MAX_LEN - 2:
        seq = seq[:MAX_LEN - 2]

    _, _, tokens = batch_converter([(seq_id, seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)

    rep = results["representations"][33][0, 1:len(seq) + 1, :]
    mean_rep = rep.mean(dim=0).cpu().numpy()
    return mean_rep

#cache
def load_cache(path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache, path):
    with open(path, "wb") as f:
        pickle.dump(cache, f)

heavy_cache_path = cache_dir / "antibody_heavy.pkl"
light_cache_path = cache_dir / "antibody_light.pkl"
virus_cache_path = cache_dir / "virus_env.pkl"

heavy_cache = load_cache(heavy_cache_path)
light_cache = load_cache(light_cache_path)
virus_cache = load_cache(virus_cache_path)


def embed_all(seqs_to_embed, cache, name, cache_path):
    todo = [(k, v) for k, v in seqs_to_embed.items() if k not in cache]
    if not todo:
        print(f"  {name}: all {len(seqs_to_embed)} already cached")
        return cache

    print(f"  {name}: {len(cache)} cached, {len(todo)} new to embed")

    start = time.time()
    n_failed = 0
    for i, (sid, seq) in enumerate(tqdm(todo, desc=name, ncols=80)):
        try:
            cache[sid] = embed_one(seq, sid)
        except Exception as e:
            n_failed += 1
            print(f"\n  WARN: {sid} failed ({e.__class__.__name__}: {e})")
            continue
        if (i + 1) % 50 == 0:
            save_cache(cache, cache_path)

    save_cache(cache, cache_path)
    elapsed = time.time() - start
    print(f"  {name}: {len(todo) - n_failed} embedded in {elapsed:.0f}s "
          f"({elapsed/max(len(todo),1):.2f}s per seq), {n_failed} failed")
    return cache



print("EMBEDDING")

print("\nembedding antibody heavy chains...")
heavy_cache = embed_all(ab_heavy_seqs, heavy_cache, "heavy", heavy_cache_path)

print("\nembedding antibody light chains...")
light_cache = embed_all(ab_light_seqs, light_cache, "light", light_cache_path)

print("\nembedding virus env sequences...")
virus_cache = embed_all(virus_seqs, virus_cache, "virus", virus_cache_path)


#summary for logbook
print("CACHE SUMMARY")

for name, c, p in [("heavy", heavy_cache, heavy_cache_path),
                    ("light", light_cache, light_cache_path),
                    ("virus", virus_cache, virus_cache_path)]:
    if len(c) > 0:
        sample_shape = next(iter(c.values())).shape
        print(f"{name}: {len(c)} embeddings  shape={sample_shape}  file={p}")
    else:
        print(f"{name}: EMPTY")

print(f"\ncache dir size: {sum(f.stat().st_size for f in cache_dir.glob('*.pkl')) / 1e6:.1f} MB")
print(f"\nmodule 3.1 done.")