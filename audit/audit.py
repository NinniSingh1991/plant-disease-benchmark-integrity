"""
Integrity audit of public plant-disease benchmarks: exact and near duplicates, label conflicts,
split crossings and cross-benchmark overlap.

Every image is reduced to a 64-bit difference hash (dHash) of its 9x8 greyscale thumbnail, the same
hash as the HCP-Net partition audit. Pairs within Hamming distance R are retrieved exactly by
multi-index hashing: the code is cut into R+1 bands and, by the pigeonhole principle, any pair within
distance R agrees on at least one band. Each candidate pair is then verified on a 32x32 greyscale
thumbnail (Pearson correlation), which removes hash collisions between unrelated images.

    python audit.py hash  <dataset> ...      # hash and cache one or more datasets
    python audit.py report                    # audit every cached dataset and all dataset pairs
"""
import os
import hashlib, io, json, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
DATA, CACHE, OUT = ROOT / "data", ROOT / "cache", ROOT / "results"
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
R = 5                     # Hamming radius of a near duplicate (as in the HCP-Net audit)
VERIFY = 0.90             # minimum thumbnail correlation for a hash match to count
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------------------------------------------------ image signature
def signature(data):
    """bytes -> (dhash, 32x32 thumbnail, md5 of the file bytes); None if the image is unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.draft("L", (256, 256))                  # fast JPEG decode at reduced size
            g = im.convert("L")
            a = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
            t = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = int("".join("1" if b else "0" for b in bits), 2)
    return h, t.tobytes(), hashlib.md5(data).hexdigest()

def sig_file(p):
    with open(p, "rb") as f: return signature(f.read())


# ------------------------------------------------------------------ dataset loaders
# Each loader returns a list of records {id, label, split} and a parallel list of image sources
# (file paths, or raw bytes for parquet-packed copies). Labels are the dataset's own class names.
def folder_items(root, split_from_parent=None):
    recs, srcs = [], []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in EXT: continue
        rel = p.relative_to(root).parts
        if len(rel) < (3 if split_from_parent else 2): continue      # files outside class folders
        split = rel[0] if split_from_parent else "all"
        label = rel[1] if split_from_parent else rel[0]
        recs.append({"id": "/".join(rel), "label": label, "split": split}); srcs.append(str(p))
    return recs, srcs

def parquet_items(root, label_col="label", split_from_name=True):
    import pyarrow.parquet as pq
    recs, srcs = [], []
    for f in sorted(Path(root).rglob("*.parquet")):
        split = f.stem.split("-")[0] if split_from_name else "all"
        t = pq.ParquetFile(f)
        names = None
        feat = t.schema_arrow.field(label_col).type if label_col in t.schema_arrow.names else None
        for rg in range(t.num_row_groups):
            tb = t.read_row_group(rg).to_pydict()
            for k in range(len(tb["image"])):
                lab = tb[label_col][k]
                if isinstance(lab, list): lab = "+".join(sorted(str(x) for x in lab)) or "none"
                recs.append({"id": "%s#%d#%d" % (f.name, rg, k), "label": str(lab), "split": split})
                srcs.append(tb["image"][k]["bytes"])
    return recs, srcs

def load(name):
    d = DATA / name
    if name == "plantvillage":
        return folder_items(Path(os.environ.get("PLANTVILLAGE_DIR", str(DATA / "plantvillage" / "color"))))
    if name == "plantdoc":
        return folder_items(DATA / "PlantDoc-Dataset", split_from_parent=True)
    if name == "cassava2020":
        import csv
        rows = list(csv.DictReader(open(d / "train.csv")))
        recs = [{"id": r["image_id"], "label": r["label"], "split": "train"} for r in rows]
        return recs, [str(d / "train_images" / r["image_id"]) for r in rows]
    if name in ("paddydoctor", "rice_leaf"):
        return parquet_items(d)
    if name == "plantpathology2021":
        return parquet_items(d, label_col="labels")
    if name == "corn_maize":
        return parquet_items(d)          # Project-AgML packing of the Kaggle "Corn or Maize Leaf Disease" set (4,188 images)
    if name == "plantwild":
        return folder_items(next(p for p in d.rglob("*") if p.is_dir() and len([c for c in p.iterdir() if c.is_dir()]) > 50))
    raise KeyError(name)


# ------------------------------------------------------------------ hashing with cache
def hash_parquet_streaming(name, label_col, workers=10):
    """Large parquet copies (Plant Pathology 2021): hash one row group at a time to bound memory."""
    import pyarrow.parquet as pq
    recs, sigs = [], []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in sorted((DATA / name).rglob("*.parquet")):
            t = pq.ParquetFile(f); split = f.stem.split("-")[0]
            for rg in range(t.num_row_groups):
                tb = t.read_row_group(rg, columns=["image", label_col]).to_pydict()
                sigs += list(ex.map(signature, [x["bytes"] for x in tb["image"]], chunksize=4))
                for k, lab in enumerate(tb[label_col]):
                    if isinstance(lab, list): lab = "+".join(sorted(str(x) for x in lab)) or "none"
                    recs.append({"id": "%s#%d#%d" % (f.name, rg, k), "label": str(lab), "split": split})
            print("  %s done (%d images, %.0f s)" % (f.name, len(recs), time.time() - t0), flush=True)
    return recs, sigs

def hash_dataset(name, workers=10):
    if name == "plantpathology2021":
        recs, sigs = hash_parquet_streaming(name, "labels", workers); t0 = time.time()
        keep = [i for i, s in enumerate(sigs) if s is not None]; bad = len(sigs) - len(keep)
        np.save(CACHE / (name + "_hash.npy"), np.array([sigs[i][0] for i in keep], dtype=np.uint64))
        np.save(CACHE / (name + "_thumb.npy"), np.frombuffer(b"".join(sigs[i][1] for i in keep), np.uint8).reshape(-1, 1024))
        (CACHE / (name + "_meta.json")).write_text(json.dumps({"records": [dict(recs[i], md5=sigs[i][2]) for i in keep], "unreadable": bad}))
        print("%s: %d images hashed, %d unreadable" % (name, len(keep), bad), flush=True); return
    recs, srcs = load(name)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        if srcs and isinstance(srcs[0], str): sigs = list(ex.map(sig_file, srcs, chunksize=64))
        else: sigs = list(ex.map(signature, srcs, chunksize=16))
    keep = [i for i, s in enumerate(sigs) if s is not None]
    bad = len(sigs) - len(keep)
    np.save(CACHE / (name + "_hash.npy"), np.array([sigs[i][0] for i in keep], dtype=np.uint64))
    np.save(CACHE / (name + "_thumb.npy"), np.frombuffer(b"".join(sigs[i][1] for i in keep), np.uint8).reshape(-1, 1024))
    meta = [dict(recs[i], md5=sigs[i][2]) for i in keep]
    (CACHE / (name + "_meta.json")).write_text(json.dumps({"records": meta, "unreadable": bad}))
    print("%s: %d images hashed, %d unreadable, %.0f s" % (name, len(keep), bad, time.time() - t0), flush=True)


# ------------------------------------------------------------------ matching
def popcount(x):
    x = x.astype(np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)

def candidate_pairs(h, r=R):
    """All index pairs (i<j) with Hamming distance <= r, exactly, by multi-index hashing."""
    nb = r + 1; widths = [64 // nb + (1 if k < 64 % nb else 0) for k in range(nb)]
    found, shift = set(), 0
    for w in widths:
        band = (h >> np.uint64(shift)) & np.uint64((1 << w) - 1); shift += w
        order = np.argsort(band, kind="stable"); b = band[order]
        cut = np.flatnonzero(np.diff(b)) + 1
        for grp in np.split(order, cut):
            if len(grp) < 2: continue
            if len(grp) > 4000: grp = grp[:4000]           # degenerate band value; guarded below
            g = np.sort(grp)
            x = h[g][:, None] ^ h[g][None, :]
            d = np.unpackbits(x.view(np.uint8).reshape(len(g), len(g), 8), axis=2).sum(2)
            ii, jj = np.where(np.triu(d <= r, 1))
            found.update(zip(g[ii].tolist(), g[jj].tolist()))
    if not found: return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    P = np.array(sorted(found), dtype=np.int64)
    dist = popcount(h[P[:, 0]] ^ h[P[:, 1]])
    return P, dist

def thumb_corr(T, P):
    a = T[P[:, 0]].astype(np.float32); b = T[P[:, 1]].astype(np.float32)
    a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1)); den[den == 0] = 1
    return (a * b).sum(1) / den

def clusters(n, P):
    par = np.arange(n)
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in P:
        a, b = f(i), f(j)
        if a != b: par[a] = b
    groups = defaultdict(list)
    for i in set(P.flatten().tolist()): groups[f(i)].append(i)
    return list(groups.values())


# ------------------------------------------------------------------ report
def load_cache(name):
    m = json.loads((CACHE / (name + "_meta.json")).read_text())
    return np.load(CACHE / (name + "_hash.npy")), np.load(CACHE / (name + "_thumb.npy")), m["records"], m["unreadable"]

def summarise(recs, P, dist, corr, test_split="test"):
    n = len(recs)
    out = {}
    for tag, sel in (("identical_hash", (dist == 0) & (corr >= VERIFY)), ("near_r5", corr >= VERIFY)):
        Q = P[sel]
        cl = clusters(n, Q)
        inv = sum(len(c) for c in cl)
        conflict = [c for c in cl if len({recs[i]["label"] for i in c}) > 1]
        o = {"pairs": int(len(Q)), "clusters": len(cl), "images_in_clusters": inv,
             "pct_images_in_clusters": 100 * inv / n,
             "redundant_images": inv - len(cl), "pct_redundant": 100 * (inv - len(cl)) / n,
             "conflicting_clusters": len(conflict), "images_in_conflicting_clusters": sum(len(c) for c in conflict),
             "pct_clusters_conflicting": (100 * len(conflict) / len(cl)) if cl else 0.0}
        splits = sorted({r["split"] for r in recs})
        if test_split in splits and len(splits) > 1:
            te = [i for i in range(n) if recs[i]["split"] == test_split]
            tr_set = {i for i in range(n) if recs[i]["split"] != test_split}
            nb = defaultdict(set)
            for i, j in Q: nb[i].add(j); nb[j].add(i)
            hit = [i for i in te if nb[i] & tr_set]
            bad = [i for i in hit if recs[i]["label"] not in {recs[j]["label"] for j in nb[i] & tr_set}]
            o.update({"test_images": len(te), "test_with_train_match": len(hit), "pct_test_with_train_match": 100 * len(hit) / len(te),
                      "test_match_with_different_label": len(bad)})
        out[tag] = o
    md5 = defaultdict(list)
    for i, r in enumerate(recs): md5[r["md5"]].append(i)
    byte = [g for g in md5.values() if len(g) > 1]
    out["byte_identical_files"] = {"groups": len(byte), "images": sum(len(g) for g in byte),
                                   "conflicting_groups": sum(len({recs[i]["label"] for i in g}) > 1 for g in byte)}
    return out

def simulated_split_exposure(recs, P, dist, corr, test_frac=0.2, seeds=range(10)):
    """Share of test images with a train duplicate under a random stratified image-level split,
    the protocol most papers use when a benchmark ships without an official split."""
    n = len(recs); Q = P[corr >= VERIFY]; Q0 = P[(dist == 0) & (corr >= VERIFY)]
    labels = np.array([r["label"] for r in recs])
    res = {"near_r5": [], "identical_hash": [], "conflicting_r5": []}
    for s in seeds:
        rng = np.random.default_rng(s); is_te = np.zeros(n, bool)
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c); rng.shuffle(idx)
            is_te[idx[:int(round(test_frac * len(idx)))]] = True
        for tag, QQ in (("near_r5", Q), ("identical_hash", Q0)):
            hit = set()
            for i, j in QQ:
                if is_te[i] != is_te[j]: hit.add(i if is_te[i] else j)
            res[tag].append(100 * len(hit) / is_te.sum())
        bad = set()
        for i, j in Q:
            if is_te[i] != is_te[j] and labels[i] != labels[j]: bad.add(i if is_te[i] else j)
        res["conflicting_r5"].append(100 * len(bad) / is_te.sum())
    return {k: {"mean_pct_test": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in res.items()}

def report():
    names = sorted(p.name[:-10] for p in CACHE.glob("*_meta.json"))
    H, T, REC, UNR, OFF = [], [], [], {}, {}
    out = {"_settings": {"radius": R, "verify_corr": VERIFY, "hash": "dHash 64-bit (9x8)", "thumb": "32x32 grey"}, "datasets": {}, "cross": {}}
    for nm in names:
        h, t, rec, unr = load_cache(nm)
        P, dist = candidate_pairs(h)
        corr = thumb_corr(t, P) if len(P) else np.zeros(0)
        info = {"images": len(rec), "unreadable": unr, "classes": len({r["label"] for r in rec}),
                "splits": dict(sorted({s: sum(r["split"] == s for r in rec) for s in {r["split"] for r in rec}}.items())),
                "hash_matches_rejected_by_verification": int((corr < VERIFY).sum()),
                "degenerate_hashes": int(((h == 0) | (h == np.uint64(2**64 - 1))).sum())}
        info.update(summarise(rec, P, dist, corr))
        if len(info["splits"]) == 1 or "test" not in info["splits"]:
            info["random_split_80_20"] = simulated_split_exposure(rec, P, dist, corr)
        out["datasets"][nm] = info
        np.save(OUT / (nm + "_pairs.npy"), np.c_[P, dist.astype(np.int64), (corr * 1000).astype(np.int64)] if len(P) else np.zeros((0, 4), np.int64))
        print(nm, json.dumps({k: info[k] for k in ("images", "classes")}), "near:", json.dumps(info["near_r5"])[:400], flush=True)
        OFF[nm] = sum(len(x) for x in REC); H.append(h); T.append(t); REC.extend(dict(r, ds=nm) for r in rec)
    h = np.concatenate(H); t = np.concatenate(T)
    P, dist = candidate_pairs(h); corr = thumb_corr(t, P)
    ds = np.array([r["ds"] for r in REC])
    x = ds[P[:, 0]] != ds[P[:, 1]]
    P, dist, corr = P[x], dist[x], corr[x]
    ok = corr >= VERIFY
    for a in names:
        for b in names:
            if a >= b: continue
            s = ok & (((ds[P[:, 0]] == a) & (ds[P[:, 1]] == b)) | ((ds[P[:, 0]] == b) & (ds[P[:, 1]] == a)))
            if s.any():
                Q = P[s]
                ia = {i for i in Q.flatten().tolist() if ds[i] == a}; ib = {i for i in Q.flatten().tolist() if ds[i] == b}
                out["cross"]["%s|%s" % (a, b)] = {"pairs": int(s.sum()), "identical_hash_pairs": int((s & (dist == 0)).sum()),
                                                  "images_%s" % a: len(ia), "images_%s" % b: len(ib)}
    np.save(OUT / "cross_pairs.npy", np.c_[P[ok], dist[ok]])
    (OUT / "cross_records.json").write_text(json.dumps([{"ds": r["ds"], "id": r["id"], "label": r["label"]} for r in REC]))
    (OUT / "audit_results.json").write_text(json.dumps(out, indent=1))
    print("cross:", json.dumps(out["cross"], indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "hash":
        for nm in sys.argv[2:]: hash_dataset(nm)
    else:
        report()
