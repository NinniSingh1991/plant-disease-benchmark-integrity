# %% [markdown]
# # Benchmark integrity: does removing duplicate and conflicting images, and balancing the classes, improve the models?
#
# For each dataset: (1) exact audit by 64-bit difference hash with multi-index retrieval (radius 5) and thumbnail
# verification (Pearson correlation >= 0.90 on a 32x32 grey thumbnail); (2) one fixed, clean test set; (3) a grid of
# training conditions: training data {raw, cleaned} x balancing {none, class-weighted loss, oversampling, focal loss},
# three seeds each, identical recipe otherwise.
#
# Training recipe (identical to the HCP-Net runs): 224 px, Adam lr 1e-3 (ViT-S 1e-4), cosine schedule, 15 epochs,
# batch 64, mixed precision, the same augmentation, the final epoch is used (no model selection on the test set).
#
# Kaggle: GPU T4, Internet ON. Inputs: the competitions "cassava-leaf-disease-classification" and/or
# "paddy-disease-classification" (Join the competition once, then Add Input -> Competitions). PlantDoc is cloned
# from GitHub. Choose the datasets in RUN below. Use "Save Version -> Save & Run All (Commit)".
# The last cell prints every result as JSON, so the numbers survive even if downloading fails.

# %%
RUN = ["plantdoc", "paddydoctor"]          # edit: any of "plantdoc", "paddydoctor", "cassava2020", "plantwild", "rice_leaf"
VIT_ON = ["plantdoc"]                      # datasets on which ViT-S also runs the full grid
GRID_ON = None                             # datasets that run the clean/balance grid (None = all of RUN)
LEAK_ON = []                               # datasets that also run the leakage test (naive random split vs copy-free test images)
import os, sys, json, time, glob, subprocess, random, io, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import timm
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("device %s %s" % (DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else ""))
IMG, CACHE_PX, EPOCHS, BATCH, LR, LR_VIT = 224, 256, 15, 64, 1e-3, 1e-4
SEEDS = [42, 43, 44]
RADIUS, VERIFY = 5, 0.90
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE: EPOCHS, SEEDS = 1, [42]
if os.environ.get("RUN"): RUN = os.environ["RUN"].split(",")
if os.environ.get("GRID_ON") is not None: GRID_ON = [x for x in os.environ["GRID_ON"].split(",") if x]
if os.environ.get("LEAK_ON") is not None: LEAK_ON = [x for x in os.environ["LEAK_ON"].split(",") if x]
if GRID_ON is None: GRID_ON = list(RUN)
TMP = Path(os.environ.get("TMP_DIR", "/tmp/benchdata")); TMP.mkdir(parents=True, exist_ok=True)   # datasets live here, not in the output folder
WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working")); OUTD = WORK / "integrity_out"; OUTD.mkdir(parents=True, exist_ok=True)
WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 2)
RESULTS = {"_settings": {"img": IMG, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "lr_vit": LR_VIT, "seeds": SEEDS,
                         "radius": RADIUS, "verify_corr": VERIFY, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"}}
def save(): (OUTD / "integrity_results.json").write_text(json.dumps(RESULTS, indent=1))
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# %% loading: every image is decoded once, resized to 256 x 256 and kept in memory with its signature
def decode(path):
    """path (or raw bytes) -> (256x256x3 uint8, dHash, 32x32 grey thumbnail, md5) or None if unreadable."""
    try:
        data = path if isinstance(path, bytes) else open(path, "rb").read()
        with Image.open(io.BytesIO(data)) as im:
            im.draft("RGB", (CACHE_PX * 2, CACHE_PX * 2))
            rgb = im.convert("RGB"); g = rgb.convert("L")
            a = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
            th = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.uint8)
            arr = np.asarray(rgb.resize((CACHE_PX, CACHE_PX), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return arr, int("".join("1" if b else "0" for b in bits), 2), th.reshape(-1), hashlib.md5(data).hexdigest()

def find_comp(csv_cols):
    for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob(os.environ.get("LOCAL_GLOB", "/nonexistent"), recursive=True):
        head = open(p).readline().strip().split(",")
        if set(csv_cols) <= set(head) and (Path(p).parent / "train_images").exists(): return Path(p).parent
    raise SystemExit("INPUT MISSING: no train.csv with columns %s and a train_images folder. Join the competition and "
                     "add it with Add Input -> Competitions. Inputs present: %s" % (csv_cols, os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "-"))

def list_dataset(name):
    """-> list of (path, label_name, official_split)"""
    import csv
    if name == "plantdoc":
        root = Path(os.environ.get("PLANTDOC_DIR", str(TMP / "PlantDoc-Dataset")))
        if not root.exists():
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/pratikkayal/PlantDoc-Dataset.git", str(root)], check=True)
        out = []
        for split in ("train", "test"):
            for c in sorted(d.name for d in (root / split).iterdir() if d.is_dir()):
                for p in sorted((root / split / c).iterdir()):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}: out.append((str(p), c, split))
        return out
    if name == "cassava2020":
        d = find_comp(["image_id", "label"]); rows = list(csv.DictReader(open(d / "train.csv")))
        names = json.load(open(d / "label_num_to_disease_map.json")) if (d / "label_num_to_disease_map.json").exists() else {}
        return [(str(d / "train_images" / r["image_id"]), names.get(r["label"], r["label"]), "all") for r in rows]
    if name == "paddydoctor":
        d = find_comp(["image_id", "label", "variety"]); rows = list(csv.DictReader(open(d / "train.csv")))
        return [(str(d / "train_images" / r["label"] / r["image_id"]), r["label"], "all") for r in rows]
    if name == "plantwild":                                     # PlantWild v1 (89 classes), Hugging Face mirror
        root = Path(os.environ.get("PLANTWILD_DIR", str(TMP / "plantwild" / "images")))
        if not root.exists():
            from huggingface_hub import hf_hub_download
            import zipfile
            z = hf_hub_download("uqtwei2/PlantWild", "plantwild.zip", repo_type="dataset", local_dir=str(TMP))
            zipfile.ZipFile(z).extractall(TMP); os.remove(z)
        return [(str(p), c.name, "all") for c in sorted(root.iterdir()) if c.is_dir()
                for p in sorted(c.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if name == "rice_leaf":                                     # Rice Leaf Disease (6 classes), AgML parquet mirror
        import pyarrow.parquet as pq
        root = Path(os.environ.get("RICE_DIR", str(TMP / "rice_leaf")))
        if not list(root.rglob("*.parquet")):
            from huggingface_hub import snapshot_download
            snapshot_download("Project-AgML/rice_leaf_disease_classification", repo_type="dataset", local_dir=str(root), max_workers=2)
        out = []
        for f in sorted(root.rglob("*.parquet")):
            t = pq.read_table(f).to_pydict()
            out += [(im["bytes"], "class_%s" % lab, "all") for im, lab in zip(t["image"], t["label"])]
        return out
    raise KeyError(name)

def load_dataset(name):
    items = list_dataset(name)
    if SMOKE: items = items[::25]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex: dec = list(ex.map(decode, [p for p, _, _ in items]))
    keep = [i for i, d in enumerate(dec) if d is not None]
    X = np.stack([dec[i][0] for i in keep]); H = np.array([dec[i][1] for i in keep], dtype=np.uint64)
    TH = np.stack([dec[i][2] for i in keep]); MD5 = [dec[i][3] for i in keep]
    labels = [items[i][1] for i in keep]; split = np.array([items[i][2] for i in keep])
    paths = [items[i][0] if isinstance(items[i][0], str) else "%s#%d" % (name, i) for i in keep]
    classes = sorted(set(labels)); y = np.array([classes.index(l) for l in labels])
    log("%s: %d images, %d classes, %d unreadable" % (name, len(keep), len(classes), len(items) - len(keep)))
    return dict(X=X, H=H, TH=TH, MD5=MD5, y=y, split=split, classes=classes, paths=paths, unreadable=len(items) - len(keep))

# %% audit: exact retrieval within the radius (multi-index hashing), then thumbnail verification
def candidate_pairs(h, r=RADIUS):
    nb = r + 1; widths = [64 // nb + (1 if k < 64 % nb else 0) for k in range(nb)]
    found, shift = set(), 0
    for w in widths:
        band = (h >> np.uint64(shift)) & np.uint64((1 << w) - 1); shift += w
        order = np.argsort(band, kind="stable"); b = band[order]
        for grp in np.split(order, np.flatnonzero(np.diff(b)) + 1):
            if len(grp) < 2: continue
            g = np.sort(grp[:4000]); x = h[g][:, None] ^ h[g][None, :]
            d = np.unpackbits(x.view(np.uint8).reshape(len(g), len(g), 8), axis=2).sum(2)
            ii, jj = np.where(np.triu(d <= r, 1)); found.update(zip(g[ii].tolist(), g[jj].tolist()))
    P = np.array(sorted(found), dtype=np.int64).reshape(-1, 2)
    dist = np.unpackbits((h[P[:, 0]] ^ h[P[:, 1]]).view(np.uint8).reshape(-1, 8), axis=1).sum(1) if len(P) else np.zeros(0, int)
    return P, dist
def thumb_corr(T, P):
    a = T[P[:, 0]].astype(np.float32); b = T[P[:, 1]].astype(np.float32)
    a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1)); den[den == 0] = 1
    return (a * b).sum(1) / den
def components(n, P):
    par = list(range(n))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in P: par[f(i)] = f(j)
    g = defaultdict(list)
    for i in range(n): g[f(i)].append(i)
    return [v for v in g.values() if len(v) > 1]

def audit(D):
    n = len(D["y"]); P, dist = candidate_pairs(D["H"]); corr = thumb_corr(D["TH"], P) if len(P) else np.zeros(0)
    V = P[corr >= VERIFY]
    groups = components(n, V)
    conflict = [g for g in groups if len(set(D["y"][g].tolist())) > 1]
    md5 = defaultdict(list)
    for i, m in enumerate(D["MD5"]): md5[m].append(i)
    byte = [g for g in md5.values() if len(g) > 1]
    a = {"images": n, "classes": len(D["classes"]), "unreadable": D["unreadable"],
         "class_counts": {c: int((D["y"] == k).sum()) for k, c in enumerate(D["classes"])},
         "hash_candidates": int(len(P)), "verified_pairs": int(len(V)), "identical_hash_verified_pairs": int(((dist == 0) & (corr >= VERIFY)).sum()),
         "duplicate_groups": len(groups), "images_in_groups": sum(map(len, groups)),
         "pct_images_in_groups": 100 * sum(map(len, groups)) / n, "redundant_copies": sum(map(len, groups)) - len(groups),
         "conflicting_groups": len(conflict), "images_in_conflicting_groups": sum(map(len, conflict)),
         "byte_identical_groups": len(byte), "byte_identical_conflicting": sum(len(set(D["y"][g].tolist())) > 1 for g in byte)}
    cnt = np.bincount(D["y"], minlength=len(D["classes"])); a["imbalance_ratio"] = float(cnt.max() / max(cnt.min(), 1))
    # exposure of a naive random image-level 80/20 split (the protocol used when no official split exists)
    exp = []
    for s in range(10):
        rng = np.random.default_rng(s); te = np.zeros(n, bool)
        for c in range(len(D["classes"])):
            idx = np.flatnonzero(D["y"] == c); rng.shuffle(idx); te[idx[:int(round(0.2 * len(idx)))]] = True
        exp.append(100 * len({(i if te[i] else j) for i, j in V if te[i] != te[j]}) / te.sum())
    a["naive_split_pct_test_with_train_copy"] = {"mean": float(np.mean(exp)), "min": float(np.min(exp)), "max": float(np.max(exp))}
    return a, groups, conflict

# %% fixed clean test set and the two training sets
def make_split(D, groups, conflict, name):
    n = len(D["y"]); gid = np.arange(n)
    for k, g in enumerate(groups): gid[g] = n + k                 # one id per duplicate group
    in_conflict = np.zeros(n, bool)
    for g in conflict: in_conflict[g] = True
    if (D["split"] == "test").any():                               # official split (PlantDoc)
        is_te = D["split"] == "test"
        tr_gids = set(gid[~is_te].tolist())
        drop_te = is_te & (np.isin(gid, list(tr_gids)) | in_conflict)   # test copies of train images, conflicting labels
        test = np.flatnonzero(is_te & ~drop_te); train = np.flatnonzero(~is_te)
        note = "official split; %d test images removed (copy of a training image or conflicting label)" % int(drop_te.sum())
    else:                                                           # group-aware stratified 80/20 split, seed 0
        rng = np.random.default_rng(0); is_te = np.zeros(n, bool); members = defaultdict(list)
        for i, g in enumerate(gid.tolist()): members[g].append(i)
        ug = sorted(members)
        g_label = {g: int(np.bincount(D["y"][members[g]]).argmax()) for g in ug}
        g_conf = {g: bool(in_conflict[members[g]].any()) for g in ug}
        for c in range(len(D["classes"])):
            cand = [g for g in ug if g_label[g] == c and not g_conf[g]]; rng.shuffle(cand)
            need = 0.2 * (D["y"] == c).sum(); tot = 0
            for g in cand:
                if tot >= need: break
                is_te[members[g]] = True; tot += len(members[g])
        test = np.flatnonzero(is_te); train = np.flatnonzero(~is_te)
        note = "group-aware stratified 80/20 split (seed 0); conflicting groups kept out of the test set"
    # cleaned training set: drop conflicting groups entirely, keep one image of every other duplicate group
    keep = np.ones(n, bool); keep[in_conflict] = False
    for g in groups:
        if not in_conflict[g[0]]:
            for i in sorted(g)[1:]: keep[i] = False
    train_clean = train[keep[train]]
    s = {"note": note, "test": int(len(test)), "train_raw": int(len(train)), "train_clean": int(len(train_clean)),
         "removed_from_train": int(len(train) - len(train_clean)),
         "test_classes_present": int(len(np.unique(D["y"][test])))}
    log("%s split: %s" % (name, s))
    return test, train, train_clean, s

# %% training
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([transforms.RandomResizedCrop(IMG, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.3, 0.3, 0.3, 0.05), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
                               transforms.RandomErasing(p=0.25)])
eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
class ArrDS(Dataset):
    def __init__(self, X, y, idx, tf): self.X, self.y, self.idx, self.tf = X, y, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]; return self.tf(Image.fromarray(self.X[i])), int(self.y[i])

def focal_loss(logits, y, gamma=2.0):
    logp = F.log_softmax(logits.float(), 1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** gamma * logp).mean()

def train_eval(D, tr, te, arch, balance, seed):
    seed_all(seed); nc = len(D["classes"]); ytr = D["y"][tr]
    cnt = np.bincount(ytr, minlength=nc).astype(np.float64)
    w = np.where(cnt > 0, len(ytr) / (nc * np.maximum(cnt, 1)), 0.0)          # inverse-frequency class weights
    if balance == "oversample":
        sampler = WeightedRandomSampler(torch.as_tensor(w[ytr], dtype=torch.double), num_samples=len(tr), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
        dl = DataLoader(ArrDS(D["X"], D["y"], tr, train_tf), batch_size=BATCH, sampler=sampler, num_workers=WORKERS, pin_memory=True, drop_last=True)
    else:
        dl = DataLoader(ArrDS(D["X"], D["y"], tr, train_tf), batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    wt = torch.as_tensor(w, dtype=torch.float32, device=DEV)
    model = timm.create_model(arch, pretrained=True, num_classes=nc).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR_VIT if "vit" in arch else LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    for ep in range(EPOCHS):
        model.train(); tot = nb = 0
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                out = model(x)
            if balance == "weighted": loss = F.cross_entropy(out.float(), y, weight=wt)
            elif balance == "focal": loss = focal_loss(out, y)
            else: loss = F.cross_entropy(out.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item(); nb += 1
        sched.step()
    model.eval(); P = []
    with torch.no_grad():
        for x, _ in DataLoader(ArrDS(D["X"], D["y"], te, eval_tf), batch_size=128, num_workers=WORKERS):
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                P.append(torch.softmax(model(x.to(DEV)).float(), 1).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(P), tot / max(nb, 1)

def metrics(y, P, small):
    from sklearn.metrics import f1_score, balanced_accuracy_score
    pred = P.argmax(1); conf = P.max(1); ok = pred == y
    e = 0.0; edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    rec = {int(c): float(ok[y == c].mean()) for c in np.unique(y)}
    sm = [rec[c] for c in small if c in rec]
    return {"accuracy": 100 * float(ok.mean()), "macro_f1": 100 * float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_accuracy": 100 * float(balanced_accuracy_score(y, pred)), "ece": float(e),
            "minority_recall": 100 * float(np.mean(sm)) if sm else None}

# %% leakage test: the usual image-level random 80/20 split, scored on all test images, on the test images that have a
# verified copy in training, and on the copy-free test images
def leak_test(D, groups, name):
    n = len(D["y"]); rng = np.random.default_rng(0); te = np.zeros(n, bool)
    for c in range(len(D["classes"])):
        idx = np.flatnonzero(D["y"] == c); rng.shuffle(idx); te[idx[:int(round(0.2 * len(idx)))]] = True
    gid = np.arange(n)
    for k, g in enumerate(groups): gid[g] = n + k
    tr_g = set(gid[~te].tolist()); test = np.flatnonzero(te); train = np.flatnonzero(~te)
    copied = np.array([gid[i] in tr_g and gid[i] >= n for i in test])
    np.save(OUTD / ("%s_leak_split.npy" % name), np.stack([test, copied.astype(np.int64)]))
    res = {"test": int(len(test)), "test_with_train_copy": int(copied.sum()), "pct": 100 * float(copied.mean()), "runs": []}
    yte = D["y"][test]
    for seed in SEEDS:
        t = time.time(); P, _ = train_eval(D, train, test, "efficientnet_b0", "none", seed); ok = P.argmax(1) == yte
        r = {"seed": seed, "acc_all": 100 * float(ok.mean()), "acc_copied": 100 * float(ok[copied].mean()) if copied.any() else None,
             "acc_copy_free": 100 * float(ok[~copied].mean()), "minutes": (time.time() - t) / 60}
        res["runs"].append(r); np.save(OUTD / ("%s_leak_efficientnet_b0_%d.npy" % (name, seed)), P.astype(np.float16))
        log("%s LEAK seed %d: all %.2f  copied %s  copy-free %.2f" % (name, seed, r["acc_all"],
            "%.2f" % r["acc_copied"] if r["acc_copied"] is not None else "-", r["acc_copy_free"]))
    return res

# %% main loop
GRID = [(data, bal) for data in ("raw", "cleaned") for bal in ("none", "weighted", "oversample", "focal")]
for name in RUN:
    log("=" * 20 + " " + name)
    D = load_dataset(name)
    a, groups, conflict = audit(D); log("%s audit: %s" % (name, {k: v for k, v in a.items() if k != "class_counts"}))
    test, train, train_clean, s = make_split(D, groups, conflict, name)
    np.savez_compressed(OUTD / ("%s_audit.npz" % name), H=D["H"], TH=D["TH"], y=D["y"], test=test, train=train, train_clean=train_clean)
    (OUTD / ("%s_paths.json" % name)).write_text(json.dumps({"paths": D["paths"], "classes": D["classes"], "groups": groups, "conflict": conflict}))
    cnt = np.bincount(D["y"][train], minlength=len(D["classes"]))
    small = [int(c) for c in np.argsort(cnt)[: max(1, len(cnt) // 3)]]          # smallest third of the classes
    RESULTS[name] = {"audit": a, "split": s, "minority_classes": [D["classes"][c] for c in small], "runs": []}; save()
    yte = D["y"][test]
    if os.environ.get("AUDIT_ONLY") == "1": continue
    if name in LEAK_ON:
        RESULTS[name]["leak"] = leak_test(D, groups, name); save()
    if name not in GRID_ON:
        np.save(OUTD / ("%s_test_labels.npy" % name), yte); del D; continue
    for arch in ["efficientnet_b0"] + (["vit_small_patch16_224"] if name in VIT_ON else []):
        for data, bal in GRID:
            for seed in SEEDS:
                tr = train if data == "raw" else train_clean
                t = time.time(); P, last_loss = train_eval(D, tr, test, arch, bal, seed)
                m = metrics(yte, P, small); m.update({"arch": arch, "train_data": data, "balance": bal, "seed": seed,
                                                      "n_train": int(len(tr)), "final_loss": last_loss, "minutes": (time.time() - t) / 60})
                RESULTS[name]["runs"].append(m); save()
                np.save(OUTD / ("%s_%s_%s_%s_%d.npy" % (name, arch, data, bal, seed)), P.astype(np.float16))
                log("%s %s %-7s %-10s seed %d: acc %.2f  macroF1 %.2f  minority %.2f  (%.1f min)" % (
                    name, arch, data, bal, seed, m["accuracy"], m["macro_f1"], m["minority_recall"] or -1, m["minutes"]))
    np.save(OUTD / ("%s_test_labels.npy" % name), yte)
    del D

# %% print everything (survives a failed download)
save(); log("finished")
print("=" * 30, "COPY EVERYTHING BELOW", "=" * 30)
print(json.dumps(RESULTS, indent=1))
