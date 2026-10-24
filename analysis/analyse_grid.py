# Analysis of the clean-and-balance grid from the saved Kaggle outputs.
# Usage: python analyse_grid.py out_A out_B ...   (each folder = a downloaded integrity_out)
# Every metric is recomputed from the saved probabilities, so the JSON log is not required.
# Writes grid_summary.json and grid_tables.md next to this script.
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from collections import defaultdict
import numpy as np

BAL = ["none", "weighted", "oversample", "focal"]
DATA = ["raw", "cleaned"]
BOOT = 2000

def ece(P, y, bins=15):
    conf = P.max(1); ok = P.argmax(1) == y; e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e

def per_class_recall(pred, y, nc):
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(nc)])

def macro_f1(pred, y, nc):
    f = []
    for c in range(nc):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        if tp + fp + fn: f.append(2 * tp / (2 * tp + fp + fn))
    return 100 * float(np.mean(f))

def metrics(P, y, nc, small):
    pred = P.argmax(1); rec = per_class_recall(pred, y, nc)
    sm = [rec[c] for c in small if not np.isnan(rec[c])]
    return {"accuracy": 100 * float((pred == y).mean()), "macro_f1": macro_f1(pred, y, nc),
            "balanced_accuracy": 100 * float(np.nanmean(rec)), "minority_recall": 100 * float(np.mean(sm)), "ece": float(ece(P, y))}

def mcnemar(a_ok, b_ok):
    from math import comb
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum()); n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p

def boot_delta(preds_a, preds_b, y, nc, small, key, rng):
    """paired bootstrap over test images of the seed-averaged metric difference (b - a)"""
    n = len(y); d = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n); yy = y[i]
        ma = np.mean([metrics_fast(p[i], yy, nc, small)[key] for p in preds_a])
        mb = np.mean([metrics_fast(p[i], yy, nc, small)[key] for p in preds_b])
        d.append(mb - ma)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def metrics_fast(pred, y, nc, small):
    rec = per_class_recall(pred, y, nc); sm = [rec[c] for c in small if not np.isnan(rec[c])]
    return {"accuracy": 100 * (pred == y).mean(), "macro_f1": macro_f1(pred, y, nc),
            "minority_recall": 100 * np.mean(sm) if sm else np.nan}

def main(folders):
    files = defaultdict(dict); meta = {}
    for f in folders:
        f = Path(f)
        for p in f.glob("*_test_labels.npy"):
            name = p.name[: -len("_test_labels.npy")]
            z = np.load(f / ("%s_audit.npz" % name)); cls = json.loads((f / ("%s_paths.json" % name)).read_text())["classes"]
            nc = len(cls); cnt = np.bincount(z["y"][z["train"]], minlength=nc)
            small = [int(c) for c in np.argsort(cnt)[: max(1, nc // 3)]]
            meta[name] = dict(y=np.load(p), nc=nc, small=small, classes=cls, minority=[cls[c] for c in small])
        for p in f.glob("*.npy"):
            m = re.match(r"(.+?)_(efficientnet_b0|vit_small_patch16_224)_(raw|cleaned)_(none|weighted|oversample|focal)_(\d+)\.npy$", p.name)
            if m: files[(m[1], m[2])][(m[3], m[4], int(m[5]))] = p
    out, md, rng = {}, [], np.random.default_rng(0)
    for (name, arch), runs in sorted(files.items()):
        if name not in meta: print("skip", name, "(no test labels: run not finished)"); continue
        y, nc, small = meta[name]["y"], meta[name]["nc"], meta[name]["small"]
        P = {k: np.load(v).astype(np.float32) for k, v in runs.items()}
        seeds = sorted({k[2] for k in P}); key = "%s/%s" % (name, arch); out[key] = {"seeds": seeds, "minority_classes": meta[name]["minority"], "cells": {}}
        md += ["", "## %s, %s (test n = %d, seeds %s)" % (name, arch, len(y), seeds), "",
               "| Training data | Balancing | Accuracy | Macro-F1 | Balanced acc. | Minority recall | ECE | Δ macro-F1 vs raw/none [95% CI] | McNemar b/c, p (seed ensemble) |",
               "|---|---|---|---|---|---|---|---|---|"]
        base = [P[("raw", "none", s)] for s in seeds if ("raw", "none", s) in P]
        base_ens = np.mean(base, 0).argmax(1) == y if base else None
        for d in DATA:
            for b in BAL:
                ps = [P[(d, b, s)] for s in seeds if (d, b, s) in P]
                if not ps: continue
                M = [metrics(p, y, nc, small) for p in ps]
                agg = {k: (float(np.mean([m[k] for m in M])), float(np.std([m[k] for m in M], ddof=1)) if len(M) > 1 else 0.0) for k in M[0]}
                cell = {"n_seeds": len(ps), **{k: {"mean": v[0], "sd": v[1]} for k, v in agg.items()}}
                cmp = "–"; mc = "–"
                if base and (d, b) != ("raw", "none"):
                    lo, hi = boot_delta([p.argmax(1) for p in base], [p.argmax(1) for p in ps], y, nc, small, "macro_f1", rng)
                    dm = agg["macro_f1"][0] - np.mean([metrics(p, y, nc, small)["macro_f1"] for p in base])
                    bb, cc, pv = mcnemar(base_ens, np.mean(ps, 0).argmax(1) == y)
                    cell.update({"delta_macro_f1": float(dm), "delta_ci95": [lo, hi], "mcnemar": {"base_only": bb, "cond_only": cc, "p": pv}})
                    cmp = "%+.2f [%+.2f, %+.2f]" % (dm, lo, hi); mc = "%d/%d, %.3f" % (bb, cc, pv)
                out[key]["cells"]["%s/%s" % (d, b)] = cell
                f = lambda k, s=2: "%.*f ± %.*f" % (s, agg[k][0], s, agg[k][1])
                md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (d, b, f("accuracy"), f("macro_f1"), f("balanced_accuracy"),
                                                                       f("minority_recall"), f("ece", 3), cmp, mc))
    here = Path(__file__).parent
    (here / "grid_summary.json").write_text(json.dumps(out, indent=1))
    (here / "grid_tables.md").write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main(sys.argv[1:] or ["out_A", "out_B"])
