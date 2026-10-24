# Analysis of the adaptive-fusion run (notebook F): accuracy, macro-F1, ECE (raw and temperature-scaled), gate values,
# and significance of the adaptive model against every fixed alternative. Writes adaptive_tables.md / adaptive_summary.json.
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import analyse_grid
from analyse_grid import metrics, mcnemar, boot_delta
analyse_grid.BOOT = 1000

HERE = Path(__file__).parent; F = HERE / "out_F"
OLD = {"plantdoc": "out_A", "paddydoctor": "out_A", "cassava2020": "out_B", "rice_leaf": "out_C", "plantwild": "out_D"}
R = json.loads((F / "integrity_results.json").read_text())["adaptive"]
MODELS = [("effnet_ft", "keep"), ("clip_l", "keep"), ("dinov2_l", "keep"), ("fusion2", "keep"), ("fusion2", "soft"),
          ("fusion3", "keep"), ("fusion3", "soft"), ("adaptive", "keep"), ("adaptive", "soft")]
out, md, rng = {}, [], np.random.default_rng(0)
for name, r in R.items():
    z = np.load(F / ("%s_ada_split.npz" % name)); nv = len(z["val"])
    y = np.load(HERE / OLD[name] / ("%s_audit.npz" % name))["y"]; yt = y[z["test"]]
    cls = json.loads((HERE / OLD[name] / ("%s_paths.json" % name)).read_text())["classes"]; nc = len(cls)
    cnt = np.bincount(y[z["train"]], minlength=nc); small = [int(c) for c in np.argsort(cnt)[: max(1, nc // 3)]]
    seeds = sorted({x["seed"] for x in r["runs"]})
    # every metric is recomputed from the saved (float16) probabilities so that tables and tests share one source;
    # the temperature is refitted on the validation part exactly as in the notebook
    TEMPS = np.exp(np.linspace(np.log(0.05), np.log(20), 241)); yv = y[z["val"]]
    def nll(L, yy, T):
        Z = L / T; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True)); return -lp[np.arange(len(yy)), yy].mean()
    agg = defaultdict(list)
    for m in MODELS:
        for s_ in seeds:
            Pf = np.load(F / ("%s_ada_%s_%s_%d.npy" % (name, m[0], m[1], s_))).astype(np.float32); L = np.log(np.clip(Pf, 1e-8, 1))
            T = TEMPS[np.argmin([nll(L[:nv], yv, t) for t in TEMPS])] if nv else 1.0
            Z = L[nv:] / T; Z = Z - Z.max(1, keepdims=True); Pts = np.exp(Z); Pts /= Pts.sum(1, keepdims=True)
            agg[m + ("raw",)].append(metrics(Pf[nv:], yt, nc, small)); agg[m + ("ts",)].append(metrics(Pts, yt, nc, small))
    st = lambda k, key: (float(np.mean([v[key] for v in agg[k]])), float(np.std([v[key] for v in agg[k]], ddof=1)) if len(agg[k]) > 1 else 0.0)
    md += ["", "## %s (test n = %d, validation n = %d, train90 n = %d, soft-label images %d)" % (name, len(yt), nv, r["train90"], r["n_soft"]), "",
           "| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE raw | ECE after TS |", "|---|---|---|---|---|---|---|"]
    o = {"models": {}, "gate": r["gate"], "tests": {}}
    for m in MODELS:
        if (m[0], m[1], "raw") not in agg: continue
        a, f, mr, e = st(m + ("raw",), "accuracy"), st(m + ("raw",), "macro_f1"), st(m + ("raw",), "minority_recall"), st(m + ("raw",), "ece")
        ets = st(m + ("ts",), "ece")
        o["models"]["%s/%s" % m] = {"accuracy": a, "macro_f1": f, "minority_recall": mr, "ece_raw": e, "ece_ts": ets}
        md.append("| %s | %s | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f | %.3f | %.3f |" % (m[0], m[1], *a, *f, *mr, e[0], ets[0]))
    al = [g["alpha"] for g in r["gate"] if g["labels"] == "soft"]
    md += ["", "Gate (weight of the fine-tuned CNN branch, soft labels): %s, mean %.2f" % (", ".join("%.2f" % v for v in al), np.mean(al)), "",
           "| Adaptive (soft) vs | Δ macro-F1 [95% CI] | McNemar b/c, p |", "|---|---|---|"]
    P = {m: [np.load(F / ("%s_ada_%s_%s_%d.npy" % (name, m[0], m[1], s))).astype(np.float32)[nv:] for s in seeds] for m in MODELS}
    for ref in [("fusion2", "soft"), ("fusion2", "keep"), ("fusion3", "soft"), ("effnet_ft", "keep"), ("dinov2_l", "keep"), ("adaptive", "keep")]:
        lo, hi = boot_delta([p.argmax(1) for p in P[ref]], [p.argmax(1) for p in P[("adaptive", "soft")]], yt, nc, small, "macro_f1", rng)
        d = np.mean([metrics(p, yt, nc, small)["macro_f1"] for p in P[("adaptive", "soft")]]) - np.mean([metrics(p, yt, nc, small)["macro_f1"] for p in P[ref]])
        b, c, pv = mcnemar(np.mean(P[ref], 0).argmax(1) == yt, np.mean(P[("adaptive", "soft")], 0).argmax(1) == yt)
        o["tests"]["%s/%s" % ref] = {"delta_macro_f1": float(d), "ci95": [lo, hi], "mcnemar": [b, c, pv]}
        md.append("| %s/%s | %+.2f [%+.2f, %+.2f] | %d/%d, %.3f |" % (*ref, d, lo, hi, b, c, pv))
    out[name] = o
(HERE / "adaptive_summary.json").write_text(json.dumps(out, indent=1))
(HERE / "adaptive_tables.md").write_text("\n".join(md).strip() + "\n", encoding="utf-8")
print("\n".join(md))
