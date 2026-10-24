# Significance tests for the hybrid-fusion runs (notebook E), from the saved probabilities.
# H1: fusion vs the best single model; H2: label treatment (soft vs keep vs drop); H3: rankings, official vs clean test.
# Writes fusion_tables.md and fusion_summary.json next to this script.
import sys, json
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import analyse_grid
from analyse_grid import metrics, mcnemar, boot_delta
analyse_grid.BOOT = 1000

HERE = Path(__file__).parent; E = HERE / "out_E"
SEEDS = [42, 43, 44]
OLD = {"plantdoc": "out_A", "paddydoctor": "out_A", "cassava2020": "out_B", "rice_leaf": "out_C", "plantwild": "out_D"}

def load_meta(name):
    z = np.load(E / ("%s_fusion_split.npz" % name)); cls = json.loads((HERE / OLD[name] / ("%s_paths.json" % name)).read_text())["classes"]
    nc = len(cls); y = np.load(HERE / OLD[name] / ("%s_audit.npz" % name))["y"]
    cnt = np.bincount(y[z["train"]], minlength=nc); small = [int(c) for c in np.argsort(cnt)[: max(1, nc // 3)]]
    te_all = z["te_all"]; pos = {int(i): k for k, i in enumerate(te_all)}
    sel = {"clean": np.array([pos[int(i)] for i in z["test"]]), "official": np.array([pos[int(i)] for i in z["off_test"]])}
    return y, te_all, sel, nc, small

def runs(name, model, labels):
    return [np.load(E / ("%s_fusion_%s_%s_%d.npy" % (name, model, labels, s))).astype(np.float32) for s in SEEDS]

def old_vit(name):            # fine-tuned ViT-S from the grid (PlantDoc only), clean test only
    p = HERE / OLD[name]
    f = [p / ("%s_vit_small_patch16_224_raw_none_%d.npy" % (name, s)) for s in SEEDS]
    return [np.load(x).astype(np.float32) for x in f] if all(x.exists() for x in f) else None

def summ(P, y, nc, small):
    M = [metrics(p, y, nc, small) for p in P]
    return {k: (float(np.mean([m[k] for m in M])), float(np.std([m[k] for m in M], ddof=1))) for k in M[0]}

def compare(Pa, Pb, y, nc, small, rng):
    lo, hi = boot_delta([p.argmax(1) for p in Pa], [p.argmax(1) for p in Pb], y, nc, small, "macro_f1", rng)
    d = np.mean([metrics(p, y, nc, small)["macro_f1"] for p in Pb]) - np.mean([metrics(p, y, nc, small)["macro_f1"] for p in Pa])
    b, c, pv = mcnemar(np.mean(Pa, 0).argmax(1) == y, np.mean(Pb, 0).argmax(1) == y)
    return {"delta_macro_f1": float(d), "ci95": [lo, hi], "mcnemar": [b, c, pv]}

MODELS = [("effnet_ft", "keep"), ("clip_l", "keep"), ("dinov2_l", "keep"), ("fusion2", "keep"), ("fusion2", "drop"), ("fusion2", "soft"),
          ("fusion3", "keep"), ("fusion3", "drop"), ("fusion3", "soft")]
out, md, rng = {}, [], np.random.default_rng(0)
for name in OLD:
    y_all, te_all, sel, nc, small = load_meta(name)
    yc = y_all[te_all[sel["clean"]]]
    P = {m: [p[sel["clean"]] for p in runs(name, *m)] for m in MODELS}
    vit = old_vit(name)
    if vit is not None: P[("vit_s_ft", "keep")] = vit
    S = {m: summ(v, yc, nc, small) for m, v in P.items()}
    singles = [m for m in P if m[0] in ("effnet_ft", "clip_l", "dinov2_l", "vit_s_ft")]
    best_single = max(singles, key=lambda m: S[m]["macro_f1"][0])
    o = {"best_single": "%s/%s" % best_single, "models": {"%s/%s" % m: S[m] for m in P}, "H1": {}, "H2": {}}
    md += ["", "## %s (clean test n = %d)" % (name, len(yc)), "",
           "| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE |", "|---|---|---|---|---|---|"]
    for m in P:
        s = S[m]; md.append("| %s | %s | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f | %.3f |" % (m[0], m[1], *s["accuracy"], *s["macro_f1"], *s["minority_recall"], s["ece"][0]))
    md += ["", "H1, fusion vs best single model (%s), and vs fine-tuned EfficientNet:" % best_single[0], "",
           "| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |", "|---|---|---|"]
    for f in [("fusion2", "keep"), ("fusion2", "soft"), ("fusion3", "keep"), ("fusion3", "soft")]:
        for ref in dict.fromkeys([best_single, ("effnet_ft", "keep")]):
            r = compare(P[ref], P[f], yc, nc, small, rng); o["H1"]["%s/%s vs %s/%s" % (f + ref)] = r
            md.append("| %s/%s vs %s | %+.2f [%+.2f, %+.2f] | %d/%d, %.3f |" % (f[0], f[1], ref[0], r["delta_macro_f1"], *r["ci95"], *r["mcnemar"]))
    md += ["", "H2, label treatment:", "", "| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |", "|---|---|---|"]
    for fm in ("fusion2", "fusion3"):
        for a, b in (("keep", "soft"), ("drop", "soft"), ("keep", "drop")):
            r = compare(P[(fm, a)], P[(fm, b)], yc, nc, small, rng); o["H2"]["%s %s->%s" % (fm, a, b)] = r
            md.append("| %s: %s → %s | %+.2f [%+.2f, %+.2f] | %d/%d, %.3f |" % (fm, a, b, r["delta_macro_f1"], *r["ci95"], *r["mcnemar"]))
    if len(sel["official"]) != len(sel["clean"]):
        yo = y_all[te_all[sel["official"]]]
        So = {m: summ([p[sel["official"]] for p in runs(name, *m)], yo, nc, small) for m in MODELS}
        rank_c = sorted(MODELS, key=lambda m: -S[m]["macro_f1"][0]); rank_o = sorted(MODELS, key=lambda m: -So[m]["macro_f1"][0])
        o["H3"] = {"official_n": int(len(yo)), "rank_clean": ["%s/%s" % m for m in rank_c], "rank_official": ["%s/%s" % m for m in rank_o],
                   "official": {"%s/%s" % m: So[m] for m in MODELS}}
        md += ["", "H3, ranking by macro-F1 (clean n = %d vs official n = %d):" % (len(yc), len(yo)), "",
               "| Rank | Clean test | Official test |", "|---|---|---|"]
        for k in range(len(MODELS)):
            md.append("| %d | %s/%s %.2f | %s/%s %.2f |" % (k + 1, *rank_c[k], S[rank_c[k]]["macro_f1"][0], *rank_o[k], So[rank_o[k]]["macro_f1"][0]))
    out[name] = o
(HERE / "fusion_summary.json").write_text(json.dumps(out, indent=1))
(HERE / "fusion_tables.md").write_text("\n".join(md).strip() + "\n", encoding="utf-8")
print("\n".join(md))
