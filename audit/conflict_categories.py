# Categorises every verified conflicting duplicate group: different crop, same crop healthy vs diseased, or same crop
# with two different diseases. Also checks label agreement for copies shared between benchmarks.
# Writes results/conflict_categories.json and prints a summary.
import json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent; RES = ROOT / "results"; CACHE = ROOT / "cache"
VERIFY = 900                     # correlation x 1000
NAMES = {
    "paddydoctor": "bacterial_leaf_blight bacterial_leaf_streak bacterial_panicle_blight blast brown_spot dead_heart downy_mildew hispa normal tungro".split(),
    "rice_leaf": "Bacterial_Leaf_Blight Brown_Spot Healthy_Rice_Leaf Leaf_Blast Leaf_Scald Sheath_Blight".split(),
    "plantpathology2021": "complex frog_eye_leaf_spot healthy powdery_mildew rust scab".split(),
}
MULTI = {"bell pepper": "pepper", "bell_pepper": "pepper", "grapevine": "grape", "soyabean": "soybean"}

def crop_disease(ds, label):
    """-> (crop, disease or 'healthy')"""
    if ds == "plantvillage":
        c, d = label.split("___"); c = c.split("_(")[0].split(",")[0].lower(); return c, "healthy" if d == "healthy" else d.lower()
    if ds in ("paddydoctor", "rice_leaf"):
        n = NAMES[ds][int(label)]; return "rice", "healthy" if n in ("normal", "Healthy_Rice_Leaf") else n.lower()
    if ds == "plantpathology2021":
        parts = sorted(NAMES[ds][int(x)] for x in label.split("+")); return "apple", "healthy" if parts == ["healthy"] else "+".join(parts)
    if ds == "corn_maize":
        return "corn", CORN.get(label, "class_" + label)
    l = label.lower().replace("_", " ")
    for k, v in MULTI.items():
        if l.startswith(k.replace("_", " ")): l = v + l[len(k):]
    crop = l.split()[0]; rest = re.sub(r"\bleaf\b", "", l[len(crop):]).strip()
    return crop, "healthy" if rest == "" else rest

def groups_of(ds):
    m = json.loads((CACHE / ("%s_meta.json" % ds)).read_text())["records"]
    P = np.load(RES / ("%s_pairs.npy" % ds)); P = P[P[:, 3] >= VERIFY]
    par = list(range(len(m)))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in P[:, :2]: par[f(int(i))] = f(int(j))
    g = defaultdict(list)
    for i in range(len(m)): g[f(i)].append(i)
    return [v for v in g.values() if len(v) > 1], m

# corn label names from the PlantVillage copies (92% of Corn images are PlantVillage images)
rec = json.loads((RES / "cross_records.json").read_text()); X = np.load(RES / "cross_pairs.npy").astype(int)
votes = defaultdict(Counter)
for i, j, _ in X:
    a, b = rec[i], rec[j]
    if {a["ds"], b["ds"]} == {"corn_maize", "plantvillage"}:
        c, p = (a, b) if a["ds"] == "corn_maize" else (b, a); votes[c["label"]][p["label"]] += 1
CORN = {k: crop_disease("plantvillage", v.most_common(1)[0][0])[1] for k, v in votes.items()}

out = {"corn_label_map_from_plantvillage": {k: dict(v.most_common(3)) for k, v in votes.items()}}
print("Corn labels inferred from PlantVillage copies:", CORN)
for ds in ["plantvillage", "plantdoc", "paddydoctor", "rice_leaf", "plantwild", "plantpathology2021", "corn_maize"]:
    groups, m = groups_of(ds)
    cats, pairs = Counter(), Counter()
    for g in groups:
        labs = sorted({m[i]["label"] for i in g})
        if len(labs) < 2: continue
        cd = [crop_disease(ds, l) for l in labs]; crops = {c for c, _ in cd}
        if len(crops) > 1: k = "different crop"
        elif any(d == "healthy" for _, d in cd): k = "same crop, healthy vs diseased"
        else: k = "same crop, different disease"
        cats[k] += 1; pairs[" | ".join(labs)] += 1
    n = sum(cats.values())
    out[ds] = {"conflicting_groups": n, "categories": dict(cats), "pct": {k: 100 * v / n for k, v in cats.items()} if n else {},
               "top_label_pairs": pairs.most_common(15)}
    print("\n%s: %d conflicting groups" % (ds, n)); [print("  %-32s %4d (%.1f%%)" % (k, v, 100 * v / n)) for k, v in cats.most_common()]
    if ds == "plantwild": [print("    ", c, p) for p, c in pairs.most_common(10)]

# label agreement for copies shared between benchmarks (same crop and same disease after name mapping)
SYN = {("corn", "blight"): "northern blight", ("corn", "northern blight"): "northern blight",
       ("corn", "cercospora spot gray spot"): "gray spot", ("corn", "gray spot"): "gray spot",
       ("corn", "common rust"): "rust", ("corn", "rust"): "rust",
       ("tomato", "bacterial spot"): "bacterial spot", ("tomato", "mold"): "mold", ("tomato", "yellow virus"): "yellow curl virus",
       ("tomato", "yellow curl virus"): "yellow curl virus", ("tomato", "tomato yellow curl virus"): "yellow curl virus",
       ("tomato", "septoria spot"): "septoria spot", ("tomato", "mosaic virus"): "mosaic virus", ("tomato", "tomato mosaic virus"): "mosaic virus",
       ("apple", "scab"): "scab", ("apple", "apple scab"): "scab", ("apple", "rust"): "rust", ("apple", "cedar apple rust"): "rust",
       ("squash", "powdery mildew"): "powdery mildew", ("pepper", "spot"): "bacterial spot", ("pepper", "bacterial spot"): "bacterial spot",
       ("grape", "black rot"): "black rot", ("potato", "early blight"): "early blight", ("potato", "late blight"): "late blight",
       ("tomato", "early blight"): "early blight", ("tomato", "late blight"): "late blight",
       ("grape", "blight (isariopsis spot)"): "spot"}
def norm(cd):
    c, d = cd; d = re.sub(r"\bleaf\b", " ", d.replace("_", " ")); d = re.sub(r"\s+", " ", d).strip()
    return c, SYN.get((c, d), d)
mism = Counter()
agree = defaultdict(Counter)
for i, j, _ in X:
    a, b = rec[i], rec[j]
    if a["ds"] == b["ds"]: continue
    key = " | ".join(sorted([a["ds"], b["ds"]]))
    ca, cb = norm(crop_disease(a["ds"], a["label"])), norm(crop_disease(b["ds"], b["label"]))
    if ca != cb: mism[(key, ca, cb) if ca < cb else (key, cb, ca)] += 1
    k = "same crop and condition" if ca == cb else ("same crop, other condition" if ca[0] == cb[0] else "different crop")
    agree[key][k] += 1
out["cross_benchmark_label_agreement"] = {k: dict(v) for k, v in agree.items()}
out["cross_benchmark_mismatches"] = [[k[0], list(k[1]), list(k[2]), v] for k, v in mism.most_common()]
print("\nCross-benchmark pairs:"); [print("  %-28s %s" % (k, dict(v))) for k, v in agree.items()]
print("\nTop mismatches:"); [print("  ", v, k) for k, v in mism.most_common(25)]
(RES / "conflict_categories.json").write_text(json.dumps(out, indent=1))
