# Figures F1-F5 for the manuscript. Output: Paper\figures\F*.pdf and .png (300 dpi).
# Run from the repository root: python figures/make_figures.py
import sys, io, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

PAPER = Path(__file__).resolve().parent; ROOT = PAPER.parent; OUT = PAPER / "figures"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT)); import audit
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(OUT / ("%s.%s" % (name, ext)), dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved", name)

def box(ax, x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8.5, weight="bold")
    ax.text(x + 0.015, y + h - 0.11, "\n".join(lines), ha="left", va="top", fontsize=7, linespacing=1.45)
def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#333333"))

# ---------- F1: pipeline ----------
fig, ax = plt.subplots(figsize=(7.2, 2.0)); ax.set_axis_off(); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1)
def box1(x, title, lines, fc):
    w, h = 0.30, 0.96
    ax.add_patch(FancyBboxPatch((x, 0.02), w, h, boxstyle="round,pad=0.005,rounding_size=0.015", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, 0.93, title, ha="center", va="top", fontsize=7.5, weight="bold", linespacing=1.2)
    ax.text(x + 0.012, 0.66, "\n".join("• " + l for l in lines), ha="left", va="top", fontsize=6.0, linespacing=1.45)
box1(0.00, "Stage 1\nBenchmark integrity audit", ["64-bit difference hash per image", "exact multi-index search (radius 5)",
     "verification: correlation ≥ 0.90", "duplicate and conflicting groups", "copies between benchmarks"], "#e8f0fb")
box1(0.35, "Stage 2\nLeakage-free protocol", ["test set without training copies", "conflicting groups kept out of test",
     "group-aware training split", "group-aware, conflict-free validation", "duplicate and conflict lists"], "#eaf6ea")
box1(0.70, "Stage 3\nIntegrity-aware adaptive fusion", ["fine-tuned convolutional branch", "frozen DINOv2-L + CLIP ViT-L branch",
     "conflict-aware soft labels", "validation-gated combination", "temperature scaling"], "#fbefe3")
arrow(ax, 0.30, 0.5, 0.35, 0.5); arrow(ax, 0.65, 0.5, 0.70, 0.5)
save(fig, "F1_pipeline")

# ---------- F2: verification examples by correlation band (PlantVillage) ----------
def pair_images(name, bands, k, seed=0):
    recs, srcs = audit.load(name); meta = json.loads((ROOT / "cache" / ("%s_meta.json" % name)).read_text())["records"]
    idx = {r["id"]: i for i, r in enumerate(recs)}
    P = np.load(ROOT / "results" / ("%s_pairs.npy" % name)).astype(np.int64)
    def img(i):
        s = srcs[idx[meta[i]["id"]]]; x = Image.open(s if isinstance(s, str) else io.BytesIO(s)).convert("RGB"); x.thumbnail((220, 220)); return x
    out = []
    for lo, hi in bands:
        c = P[:, 3] / 1000; sel = P[(c >= lo) & (c < hi)]; rng = np.random.default_rng(seed); sel = sel[rng.permutation(len(sel))[:k]]
        out.append([(img(i), img(j), cc / 1000, meta[i]["label"], meta[j]["label"]) for i, j, _, cc in sel])
    return out
bands = [(0.0, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]
names = ["< 0.80", "0.80\u20130.90", "0.90\u20130.95 (accepted)", "\u2265 0.95 (accepted)"]
ex = pair_images("plantvillage", bands, 3)
fig, axes = plt.subplots(3, 8, figsize=(7.2, 3.0))
for b, col in enumerate(ex):
    for r, (a, bimg, cc, la, lb) in enumerate(col):
        for t, im in enumerate((a, bimg)):
            axx = axes[r, 2 * b + t]; axx.imshow(im); axx.set_xticks([]); axx.set_yticks([])
            for sp in axx.spines.values(): sp.set_visible(False)
        axes[r, 2 * b + 1].text(-0.04, -0.08, "r = %.2f" % cc, transform=axes[r, 2 * b + 1].transAxes, ha="center", va="top", fontsize=6)
    axes[0, 2 * b].set_title(names[b], fontsize=7.5, loc="left", x=0.15)
fig.subplots_adjust(wspace=0.04, hspace=0.32)
save(fig, "F2_verification_bands")

# ---------- F3: architecture ----------
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.set_axis_off(); ax.set_xlim(-0.02, 1.01); ax.set_ylim(0, 1)
def b(x, y, w, h, t, fc): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc, ec="#333", lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7)
b(0.00, 0.40, 0.10, 0.20, "Leaf\nimage", "#ffffff")
b(0.15, 0.72, 0.18, 0.18, "EfficientNet-B0\n(fine-tuned)", "#fbefe3")
b(0.15, 0.40, 0.18, 0.18, "DINOv2 ViT-L/14\n(frozen)", "#e8f0fb")
b(0.15, 0.08, 0.18, 0.18, "CLIP ViT-L/14\n(frozen)", "#e8f0fb")
b(0.40, 0.72, 0.17, 0.18, "softmax\nlog p_CNN", "#fbefe3")
b(0.40, 0.24, 0.17, 0.18, "L2-normalise,\nconcatenate,\nstandardise", "#e8f0fb")
b(0.615, 0.22, 0.16, 0.22, "Hybrid head\n(linear +\nprototypes)\nlog p_FM", "#e8f0fb")
b(0.62, 0.55, 0.15, 0.20, "Gate\n\u03b1 log p_CNN +\n(1\u2212\u03b1) log p_FM", "#f3e6f7")
b(0.82, 0.55, 0.17, 0.20, "Temperature\nscaling \u2192\ncalibrated p(y|x)", "#f3e6f7")
for (x0, y0, x1, y1) in [(0.10, 0.5, 0.15, 0.81), (0.10, 0.5, 0.15, 0.49), (0.10, 0.5, 0.15, 0.17), (0.33, 0.81, 0.40, 0.81),
                         (0.33, 0.49, 0.40, 0.36), (0.33, 0.17, 0.40, 0.30), (0.57, 0.33, 0.62, 0.33), (0.57, 0.81, 0.69, 0.75),
                         (0.69, 0.42, 0.69, 0.55), (0.77, 0.65, 0.82, 0.65)]:
    arrow(ax, x0, y0, x1, y1)
ax.text(0.695, 0.06, "\u03b1 and temperature chosen by validation NLL\n(group-aware, conflict-free split); training labels:\nconflict-aware soft labels, one unit of weight per duplicate group",
        ha="center", va="bottom", fontsize=6.5, style="italic")
save(fig, "F3_architecture")

# ---------- F4: conflicting-label examples (PlantWild) ----------
recs, srcs = audit.load("plantwild"); meta = json.loads((ROOT / "cache" / "plantwild_meta.json").read_text())["records"]
idx = {r["id"]: i for i, r in enumerate(recs)}
P = np.load(ROOT / "results" / "plantwild_pairs.npy").astype(np.int64); P = P[P[:, 3] >= 900]
want = [("cabbage alternaria leaf spot", "cauliflower alternaria leaf spot"), ("corn gray leaf spot", "corn northern leaf blight"),
        ("cucumber powdery mildew", "squash powdery mildew"), ("potato early blight", "tomato early blight"),
        ("ginger sheath blight", "rice sheath blight"), ("potato early blight", "potato late blight")]
def img(i):
    s = srcs[idx[meta[i]["id"]]]; x = Image.open(s).convert("RGB"); w, h = x.size; m = min(w, h); x = x.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((260, 260)); return x
fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.0))
for c, (la, lb) in enumerate(want):
    hit = next(((i, j, cc) for i, j, _, cc in P if {meta[i]["label"], meta[j]["label"]} == {la, lb}), None)
    if hit is None: continue
    i, j, cc = hit
    if meta[i]["label"] != la: i, j = j, i
    for r, k in enumerate((i, j)):
        axx = axes[r, c]; axx.imshow(img(k)); axx.set_xticks([]); axx.set_yticks([])
        for sp in axx.spines.values(): sp.set_visible(False)
        import textwrap; axx.set_xlabel(textwrap.fill(meta[k]["label"], 18), fontsize=5.8, labelpad=2)
    axes[0, c].set_title("r = %.3f" % (cc / 1000), fontsize=6.5)
fig.subplots_adjust(wspace=0.08, hspace=0.35)
save(fig, "F4_conflict_examples")

# ---------- F5: gate weights and gain of the adaptive model ----------
A = json.loads((ROOT / "kaggle" / "adaptive_summary.json").read_text())
order = [("plantdoc", "PlantDoc", "field/web"), ("plantwild", "PlantWild", "field/web"), ("cassava2020", "Cassava", "field/web"),
         ("rice_leaf", "Rice Leaf", "rice"), ("paddydoctor", "Paddy Doctor", "rice")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.5))
for k, (key, lab, kind) in enumerate(order):
    al = [g["alpha"] for g in A[key]["gate"] if g["labels"] == "soft"]; col = "#4c78a8" if kind == "field/web" else "#e45756"
    a1.bar(k, np.mean(al), color=col, alpha=0.85, width=0.6); a1.scatter([k] * len(al), al, color="k", s=8, zorder=3)
    m = A[key]["models"]; ada = m["adaptive/soft"]["macro_f1"][0]
    best_fixed = max(m["fusion2/soft"]["macro_f1"][0], m["fusion3/soft"]["macro_f1"][0], m["effnet_ft/keep"]["macro_f1"][0])
    a2.bar(k, ada - best_fixed, color=col, alpha=0.85, width=0.6)
for a in (a1, a2): a.set_xticks(range(len(order))); a.set_xticklabels([o[1] for o in order], rotation=20, ha="right")
a1.set_ylabel("gate \u03b1 (weight of CNN branch)"); a1.set_ylim(0, 0.6)
a2.set_ylabel("\u0394 macro-F1 vs best fixed model"); a2.axhline(0, color="k", lw=0.6)
from matplotlib.patches import Patch
a1.legend(handles=[Patch(color="#4c78a8", label="PlantDoc, PlantWild, Cassava"), Patch(color="#e45756", label="rice benchmarks")], frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); save(fig, "F5_gate")
