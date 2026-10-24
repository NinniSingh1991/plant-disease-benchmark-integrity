# %% integrity-aware adaptive fusion: validation-gated combination of a fine-tuned CNN branch and a frozen
# foundation-model branch, conflict-aware soft labels, temperature scaling. Every choice uses a held-out validation split.
ADA_RUN = ["plantdoc", "rice_leaf", "paddydoctor", "plantwild", "cassava2020"]
if os.environ.get("ADA_RUN"): ADA_RUN = os.environ["ADA_RUN"].split(",")
VAL_FRAC, VAL_SEED = 0.10, 1
ALPHAS = np.round(np.linspace(0, 1, 21), 2)
TEMPS = np.exp(np.linspace(np.log(0.05), np.log(20), 241))
RESULTS["_settings"].update({"val_frac": VAL_FRAC, "val_seed": VAL_SEED, "alpha_grid": ALPHAS.tolist()})

def make_val(D, train, groups, conflict):
    """group-aware stratified 10% of the training set; conflicting groups never enter validation; classes with fewer
    than 5 training images give no validation images"""
    n = len(D["y"]); gid = np.arange(n)
    for k, g in enumerate(groups): gid[g] = n + k
    bad = np.zeros(n, bool)
    for g in conflict: bad[g] = True
    members = defaultdict(list)
    for i in train: members[int(gid[i])].append(int(i))
    rng = np.random.default_rng(VAL_SEED); val = []
    for c in range(len(D["classes"])):
        cand = [g for g, m in members.items() if np.bincount(D["y"][m]).argmax() == c and not bad[m].any()]
        need = int(np.floor(VAL_FRAC * sum(len(members[g]) for g in members if np.bincount(D["y"][members[g]]).argmax() == c)))
        if (D["y"][train] == c).sum() < 5: continue
        rng.shuffle(cand); tot = 0
        for g in cand:
            if tot >= need: break
            val += members[g]; tot += len(members[g])
    val = np.array(sorted(val), dtype=np.int64)
    return val, np.setdiff1d(train, val)

def logp(P): return np.log(np.clip(P, 1e-8, 1.0))
def nll(L, y, T=1.0):
    Z = L / T; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
    return float(-lp[np.arange(len(y)), y].mean())
def soft(L, T=1.0):
    Z = L / T; Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)
def fit_temp(L, y): return float(TEMPS[np.argmin([nll(L, y, t) for t in TEMPS])]) if len(y) else 1.0   # no validation data: no scaling

def fit_gate(Lc, Lf, y):
    """alpha weights the CNN branch: L = alpha * log p_cnn + (1 - alpha) * log p_foundation; alpha and the temperature
    minimise validation NLL"""
    if not len(y): return 0.5, 1.0, float("nan")                 # no validation data: equal weights
    best = None
    for a in ALPHAS:
        L = a * Lc + (1 - a) * Lf; T = fit_temp(L, y); v = nll(L, y, T)
        if best is None or v < best[0] - 1e-9: best = (v, float(a), T)
    return best[1], best[2], best[0]

# %% adaptive fusion main loop
for name in ADA_RUN:
    log("=" * 20 + " adaptive " + name)
    D = load_dataset(name)
    a_, groups, conflict = audit(D)
    test, train, train_clean, s = make_split(D, groups, conflict, name)
    val, tr90 = make_val(D, train, groups, conflict)
    tr90_clean = np.intersect1d(train_clean, tr90)
    ev = np.concatenate([val, test]); nv = len(val)
    yv, yt = D["y"][val], D["y"][test]
    np.savez_compressed(OUTD / ("%s_ada_split.npz" % name), test=test, train=train, val=val, tr90=tr90, tr90_clean=tr90_clean)
    variants, n_soft = label_variants(D, tr90, tr90_clean, groups, conflict)
    cnt = np.bincount(D["y"][train], minlength=len(D["classes"])); small = [int(c) for c in np.argsort(cnt)[: max(1, len(cnt) // 3)]]
    R = RESULTS.setdefault("adaptive", {}).setdefault(name, {"split": s, "val": int(nv), "train90": int(len(tr90)), "n_soft": n_soft, "runs": [], "gate": []})
    log("%s: val %d, train90 %d, soft-label images %d" % (name, nv, len(tr90), n_soft))
    fz = frozen_features(D, np.concatenate([tr90, ev])); n90 = len(tr90)
    fz_tr = {k: v[:n90] for k, v in fz.items()}; fz_ev = {k: v[n90:] for k, v in fz.items()}
    def record(model, labels, seed, P_ev, calib):
        """P_ev: probabilities on [val, test]; calib: raw metrics and metrics after temperature scaling fitted on val"""
        np.save(OUTD / ("%s_ada_%s_%s_%d.npy" % (name, model, labels, seed)), P_ev.astype(np.float16))
        L = logp(P_ev); T = fit_temp(L[:nv], yv)
        for kind, P in (("raw", P_ev[nv:]), ("ts", soft(L[nv:], T))):
            m = metrics(yt, P, small); m.update({"model": model, "labels": labels, "seed": seed, "calib": kind, "T": T}); R["runs"].append(m)
        log("%s %-14s %-5s seed %d: acc %.2f macroF1 %.2f ece %.3f (ts %.3f)" % (name, model, labels, seed, R["runs"][-2]["accuracy"],
            R["runs"][-2]["macro_f1"], R["runs"][-2]["ece"], R["runs"][-1]["ece"]))
    for seed in SEEDS:
        t = time.time(); P_cnn, c_tr, c_ev = finetune_cnn(D, tr90, ev, seed); log("  cnn fine-tune seed %d (%.1f min)" % (seed, (time.time() - t) / 60))
        record("effnet_ft", "keep", seed, P_cnn, True)
        blocks = {"dinov2_l": (fz_tr["dinov2_l"], fz_ev["dinov2_l"]), "clip_l": (fz_tr["clip_l"], fz_ev["clip_l"]), "effnet": (c_tr, c_ev)}
        combos = {"dinov2_l": ["dinov2_l"], "clip_l": ["clip_l"], "fusion2": ["dinov2_l", "clip_l"], "fusion3": ["effnet", "dinov2_l", "clip_l"]}
        P_f2 = {}
        for model, parts in combos.items():
            Ftr, Fev = prep([blocks[p][0] for p in parts], [blocks[p][1] for p in parts])
            for labels in ("keep", "soft"):
                if model in ("dinov2_l", "clip_l") and labels != "keep": continue
                rows, Q, w = variants[labels]; P = train_head(Ftr[rows], Q, w, Fev, seed); record(model, labels, seed, P, True)
                if model == "fusion2": P_f2[labels] = P
        for labels in ("keep", "soft"):                       # the proposed model uses soft; keep is the ablation
            Lc, Lf = logp(P_cnn), logp(P_f2[labels])
            alpha, T, v = fit_gate(Lc[:nv], Lf[:nv], yv)
            P_ada = soft(alpha * Lc + (1 - alpha) * Lf, 1.0)          # temperature applied inside record (fitted on val)
            R["gate"].append({"labels": labels, "seed": seed, "alpha": alpha, "T_joint": T, "val_nll": v})
            log("  gate %s seed %d: alpha %.2f (weight of the CNN branch), T %.2f" % (labels, seed, alpha, T))
            record("adaptive", labels, seed, P_ada, True)
        save()
    np.save(OUTD / ("%s_ada_eval_labels.npy" % name), D["y"][ev])
    del D, fz, fz_tr, fz_ev; import gc; gc.collect()

# %% print everything (survives a failed download)
save(); log("finished")
print("=" * 30, "COPY EVERYTHING BELOW", "=" * 30)
print(json.dumps({k: {"gate": v["gate"], "split": v["split"], "val": v["val"]} for k, v in RESULTS.get("adaptive", {}).items()}, indent=1))
