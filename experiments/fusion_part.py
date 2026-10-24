# %% hybrid fusion: frozen foundation-model features (DINOv2-L, CLIP ViT-L) + fine-tuned EfficientNet-B0 features
FUSION_RUN = ["plantdoc", "rice_leaf", "paddydoctor", "plantwild", "cassava2020"]
if os.environ.get("FUSION_RUN"): FUSION_RUN = os.environ["FUSION_RUN"].split(",")
FROZEN = {"dinov2_l": os.environ.get("DINO_NAME", "vit_large_patch14_dinov2.lvd142m"),
          "clip_l": os.environ.get("CLIP_NAME", "vit_large_patch14_clip_224.openai")}
HEAD_EPOCHS, HEAD_BS, HEAD_LR, HEAD_WD, PROTOS = 60, 256, 1e-3, 1e-4, 4
if SMOKE: HEAD_EPOCHS = 5
RESULTS["_settings"].update({"frozen": FROZEN, "head_epochs": HEAD_EPOCHS, "head_lr": HEAD_LR, "protos_per_class": PROTOS})

class FeatDS(Dataset):
    def __init__(self, X, idx, tf): self.X, self.idx, self.tf = X, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, k): return self.tf(Image.fromarray(self.X[self.idx[k]]))

@torch.no_grad()
def extract(model, X, idx, tf):
    model.eval(); out = []
    for x in DataLoader(FeatDS(X, idx, tf), batch_size=128, num_workers=WORKERS):
        with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
            out.append(model(x.to(DEV)).float().cpu().numpy())
    return np.concatenate(out)

def frozen_features(D, idx):
    """frozen backbone features for every image in idx (one pass per backbone)"""
    feats = {}
    for key, name in FROZEN.items():
        m = timm.create_model(name, pretrained=True, num_classes=0, img_size=IMG).to(DEV)
        cfg = timm.data.resolve_data_config({}, model=m)
        tf = transforms.Compose([transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.BICUBIC),
                                 transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])
        t = time.time(); feats[key] = extract(m, D["X"], idx, tf); log("  %s features %s (%.1f min)" % (key, feats[key].shape, (time.time() - t) / 60))
        del m; torch.cuda.empty_cache()
    return feats

def finetune_cnn(D, tr, te_all, seed):
    """EfficientNet-B0 fine-tuned on the raw training set (same recipe as the grid); returns its test softmax on te_all
    and penultimate features for tr and te_all"""
    seed_all(seed); nc = len(D["classes"])
    dl = DataLoader(ArrDS(D["X"], D["y"], tr, train_tf), batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True,
                    drop_last=True, generator=torch.Generator().manual_seed(seed))
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=nc).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                out = model(x)
            loss = F.cross_entropy(out.float(), y); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    class Pre(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m.forward_head(s.m.forward_features(x), pre_logits=True)
    @torch.no_grad()
    def probs(idx):
        model.eval(); P = []
        for x, _ in DataLoader(ArrDS(D["X"], D["y"], idx, eval_tf), batch_size=128, num_workers=WORKERS):
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                P.append(torch.softmax(model(x.to(DEV)).float(), 1).cpu().numpy())
        return np.concatenate(P)
    f_tr = extract(Pre(model), D["X"], tr, eval_tf); f_te = extract(Pre(model), D["X"], te_all, eval_tf); P_te = probs(te_all)
    del model; torch.cuda.empty_cache()
    return P_te, f_tr, f_te

class HybridHead(nn.Module):
    """0.5 x linear logits + 0.5 x prototype logits (scaled cosine similarity to the nearest of K prototypes per class)"""
    def __init__(s, d, nc, protos):
        super().__init__(); s.lin = nn.Linear(d, nc); s.P = nn.Parameter(protos); s.scale = nn.Parameter(torch.tensor(10.0)); s.nc = nc
    def forward(s, x):
        sim = F.normalize(x, dim=1) @ F.normalize(s.P, dim=1).T
        return 0.5 * s.lin(x) + 0.5 * s.scale * sim.view(len(x), s.nc, -1).max(-1).values

def prep(blocks_tr, blocks_te):
    """L2-normalise each backbone block, concatenate, z-score with training statistics"""
    n = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    Ftr = np.concatenate([n(b) for b in blocks_tr], 1); Fte = np.concatenate([n(b) for b in blocks_te], 1)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    return ((Ftr - mu) / sd).astype(np.float32), ((Fte - mu) / sd).astype(np.float32)

def train_head(Ftr, Q, w, Fte, seed):
    seed_all(seed); nc = Q.shape[1]; hard = Q.argmax(1); rng = np.random.default_rng(seed)
    protos = []
    for c in range(nc):
        idx = np.flatnonzero(hard == c)
        pick = rng.choice(idx, PROTOS, replace=len(idx) < PROTOS) if len(idx) else rng.choice(len(Ftr), PROTOS)
        protos.append(Ftr[pick])
    head = HybridHead(Ftr.shape[1], nc, torch.tensor(np.concatenate(protos))).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    steps = HEAD_EPOCHS * int(np.ceil(len(Ftr) / HEAD_BS)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    Xt, Qt, Wt = torch.tensor(Ftr, device=DEV), torch.tensor(Q, device=DEV), torch.tensor(w, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(HEAD_EPOCHS):
        perm = torch.randperm(len(Ftr), generator=g).to(DEV)
        for k in range(0, len(Ftr), HEAD_BS):
            b = perm[k:k + HEAD_BS]; logp = F.log_softmax(head(Xt[b]), 1)
            loss = (-(Qt[b] * logp).sum(1) * Wt[b]).sum() / Wt[b].sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    head.eval()
    with torch.no_grad(): return torch.softmax(head(torch.tensor(Fte, device=DEV)), 1).cpu().numpy()

def label_variants(D, train, train_clean, groups, conflict):
    """keep: raw training set, hard labels. drop: cleaned training set (conflicting groups removed, one image per
    duplicate group). soft: raw training set; members of a conflicting group get the group's label distribution
    (training members only) and every duplicate group carries a total weight of one."""
    nc = len(D["classes"]); pos = {int(i): k for k, i in enumerate(train)}
    Q = np.eye(nc, dtype=np.float32)[D["y"][train]]; w = np.ones(len(train), np.float32)
    n_soft = 0
    for g in groups:
        mem = [pos[i] for i in g if i in pos]
        if len(mem) < 2: continue
        w[mem] = 1.0 / len(mem)
        labs = D["y"][train][mem]
        if len(set(labs.tolist())) > 1:
            Q[mem] = np.bincount(labs, minlength=nc) / len(labs); n_soft += len(mem)
    keep_pos = np.array([pos[int(i)] for i in train_clean])
    return {"keep": (np.arange(len(train)), np.eye(nc, dtype=np.float32)[D["y"][train]], np.ones(len(train), np.float32)),
            "drop": (keep_pos, np.eye(nc, dtype=np.float32)[D["y"][train_clean]], np.ones(len(keep_pos), np.float32)),
            "soft": (np.arange(len(train)), Q, w)}, n_soft

# %% fusion main loop
for name in FUSION_RUN:
    log("=" * 20 + " fusion " + name)
    D = load_dataset(name)
    a, groups, conflict = audit(D)
    test, train, train_clean, s = make_split(D, groups, conflict, name)
    off_test = np.flatnonzero(D["split"] == "test") if (D["split"] == "test").any() else test   # official test (PlantDoc only)
    te_all = np.unique(np.concatenate([test, off_test])); tpos = {int(i): k for k, i in enumerate(te_all)}
    sel_clean = np.array([tpos[int(i)] for i in test]); sel_off = np.array([tpos[int(i)] for i in off_test])
    np.savez_compressed(OUTD / ("%s_fusion_split.npz" % name), test=test, train=train, train_clean=train_clean, off_test=off_test, te_all=te_all)
    variants, n_soft = label_variants(D, train, train_clean, groups, conflict)
    cnt = np.bincount(D["y"][train], minlength=len(D["classes"])); small = [int(c) for c in np.argsort(cnt)[: max(1, len(cnt) // 3)]]
    R = RESULTS.setdefault("fusion", {}).setdefault(name, {"split": s, "n_soft_label_images": n_soft, "official_test": int(len(off_test)), "runs": []})
    fz = frozen_features(D, np.concatenate([train, te_all])); ntr = len(train)
    fz_tr = {k: v[:ntr] for k, v in fz.items()}; fz_te = {k: v[ntr:] for k, v in fz.items()}
    def record(model_name, labels, seed, P):
        for tname, sel in (("clean", sel_clean), ("official", sel_off)):
            if tname == "official" and len(off_test) == len(test) and (off_test == test).all(): continue
            yy = D["y"][te_all[sel]]; m = metrics(yy, P[sel], small)
            m.update({"model": model_name, "labels": labels, "seed": seed, "test": tname}); R["runs"].append(m)
        np.save(OUTD / ("%s_fusion_%s_%s_%d.npy" % (name, model_name, labels, seed)), P.astype(np.float16))
        mc = R["runs"][-1] if R["runs"][-1]["test"] == "clean" else R["runs"][-2]
        log("%s %-12s %-5s seed %d: acc %.2f macroF1 %.2f" % (name, model_name, labels, seed, mc["accuracy"], mc["macro_f1"]))
    for seed in SEEDS:
        t = time.time(); P_cnn, c_tr, c_te = finetune_cnn(D, train, te_all, seed); log("  cnn fine-tune seed %d (%.1f min)" % (seed, (time.time() - t) / 60))
        record("effnet_ft", "keep", seed, P_cnn)
        blocks = {"dinov2_l": (fz_tr["dinov2_l"], fz_te["dinov2_l"]), "clip_l": (fz_tr["clip_l"], fz_te["clip_l"]), "effnet": (c_tr, c_te)}
        combos = {"dinov2_l": ["dinov2_l"], "clip_l": ["clip_l"], "fusion2": ["dinov2_l", "clip_l"], "fusion3": ["effnet", "dinov2_l", "clip_l"]}
        for model_name, parts in combos.items():
            Ftr, Fte = prep([blocks[p][0] for p in parts], [blocks[p][1] for p in parts])
            for labels, (rows, Q, w) in variants.items():
                if model_name in ("dinov2_l", "clip_l") and labels != "keep": continue
                record(model_name, labels, seed, train_head(Ftr[rows], Q, w, Fte, seed))
        save()
    np.save(OUTD / ("%s_fusion_test_labels.npy" % name), D["y"][te_all])
    del D, fz, fz_tr, fz_te; import gc; gc.collect()

# %% print everything (survives a failed download)
save(); log("finished")
print("=" * 30, "COPY EVERYTHING BELOW", "=" * 30)
print(json.dumps(RESULTS.get("fusion", {}), indent=1))
