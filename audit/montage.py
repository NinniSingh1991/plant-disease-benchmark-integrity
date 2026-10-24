# Side-by-side montage of hash-matched pairs, one row per pair, for visual verification.
import sys, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, '.'); import audit
name, lo, hi, k = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
recs, srcs = audit.load(name)
meta = json.load(open('cache/%s_meta.json' % name))['records']
idx = {r['id']: i for i, r in enumerate(recs)}
P = np.load("results/%s_pairs.npy" % name).astype(np.int64); c = P[:, 3] / 1000
sel = P[(c >= lo) & (c < hi)]
rng = np.random.default_rng(0); sel = sel[rng.permutation(len(sel))[:k]]
S = 160; im = Image.new('RGB', (2 * S + 260, S * len(sel)), 'white'); d = ImageDraw.Draw(im)
def get(i):
    s = srcs[idx[meta[i]['id']]]
    import io
    x = Image.open(s if isinstance(s, str) else io.BytesIO(s)).convert('RGB'); x.thumbnail((S, S)); return x
for r, (i, j, dist, cc) in enumerate(sel):
    im.paste(get(i), (0, r * S)); im.paste(get(j), (S, r * S))
    d.text((2 * S + 5, r * S + 5), 'd=%d corr=%.3f\nA: %s\nB: %s\n%s | %s' % (dist, cc / 1000, meta[i]['label'][:30], meta[j]['label'][:30], meta[i]['split'], meta[j]['split']), fill='black')
im.save('review/%s_%.2f_%.2f.png' % (name, lo, hi)); print(len(sel), 'pairs')
