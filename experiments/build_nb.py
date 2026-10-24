# Builds the two Kaggle notebooks (A: PlantDoc + Paddy Doctor, B: Cassava) from clean_balance_runs.py
import json, re
src = open('clean_balance_runs.py', encoding='utf-8').read()
def nb(s, out):
    cells = []
    for p in re.split(r'^# %%', s, flags=re.M):
        if not p.strip(): continue
        if p.startswith(' [markdown]'):
            body = p[len(' [markdown]'):].strip('\n')
            md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)})
        else:
            first, _, rest = p.partition('\n')
            code = (('# ' + first.strip() + '\n') if first.strip() else '') + rest
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.strip('\n').splitlines(True)})
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}, open(out, 'w', encoding='utf-8'), indent=1)
    print(out, len(cells), 'cells')
A = src.replace('RUN = ["plantdoc", "paddydoctor"]', 'RUN = ["plantdoc", "paddydoctor"]')
B = src.replace('RUN = ["plantdoc", "paddydoctor"]', 'RUN = ["cassava2020"]')
C = src.replace('RUN = ["plantdoc", "paddydoctor"]', 'RUN = ["rice_leaf", "paddydoctor"]').replace(
    'GRID_ON = None ', 'GRID_ON = ["rice_leaf"] ').replace('LEAK_ON = [] ', 'LEAK_ON = ["rice_leaf", "paddydoctor"] ')
D = src.replace('RUN = ["plantdoc", "paddydoctor"]', 'RUN = ["plantwild"]').replace('LEAK_ON = [] ', 'LEAK_ON = ["plantwild"] ')
assert len({A, B, C, D}) == 4 and 'GRID_ON = ["rice_leaf"]' in C and 'LEAK_ON = ["plantwild"]' in D
nb(A, 'notebook_A_plantdoc_paddy.ipynb'); nb(B, 'notebook_B_cassava.ipynb')
nb(C, 'notebook_C_rice_paddyleak.ipynb'); nb(D, 'notebook_D_plantwild.ipynb')

# E: hybrid fusion notebook = loading/audit/split/training cells of the grid script + fusion_part.py
head_md = '''# %% [markdown]
# # Integrity-aware hybrid fusion for plant disease recognition
#
# Same audit (64-bit dHash, multi-index radius 5, thumbnail correlation >= 0.90) and the same fixed clean test sets as the
# clean-and-balance grid. Models: fine-tuned EfficientNet-B0; frozen DINOv2-L and CLIP ViT-L with a linear + prototype
# head; fusion of DINOv2-L + CLIP ViT-L (fusion2) and of EfficientNet + DINOv2-L + CLIP ViT-L (fusion3). Training labels:
# keep (raw), drop (conflicting groups removed, one image per duplicate group), soft (conflict-aware soft labels, one unit
# of weight per duplicate group). Kaggle: GPU T4, Internet ON, inputs: paddy-disease-classification and
# cassava-leaf-disease-classification competitions.

'''
base = src[:src.index('# %% leakage test')]
base = head_md + base[base.index('# %%\nRUN ='):]
E = base + open('fusion_part.py', encoding='utf-8').read()
open('fusion_runs_full.py', 'w', encoding='utf-8').write(E)
nb(E, 'notebook_E_fusion.ipynb')

# F: adaptive fusion = grid cells + fusion functions (no fusion main loop) + adaptive_part.py
fus = open('fusion_part.py', encoding='utf-8').read(); fus = fus[:fus.index('# %% fusion main loop')]
head_f = head_md.replace('# # Integrity-aware hybrid fusion for plant disease recognition', '# # Integrity-aware adaptive fusion for plant disease recognition').replace(
    'Training labels:', 'Adaptive model: validation-gated weighting of the CNN branch against the foundation-model fusion, temperature scaling on the same validation split (10% of training, group-aware). Training labels:')
F_src = head_f + base[len(head_md):] + fus + open('adaptive_part.py', encoding='utf-8').read()
open('adaptive_runs_full.py', 'w', encoding='utf-8').write(F_src)
nb(F_src, 'notebook_F_adaptive.ipynb')
