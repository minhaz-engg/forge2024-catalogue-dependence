"""s07: stage-to-stage spatial overlap of the 16A seismicity (fraction of a stage's events within d of an earlier
stage's events), computed on the same matched events in each catalogue."""
import numpy as np, pandas as pd, json
from scipy.spatial import cKDTree
import fz

P = pd.read_csv('results/matched_pairs.csv'); P = P[P.stage.isin(fz.STAGES_16A)].reset_index(drop=True)
rows = []
for cat, cols in [('N', ['xN', 'yN', 'zN']), ('Z', ['xZ', 'yZ', 'zZ'])]:
    X = P[cols].values
    for k, s in enumerate(fz.STAGES_16A[1:], start=1):
        prev = np.isin(P.stage, fz.STAGES_16A[:k]); cur = (P.stage == s).values
        dist, _ = cKDTree(X[prev]).query(X[cur])
        # nearest-neighbour distance within the stage itself, for scale
        dself, _ = cKDTree(X[cur]).query(X[cur], k=2)
        r = dict(cat=cat, stage=s, n=cur.sum(), nn_prev_med=np.median(dist), nn_self_med=np.median(dself[:, 1]))
        for d in [25, 50, 100]:
            r[f'f{d}'] = (dist <= d).mean()
        # nearest earlier stage of each event
        rows.append(r)
O = pd.DataFrame(rows); O.to_csv('results/stage_overlap.csv', index=False)
W = O.pivot(index='stage', columns='cat', values=['f25', 'f50', 'nn_prev_med', 'nn_self_med']).reindex(fz.STAGES_16A[1:])
print(W.round(2).to_string())
out = {}
for d in [25, 50, 100]:
    a = O[O.cat == 'N'][f'f{d}'].values; b = O[O.cat == 'Z'][f'f{d}'].values
    out[f'f{d}_N_med'] = np.median(a); out[f'f{d}_Z_med'] = np.median(b)
    out[f'f{d}_absdiff_med'] = np.median(np.abs(a - b)); out[f'f{d}_absdiff_max'] = np.max(np.abs(a - b))
    out[f'f{d}_spearman'] = pd.Series(a).corr(pd.Series(b), method='spearman')
out['nnself_N_med'] = O[O.cat == 'N'].nn_self_med.median(); out['nnself_Z_med'] = O[O.cat == 'Z'].nn_self_med.median()
print({k: round(float(v), 3) for k, v in out.items()})
json.dump({k: float(v) for k, v in out.items()}, open('results/overlap.json', 'w'), indent=1)
