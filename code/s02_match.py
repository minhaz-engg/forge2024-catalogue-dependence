"""s02: one-to-one origin-time matching of the two catalogues."""
import numpy as np, pandas as pd
import fz

N = pd.read_csv('data/catalog_Niemz.csv'); Z = pd.read_csv('data/catalog_Zhu.csv')

def nearest_dt(tn, tz):
    o = np.argsort(tz); s = tz[o]
    i = np.clip(np.searchsorted(s, tn), 1, len(s) - 1)
    j = np.where(np.abs(tn - s[i - 1]) <= np.abs(s[i] - tn), i - 1, i)
    return tn - s[j], o[j]

dt, _ = nearest_dt(N.t.values, Z.t.values)
dt0 = np.median(dt[np.abs(dt) < 1.0])                      # clock / origin-time offset between catalogues
print(f'median origin-time offset (Niemz - Zhu) = {dt0*1000:.1f} ms')
h, e = np.histogram(dt - dt0, np.arange(-2, 2.01, 0.05))
print('histogram of nearest-neighbour time differences (|dt|<2 s), 0.05-s bins, counts:', h.tolist())

def match(tol):
    """Greedy one-to-one matching in order of increasing |dt - dt0|."""
    tz = Z.t.values + dt0
    pairs = []
    for i, tn in enumerate(N.t.values):
        k = np.where(np.abs(tz - tn) <= tol)[0]
        for j in k: pairs.append((abs(tz[j] - tn), i, j))
    pairs.sort(); un, uz, out = set(), set(), []
    for d, i, j in pairs:
        if i in un or j in uz: continue
        un.add(i); uz.add(j); out.append((i, j, N.t.values[i] - Z.t.values[j]))
    return pd.DataFrame(out, columns=['iN', 'iZ', 'dt'])

sens = []
for tol in [0.1, 0.25, 0.5, 1.0, 2.0]:
    m = match(tol); sens.append(dict(tol=tol, n=len(m)))
    print(f'tol {tol:4.2f} s: {len(m)} pairs')
pd.DataFrame(sens).to_csv('results/match_tolerance.csv', index=False)

TOL = 0.5
m = match(TOL)
a, b = N.iloc[m.iN].reset_index(drop=True), Z.iloc[m.iZ].reset_index(drop=True)
P = pd.DataFrame(dict(idN=a.id, idZ=b.id, t=a.t, dt=m.dt, stageN=a.stage, stageZ=b.stage,
                      xN=a.x, yN=a.y, zN=a.z, MN=a.M, xZ=b.x, yZ=b.y, zZ=b.z, MZ=b.M, sx=b.sx, sy=b.sy, sz=b.sz))
P['stage'] = P.stageN.fillna('')
P.to_csv('results/matched_pairs.csv', index=False)
print(f'\nmatched with tol {TOL} s: {len(P)} pairs = {len(P)/len(N):.1%} of Niemz ({len(N)}), {len(P)/len(Z):.1%} of Zhu ({len(Z)})')
print('stage label agreement:', (P.stageN.fillna('') == P.stageZ.fillna('')).mean().round(4))

# matched fraction by stage and by magnitude
N['matched'] = N.id.isin(P.idN); Z['matched'] = Z.id.isin(P.idZ)
rows = []
for s in fz.STAGES_16A + ['16B-1', '16B-2', '16B-3']:
    rows.append(dict(stage=s, nN=(N.stage == s).sum(), nZ=(Z.stage == s).sum(), n_match=(P.stage == s).sum()))
bs = pd.DataFrame(rows); bs['fN'] = bs.n_match / bs.nN; bs['fZ'] = bs.n_match / bs.nZ
print(bs.round(3).to_string(index=False)); bs.to_csv('results/match_by_stage.csv', index=False)
for name, C in [('Niemz', N), ('Zhu', Z)]:
    q = pd.qcut(C.M, 5)
    print(name, 'matched fraction by magnitude quintile:', C.groupby(q, observed=True).matched.mean().round(3).to_dict())
    C[['id', 'matched']].to_csv(f'results/matched_flag_{name}.csv', index=False)
