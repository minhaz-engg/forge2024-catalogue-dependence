"""s04: one best-fit plane per 16A stage in each catalogue (all events, and the matched events only),
bootstrap uncertainty, and the angle between the two catalogues' planes."""
import numpy as np, pandas as pd
import fz

N = pd.read_csv('data/catalog_Niemz.csv'); Z = pd.read_csv('data/catalog_Zhu.csv'); P = pd.read_csv('results/matched_pairs.csv')
rng = np.random.default_rng(fz.SEED)
sets = {('Niemz', 'all'): N[['x', 'y', 'z', 'stage']], ('Zhu', 'all'): Z[['x', 'y', 'z', 'stage']],
        ('Niemz', 'matched'): P[['xN', 'yN', 'zN', 'stage']].set_axis(['x', 'y', 'z', 'stage'], axis=1),
        ('Zhu', 'matched'): P[['xZ', 'yZ', 'zZ', 'stage']].set_axis(['x', 'y', 'z', 'stage'], axis=1)}
rows = []
for (cat, ev), C in sets.items():
    for s in fz.STAGES_16A + ['16A-all']:
        X = C[C.stage.isin(fz.STAGES_16A)] if s == '16A-all' else C[C.stage == s]
        X = X[['x', 'y', 'z']].values
        if len(X) < 10: continue
        f = fz.plane_fit(X); a95, Pci = fz.bootstrap_plane(X, 1000, rng)
        rows.append(dict(cat=cat, events=ev, stage=s, **f, a95=a95, P_lo=Pci[0], P_hi=Pci[1]))
T = pd.DataFrame(rows); T.to_csv('results/stage_planes.csv', index=False)
pd.set_option('display.width', 200)
print(T[['cat', 'events', 'stage', 'n', 'l1', 'l2', 'l3', 'P', 'P_lo', 'P_hi', 'dipdir', 'dip', 'a95']].round(2).to_string(index=False))

# angle between the catalogues' poles for the same stage (same matched events) versus bootstrap uncertainty
cmp = []
for ev in ['all', 'matched']:
    for s in fz.STAGES_16A + ['16A-all']:
        a = T[(T.cat == 'Niemz') & (T.events == ev) & (T.stage == s)]; b = T[(T.cat == 'Zhu') & (T.events == ev) & (T.stage == s)]
        if len(a) == 0 or len(b) == 0: continue
        a, b = a.iloc[0], b.iloc[0]
        ang = fz.axial_angle([a.nx, a.ny, a.nz], [b.nx, b.ny, b.nz])
        cmp.append(dict(events=ev, stage=s, nN=a.n, nZ=b.n, PN=a.P, PZ=b.P, ddN=a.dipdir, dipN=a.dip, ddZ=b.dipdir,
                        dipZ=b.dip, angle=ang, a95N=a.a95, a95Z=b.a95, exceeds=ang > np.hypot(a.a95, b.a95)))
Cm = pd.DataFrame(cmp); Cm.to_csv('results/stage_plane_comparison.csv', index=False)
print(Cm.round(1).to_string(index=False))
m = Cm[(Cm.events == 'matched') & (Cm.stage != '16A-all')]
print(f'matched events, per stage: median pole angle {m.angle.median():.1f} deg (range {m.angle.min():.1f}-{m.angle.max():.1f}); '
      f'{m.exceeds.sum()}/{len(m)} exceed the combined bootstrap 95% cone; median P Niemz {m.PN.median():.2f}, Zhu {m.PZ.median():.2f}; '
      f'P higher in Niemz for {(m.PN > m.PZ).sum()}/{len(m)} stages')
