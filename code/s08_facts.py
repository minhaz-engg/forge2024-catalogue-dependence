"""s08: collect every number quoted in the manuscript into results/facts.json."""
import json, glob, pandas as pd, numpy as np
import fz

F = {}
for f in ['discrepancy', 'clusters_hdb', 'clusters_bgmm', 'stress', 'overlap']:
    F.update(json.load(open(f'results/{f}.json')))
N = pd.read_csv('data/catalog_Niemz.csv'); Z = pd.read_csv('data/catalog_Zhu.csv'); P = pd.read_csv('results/matched_pairs.csv')
F.update(nN=len(N), nZ=len(Z), nN16A=int(N.stage.isin(fz.STAGES_16A).sum()), nZ16A=int(Z.stage.isin(fz.STAGES_16A).sum()),
         n_match=len(P), n_match16A=int(P.stage.isin(fz.STAGES_16A).sum()), frac_match_N=len(P) / len(N), frac_match_Z=len(P) / len(Z),
         dt_offset_ms=1000 * float(np.median(P.dt)), dt_abs_p99_ms=1000 * float(np.percentile(np.abs(P.dt - P.dt.median()), 99)))
tol = pd.read_csv('results/match_tolerance.csv'); F['match_tol_min'] = int(tol.n.min()); F['match_tol_max'] = int(tol.n.max())
for name, C in [('N', N), ('Z', Z)]:
    fl = pd.read_csv(f'results/matched_flag_{"Niemz" if name=="N" else "Zhu"}.csv')
    C = C.merge(fl, on='id'); q = pd.qcut(C.M, 5)
    g = C.groupby(q, observed=True).matched.mean().values
    F[f'matchfrac_{name}_q1'] = g[0]; F[f'matchfrac_{name}_q5'] = g[-1]
SP = pd.read_csv('results/stage_plane_comparison.csv'); m = SP[(SP.events == 'matched') & (SP.stage != '16A-all')]
a = SP[(SP.events == 'matched') & (SP.stage == '16A-all')].iloc[0]
F.update(stage_angle_med=m.angle.median(), stage_angle_min=m.angle.min(), stage_angle_max=m.angle.max(),
         stage_angle_gt30=int((m.angle > 30).sum()), stage_exceeds=int(m.exceeds.sum()),
         stage_PN_med=m.PN.median(), stage_PZ_med=m.PZ.median(), stage_PN_gt_PZ=int((m.PN > m.PZ).sum()),
         all16A_angle=a.angle, all16A_PN=a.PN, all16A_PZ=a.PZ, all16A_ddN=a.ddN, all16A_dipN=a.dipN, all16A_ddZ=a.ddZ, all16A_dipZ=a.dipZ)
T = pd.read_csv('results/stage_planes.csv'); t = T[(T.events == 'matched') & T.stage.isin(fz.STAGES_16A)]
early = ['16A-3R', '16A-4', '16A-5', '16A-6A', '16A-6B']
for c, k in [('Niemz', 'N'), ('Zhu', 'Z')]:
    e = t[(t.cat == c) & t.stage.isin(early)]
    F[f'early_{k}_l3_min'] = e.l3.min(); F[f'early_{k}_l3_max'] = e.l3.max(); F[f'early_{k}_P_min'] = e.P.min(); F[f'early_{k}_P_max'] = e.P.max()
    F[f'early_{k}_dd_min'] = e.dipdir.min(); F[f'early_{k}_dd_max'] = e.dipdir.max(); F[f'early_{k}_dip_min'] = e.dip.min(); F[f'early_{k}_dip_max'] = e.dip.max()
json.dump({k: (float(v) if not isinstance(v, str) else v) for k, v in F.items()}, open('results/facts.json', 'w'), indent=1)
for k, v in F.items(): print(f'{k}: {v:.4g}' if isinstance(v, (int, float, np.floating, np.integer)) else f'{k}: {v}')
