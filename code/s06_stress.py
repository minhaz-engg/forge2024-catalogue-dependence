"""s06: slip tendency, dilation tendency and Coulomb critical pore-pressure increase of the planes from each catalogue,
under a Monte-Carlo sample of the published Utah FORGE stress state (normal-faulting, Sv > SH > Sh)."""
import numpy as np, pandas as pd, json
import fz

mc = fz.stress_mc(1000)
MC = list(zip(mc.Sh.values, mc.SH.values, mc.az.values))
def evaluate(T, tag):
    rows = []
    for _, p in T.iterrows():
        n = np.array([p.nx, p.ny, p.nz]); ts, td, dp = [], [], []
        for sh, sH, az in MC:
            S, _ = fz.stress_tensor(p.cz, Sh_g=sh, SH_g=sH, az_SH=az)
            a, b, c = fz.tendencies(n, S); ts.append(a); td.append(b); dp.append(c)
        rows.append(dict(set=tag, cat=p['cat'], id=p['stage'] if 'stage' in p.index else p['cluster'], n=p['n'], P=p['P'],
                         dipdir=p['dipdir'], dip=p['dip'], depth=p['cz'], Ts=np.median(ts), Ts_lo=np.percentile(ts, 5), Ts_hi=np.percentile(ts, 95),
                         Td=np.median(td), dPc=np.median(dp), dPc_lo=np.percentile(dp, 5), dPc_hi=np.percentile(dp, 95)))
    return pd.DataFrame(rows)

SP = pd.read_csv('results/stage_planes.csv')
SP = SP[(SP.events == 'matched') & SP.stage.isin(fz.STAGES_16A)].replace({'cat': {'Niemz': 'N', 'Zhu': 'Z'}})
H = pd.concat([pd.read_csv(f'results/hdb_planes_{c}.csv') for c in 'NZ'])
res = pd.concat([evaluate(SP, 'stage'), evaluate(H, 'hdbscan')], ignore_index=True)
res.to_csv('results/stress_tendencies.csv', index=False)
pd.set_option('display.width', 200)
st = res[res.set == 'stage'].pivot(index='id', columns='cat', values=['dipdir', 'dip', 'Ts', 'dPc']).reindex(fz.STAGES_16A)
print(st.round(2).to_string())
out = {}
for c in 'NZ':
    s = res[(res.set == 'stage') & (res.cat == c)]; h = res[(res.set == 'hdbscan') & (res.cat == c)]
    out[f'stage_{c}_Ts_med'] = s.Ts.median(); out[f'stage_{c}_Ts_min'] = s.Ts.min(); out[f'stage_{c}_Ts_max'] = s.Ts.max()
    out[f'stage_{c}_dPc_med'] = s.dPc.median(); out[f'stage_{c}_dPc_min'] = s.dPc.min(); out[f'stage_{c}_dPc_max'] = s.dPc.max()
    out[f'hdb_{c}_Ts_med'] = h.Ts.median(); out[f'hdb_{c}_frac_Ts06'] = (h.Ts >= 0.6).mean()
    out[f'hdb_{c}_dPc_med'] = h.dPc.median(); out[f'hdb_{c}_dPc_p10'] = h.dPc.quantile(0.1); out[f'hdb_{c}_dPc_p90'] = h.dPc.quantile(0.9)
    w = h.n / h.n.sum(); out[f'hdb_{c}_frac_events_Ts06'] = float(w[h.Ts >= 0.6].sum())
d = st['dPc']['Z'] - st['dPc']['N']; out['stage_dPc_absdiff_med'] = d.abs().median(); out['stage_dPc_absdiff_max'] = d.abs().max()
dt = st['Ts']['Z'] - st['Ts']['N']; out['stage_Ts_absdiff_med'] = dt.abs().median(); out['stage_Ts_absdiff_max'] = dt.abs().max()
# optimally oriented plane for reference (median stress)
S, Pp = fz.stress_tensor(2400.0)
best = max(((fz.tendencies(np.array([np.sin(np.radians(t)) * np.sin(np.radians(dp)), np.cos(np.radians(t)) * np.sin(np.radians(dp)),
             np.cos(np.radians(dp))]), S)[0], (t + 180) % 360, dp) for t in range(0, 360, 2) for dp in range(0, 91, 1)))
out['opt_Ts'], out['opt_dipdir'], out['opt_dip'] = best
out['dPc_opt'] = fz.tendencies(np.array([np.sin(np.radians(best[1] - 180)) * np.sin(np.radians(best[2])),
                                          np.cos(np.radians(best[1] - 180)) * np.sin(np.radians(best[2])),
                                          np.cos(np.radians(best[2]))]), S)[2]
print({k: round(float(v), 3) for k, v in out.items()})
json.dump({k: float(v) for k, v in out.items()}, open('results/stress.json', 'w'), indent=1)
