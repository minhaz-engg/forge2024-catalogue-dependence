"""s03: location and magnitude discrepancy of the matched events."""
import numpy as np, pandas as pd, json
import fz

P = pd.read_csv('results/matched_pairs.csv')
rng = np.random.default_rng(fz.SEED)
A = P[['xN', 'yN', 'zN']].values; B = P[['xZ', 'yZ', 'zZ']].values
D = B - A                                                   # Zhu minus Niemz
out = {}

# 1) rigid offset (median) with bootstrap CI
off = np.median(D, 0)
bo = np.array([np.median(D[rng.integers(0, len(D), len(D))], 0) for _ in range(1000)])
ci = np.percentile(bo, [2.5, 97.5], 0)
for k, c in enumerate('xyz'):
    out[f'offset_{c}'] = off[k]; out[f'offset_{c}_lo'] = ci[0, k]; out[f'offset_{c}_hi'] = ci[1, k]
print('median offset Zhu-Niemz (E,N,Z) m:', off.round(1), 'CI', ci.round(1).tolist())

def spread(R, tag):
    mad = 1.4826 * np.median(np.abs(R - np.median(R, 0)), 0)
    d3 = np.linalg.norm(R - np.median(R, 0), axis=1)
    l, v = fz.pca(R)
    ax = v[:, 0]; ax = ax if ax[2] >= 0 else -ax
    tr = np.degrees(np.arctan2(ax[0], ax[1])) % 360; pl = np.degrees(np.arcsin(abs(ax[2])))
    r = {f'{tag}_madE': mad[0], f'{tag}_madN': mad[1], f'{tag}_madZ': mad[2], f'{tag}_d3_med': np.median(d3),
         f'{tag}_d3_p90': np.percentile(d3, 90), f'{tag}_sd1': np.sqrt(l[0]), f'{tag}_sd2': np.sqrt(l[1]),
         f'{tag}_sd3': np.sqrt(l[2]), f'{tag}_ax1_trend': tr, f'{tag}_ax1_plunge': pl}
    print(tag, {k: round(float(x), 1) for k, x in r.items()})
    return r

R0 = D - off
out.update(spread(R0, 'shift'))

# 2) similarity transform (Umeyama 1991): rotation + scale + translation, robust to outliers by iterative trimming
def umeyama(X, Y, scale=True):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    d = np.sign(np.linalg.det(U @ Vt)); Dm = np.diag([1, 1, d])
    Rm = U @ Dm @ Vt
    s = (S * np.diag(Dm)).sum() / Xc.var(0).sum() if scale else 1.0
    return s, Rm, my - s * Rm @ mx

def robust_fit(X, Y, affine=False, it=5, keep=0.8):
    idx = np.arange(len(X))
    for _ in range(it):
        if affine:
            Xa = np.c_[X[idx], np.ones(len(idx))]; M, *_ = np.linalg.lstsq(Xa, Y[idx], rcond=None)
            f = lambda Q: np.c_[Q, np.ones(len(Q))] @ M
        else:
            s, Rm, t = umeyama(X[idx], Y[idx]); f = lambda Q, s=s, Rm=Rm, t=t: s * Q @ Rm.T + t
        r = np.linalg.norm(Y - f(X), axis=1); idx = np.argsort(r)[:int(keep * len(X))]
    return f, (s, Rm) if not affine else (None, M)

f_sim, (s, Rm) = robust_fit(A, B)
ang = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
print(f'similarity transform: scale {s:.3f}, rotation {ang:.1f} deg')
out['sim_scale'] = s; out['sim_rot_deg'] = ang
out.update(spread(B - f_sim(A), 'sim'))
f_aff, _ = robust_fit(A, B, affine=True)
out.update(spread(B - f_aff(A), 'aff'))

# 3) formal uncertainty of the Zhu locations
fs = P[['sx', 'sy', 'sz']].median().values
out.update(dict(zhu_sigma_x_med=fs[0], zhu_sigma_y_med=fs[1], zhu_sigma_z_med=fs[2]))
print('Zhu formal sigma medians (m):', fs.round(1))

# 4) dependence on magnitude and stage
d3 = np.linalg.norm(R0 - np.median(R0, 0), axis=1)
P['d3'] = d3; P['dE'] = R0[:, 0]; P['dN'] = R0[:, 1]; P['dZ'] = R0[:, 2]
P.to_csv('results/matched_pairs.csv', index=False)
q = pd.qcut(P.MN, 4)
print('median 3-D discrepancy by Niemz magnitude quartile:', P.groupby(q, observed=True).d3.median().round(0).to_dict())
rho = pd.Series(d3).corr(P.MN, method='spearman'); out['rho_d3_M'] = rho
print('Spearman(d3, M_Niemz) =', round(rho, 3))
bys = P[P.stage.isin(fz.STAGES_16A)].groupby('stage').agg(n=('d3', 'size'), d3_med=('d3', 'median'),
        dE_med=('dE', 'median'), dN_med=('dN', 'median'), dZ_med=('dZ', 'median')).reindex(fz.STAGES_16A)
print(bys.round(0)); bys.to_csv('results/discrepancy_by_stage.csv')
out['stage_shift_range_N'] = float(bys.dN_med.max() - bys.dN_med.min())
out['stage_shift_range_E'] = float(bys.dE_med.max() - bys.dE_med.min())

# 5) magnitudes: orthogonal (total least squares) regression MZ = a + b*MN
x, y = P.MN.values, P.MZ.values
C = np.cov(x, y); w, v = np.linalg.eigh(C); bb = v[1, 1] / v[0, 1]; aa = y.mean() - bb * x.mean()
res = y - (aa + bb * x)
out.update(dict(mag_tls_a=aa, mag_tls_b=bb, mag_res_sd=res.std(ddof=2), mag_r=np.corrcoef(x, y)[0, 1],
                mag_diff_med=np.median(y - x)))
print(f'M_Zhu = {aa:.2f} + {bb:.2f} M_Niemz (TLS), r = {out["mag_r"]:.2f}, residual sd {out["mag_res_sd"]:.2f}')

json.dump({k: float(v) for k, v in out.items()}, open('results/discrepancy.json', 'w'), indent=1)
