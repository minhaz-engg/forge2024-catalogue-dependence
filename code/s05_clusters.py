"""s05: unsupervised clustering (HDBSCAN; Bayesian Gaussian mixture) of the 16A hypocentres in each catalogue,
plane fits to clusters, cross-catalogue agreement on the matched events, and perturbation tests."""
import numpy as np, pandas as pd, json, time, sys
from sklearn.cluster import HDBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI
import fz

part = sys.argv[1] if len(sys.argv) > 1 else 'all'
P = pd.read_csv('results/matched_pairs.csv'); P = P[P.stage.isin(fz.STAGES_16A)].reset_index(drop=True)
XN = P[['xN', 'yN', 'zN']].values; XZ = P[['xZ', 'yZ', 'zZ']].values
R0 = (XZ - XN) - np.median(XZ - XN, 0)                      # empirical discrepancy vectors (offset removed)
SIG = P[['sx', 'sy', 'sz']].values                          # Zhu formal 1-sigma
rng = np.random.default_rng(fz.SEED)
MCS, MS = 15, 5                                             # HDBSCAN defaults used throughout

def hdb(X, mcs=MCS, ms=MS):
    return HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X)

def bgmm(X, seed):
    Xs = StandardScaler().fit_transform(X)
    g = BayesianGaussianMixture(n_components=40, weight_concentration_prior_type='dirichlet_process',
                                covariance_type='full', mean_precision_prior=0.8, max_iter=2000, tol=1e-4,
                                init_params='kmeans', n_init=1, random_state=seed).fit(Xs)
    lab = g.predict(Xs)
    return pd.Series(lab).map({k: i for i, k in enumerate(np.unique(lab))}).values

def cluster_planes(X, lab, minn=10):
    rows = []
    for k in np.unique(lab):
        if k < 0: continue
        idx = np.where(lab == k)[0]
        if len(idx) < minn: continue
        rows.append(dict(cluster=k, **fz.plane_fit(X[idx])))
    return pd.DataFrame(rows)

def ari_clustered(a, b):
    """ARI/AMI over events that are clustered (not noise) in both labelings, plus the full-set ARI with noise as one class."""
    m = (a >= 0) & (b >= 0)
    return ARI(a, b), ARI(a[m], b[m]) if m.sum() > 10 else np.nan, AMI(a[m], b[m]) if m.sum() > 10 else np.nan, m.mean()

out = {}
if part in ('all', 'hdb'):
    t0 = time.time()
    labN, labZ = hdb(XN), hdb(XZ)
    np.save('results/labels_hdb_N.npy', labN); np.save('results/labels_hdb_Z.npy', labZ)
    for tag, X, lab in [('N', XN, labN), ('Z', XZ, labZ)]:
        cp = cluster_planes(X, lab); cp['cat'] = tag; cp.to_csv(f'results/hdb_planes_{tag}.csv', index=False)
        out[f'hdb_{tag}_nclus'] = int(lab.max() + 1); out[f'hdb_{tag}_noise'] = float((lab < 0).mean())
        for thr in [0.6, 0.8, 0.9]:
            out[f'hdb_{tag}_frac_P{int(thr*10)}'] = float((cp.P >= thr).mean())
        out[f'hdb_{tag}_P_med'] = float(cp.P.median())
        print(f'HDBSCAN {tag}: {lab.max()+1} clusters, noise {np.mean(lab<0):.1%}, median cluster P {cp.P.median():.2f}, '
              f'P>=0.6: {(cp.P>=0.6).mean():.0%}, P>=0.9: {(cp.P>=0.9).mean():.0%}')
    a, ac, am, cov = ari_clustered(labN, labZ)
    out.update(hdb_ARI=a, hdb_ARI_cl=ac, hdb_AMI_cl=am, hdb_both_clustered=cov)
    out['stage_vs_hdbN_ARI'] = ARI(P.stage, labN); out['stage_vs_hdbZ_ARI'] = ARI(P.stage, labZ)
    print(f'cross-catalogue HDBSCAN agreement: ARI {a:.3f} (all), ARI {ac:.3f} / AMI {am:.3f} (events clustered in both, {cov:.0%})')
    print(f'ARI(stage, clusters): Niemz {out["stage_vs_hdbN_ARI"]:.3f}, Zhu {out["stage_vs_hdbZ_ARI"]:.3f}')

    # same events, other catalogue: refit each cluster's plane with the other catalogue's locations
    rows = []
    for tag, lab, Xa, Xb in [('N', labN, XN, XZ), ('Z', labZ, XZ, XN)]:
        for k in np.unique(lab[lab >= 0]):
            idx = np.where(lab == k)[0]
            if len(idx) < 10: continue
            fa, fb = fz.plane_fit(Xa[idx]), fz.plane_fit(Xb[idx])
            rows.append(dict(defined_in=tag, cluster=k, n=len(idx), P_own=fa['P'], P_other=fb['P'],
                             angle=fz.axial_angle([fa['nx'], fa['ny'], fa['nz']], [fb['nx'], fb['ny'], fb['nz']]),
                             l3_own=fa['l3'], l3_other=fb['l3'], ext_own=fa['l1'], ext_other=fb['l1']))
    S = pd.DataFrame(rows); S.to_csv('results/cluster_swap.csv', index=False)
    for tag in 'NZ':
        s = S[S.defined_in == tag]; sp = s[s.P_own >= 0.6]
        print(f'clusters defined in {tag}: n={len(s)}; P own {s.P_own.median():.2f} -> other {s.P_other.median():.2f}; '
              f'thickness l3 {s.l3_own.median():.0f} -> {s.l3_other.median():.0f} m; planar (P>=0.6) clusters: {len(sp)}, '
              f'median pole change {sp.angle.median():.0f} deg, still P>=0.6 in other catalogue: {(sp.P_other>=0.6).mean():.0%}')
        out[f'swap_{tag}_n'] = len(s); out[f'swap_{tag}_Pown'] = s.P_own.median(); out[f'swap_{tag}_Pother'] = s.P_other.median()
        out[f'swap_{tag}_l3own'] = s.l3_own.median(); out[f'swap_{tag}_l3other'] = s.l3_other.median()
        out[f'swap_{tag}_nplanar'] = len(sp); out[f'swap_{tag}_angle_planar'] = sp.angle.median()
        out[f'swap_{tag}_keep_planar'] = (sp.P_other >= 0.6).mean()

    # HDBSCAN parameter sensitivity of the cross-catalogue agreement
    sens = []
    for mcs in [10, 15, 25, 40]:
        for ms in [3, 5, 10]:
            a_, b_ = hdb(XN, mcs, ms), hdb(XZ, mcs, ms)
            r = ari_clustered(a_, b_)
            sens.append(dict(mcs=mcs, ms=ms, kN=a_.max() + 1, kZ=b_.max() + 1, noiseN=(a_ < 0).mean(), noiseZ=(b_ < 0).mean(),
                             ARI=r[0], ARI_cl=r[1], AMI_cl=r[2]))
    sens = pd.DataFrame(sens); sens.to_csv('results/hdb_sensitivity.csv', index=False)
    print(sens.round(3).to_string(index=False))
    out['hdb_sens_ARIcl_min'] = sens.ARI_cl.min(); out['hdb_sens_ARIcl_max'] = sens.ARI_cl.max()

    # perturbation tests (HDBSCAN, 100 realisations each)
    K = 100; res = []
    for k in range(K):
        Xj = XN + rng.normal(0, 1, XN.shape) * SIG                       # jitter by the Zhu formal sigmas
        Xr = XN + R0[rng.integers(0, len(R0), len(R0))]                  # add resampled empirical discrepancies
        Xz = XZ + rng.normal(0, 1, XZ.shape) * SIG                       # Zhu jittered by its own formal sigmas
        res.append(dict(k=k, ari_formal_N=ari_clustered(labN, hdb(Xj))[1], ari_emp_N=ari_clustered(labN, hdb(Xr))[1],
                        ari_formal_Z=ari_clustered(labZ, hdb(Xz))[1]))
    res = pd.DataFrame(res); res.to_csv('results/perturbation_ari.csv', index=False)
    for c in ['ari_formal_N', 'ari_emp_N', 'ari_formal_Z']:
        q = res[c].quantile([0.05, 0.5, 0.95]).values
        out[f'{c}_med'] = q[1]; out[f'{c}_p05'] = q[0]; out[f'{c}_p95'] = q[2]
        print(f'{c}: median {q[1]:.3f} (5-95%: {q[0]:.3f}-{q[2]:.3f})')
    print(f'hdb part done in {time.time()-t0:.0f} s')
    json.dump({k: float(v) for k, v in out.items()}, open('results/clusters_hdb.json', 'w'), indent=1)

if part in ('all', 'bgmm'):
    t0 = time.time(); rows = []; outb = {}
    for seed in range(10):
        a_, b_ = bgmm(XN, seed), bgmm(XZ, seed)
        for tag, X, lab in [('N', XN, a_), ('Z', XZ, b_)]:
            cp = cluster_planes(X, lab, minn=5)
            rows.append(dict(seed=seed, cat=tag, k=lab.max() + 1, P_med=cp.P.median(), frac_P9=(cp.P >= 0.9).mean(),
                             frac_P6=(cp.P >= 0.6).mean()))
        rows[-1]['ARI'] = ARI(a_, b_); rows[-2]['ARI'] = rows[-1]['ARI']
        if seed == 0:
            np.save('results/labels_bgmm_N.npy', a_); np.save('results/labels_bgmm_Z.npy', b_)
    B = pd.DataFrame(rows); B.to_csv('results/bgmm_runs.csv', index=False)
    print(B.round(3).to_string(index=False))
    g = B.groupby('cat').median(numeric_only=True)
    for tag in 'NZ':
        outb[f'bgmm_{tag}_k_med'] = g.loc[tag, 'k']; outb[f'bgmm_{tag}_Pmed'] = g.loc[tag, 'P_med']
        outb[f'bgmm_{tag}_fracP9'] = g.loc[tag, 'frac_P9']; outb[f'bgmm_{tag}_fracP6'] = g.loc[tag, 'frac_P6']
    outb['bgmm_ARI_med'] = B.ARI.median(); outb['bgmm_ARI_min'] = B.ARI.min(); outb['bgmm_ARI_max'] = B.ARI.max()
    # run-to-run self agreement within each catalogue (seed 0 vs others)
    selfa = {'N': [], 'Z': []}
    a0, b0 = np.load('results/labels_bgmm_N.npy'), np.load('results/labels_bgmm_Z.npy')
    for seed in range(1, 10):
        selfa['N'].append(ARI(a0, bgmm(XN, seed))); selfa['Z'].append(ARI(b0, bgmm(XZ, seed)))
    for tag in 'NZ':
        outb[f'bgmm_self_{tag}_med'] = float(np.median(selfa[tag]))
    print({k: round(float(v), 3) for k, v in outb.items()}, f'bgmm done in {time.time()-t0:.0f} s')
    json.dump({k: float(v) for k, v in outb.items()}, open('results/clusters_bgmm.json', 'w'), indent=1)
