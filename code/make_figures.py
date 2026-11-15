"""Figures 1-9 (+S1). Run after s01-s08."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import fz

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'savefig.dpi': 400, 'figure.dpi': 120, 'axes.grid': False, 'font.family': 'DejaVu Sans'})
CN, CZ, GRAY, INK = '#2a78d6', '#eb6834', '#b9b8b2', '#52514e'
LAB = {'N': 'Surface-network catalogue (Niemz)', 'Z': 'Phase-picking catalogue (Zhu)'}
W2 = 6.85
STG = fz.STAGES_16A; SHORT = [s.replace('16A-', '') for s in STG]
SEQ = plt.get_cmap('Blues')(np.linspace(0.35, 1.0, len(STG)))      # one hue, light (early) -> dark (late)
F = json.load(open('results/facts.json'))
N = pd.read_csv('data/catalog_Niemz.csv'); Z = pd.read_csv('data/catalog_Zhu.csv'); P = pd.read_csv('results/matched_pairs.csv')
P16 = P[P.stage.isin(STG)].reset_index(drop=True)
off = np.array([F['offset_x'], F['offset_y'], F['offset_z']])

def tag(ax, s, x=-0.16): ax.text(x, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')
def save(fig, name): fig.savefig(f'figures/{name}.png', bbox_inches='tight'); plt.close(fig); print('saved', name)
AZ = np.radians(104.0)                                    # cross-section azimuth, normal to the ~N14E strike
def along(x, y): return x * np.sin(AZ) + y * np.cos(AZ)

# ---------------------------------------------------------- Fig 1 workflow (schematic; to be redrawn in Figma)
fig, ax = plt.subplots(figsize=(W2, 2.3)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
boxes = [(0.1, 'Two public catalogues\nsame stimulation\n(surface network;\nML phase picking)'),
         (2.1, 'Origin-time\nmatching\n(one-to-one)'), (4.1, 'Location & magnitude\ndiscrepancy of\nthe same events'),
         (6.1, 'Interpretation in\neach catalogue:\nplanes, clusters,\nstress, overlap'), (8.1, 'Which conclusions\nsurvive the\ncatalogue choice?')]
for i, (x, t) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x, 0.6), 1.75, 1.8, boxstyle='round,pad=0.03,rounding_size=0.12', fc='#f3f2ef', ec=INK, lw=0.7))
    ax.text(x + 0.875, 1.5, t, ha='center', va='center', fontsize=7.2)
    if i < 4: ax.annotate('', (x + 1.97, 1.5), (x + 1.78, 1.5), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK))
save(fig, 'Fig01_workflow')

# ---------------------------------------------------------- Fig 2 overview
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.6), gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, (k, C) in enumerate([('N', N), ('Z', Z)]):
    C = C[C.stage.isin(STG)]
    for i, s in enumerate(STG):
        c = C[C.stage == s]
        axs[0, j].scatter(c.x, c.y, s=2.2, color=SEQ[i], lw=0, label=SHORT[i])
        axs[1, j].scatter(along(c.x, c.y), c.z, s=2.2, color=SEQ[i], lw=0)
    axs[0, j].set_title(LAB[k], color=INK); axs[0, j].set_aspect('equal'); axs[1, j].set_aspect('equal')
    axs[0, j].set_xlabel('Easting from 16A wellhead (m)'); axs[0, j].set_ylabel('Northing (m)')
    axs[1, j].set_xlabel('Distance along azimuth 104° (m)'); axs[1, j].set_ylabel('Depth below wellhead (m)'); axs[1, j].invert_yaxis()
    for a in axs[:, j]: a.set_xlim(-200, 1900) if a in axs[0] else None
for a in axs[0]: a.set_xlim(-100, 1800); a.set_ylim(-1100, 550)
for a in axs[1]: a.set_xlim(-300, 2000); a.set_ylim(3300, 1700)
h = axs[0, 0].legend(title='16A stage', markerscale=3.5, ncol=1, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
for a, t in zip(axs.flat, 'abcd'): tag(a, t)
fig.tight_layout(); save(fig, 'Fig02_overview')

# ---------------------------------------------------------- Fig 3 matching
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
dt = (P.dt - P.dt.median()) * 1000
axs[0].hist(dt.clip(-150, 150), bins=np.arange(-150, 151, 5), color=INK, lw=0)
axs[0].set_xlabel('Origin-time difference − median (ms)'); axs[0].set_ylabel('Matched pairs')
for k, C, col in [('N', N, CN), ('Z', Z, CZ)]:
    fl = pd.read_csv(f'results/matched_flag_{"Niemz" if k=="N" else "Zhu"}.csv'); C = C.merge(fl, on='id')
    q = pd.qcut(C.M, 8); g = C.groupby(q, observed=True).agg(M=('M', 'median'), f=('matched', 'mean'))
    axs[1].plot(g.M, g.f, '-o', color=col, ms=3.5, lw=1.5, label='Niemz' if k == 'N' else 'Zhu')
axs[1].set_xlabel('Magnitude in own catalogue'); axs[1].set_ylabel('Fraction matched'); axs[1].set_ylim(0, 1.02); axs[1].legend(frameon=False)
axs[2].scatter(P.MN, P.MZ, s=2, color=INK, lw=0, alpha=0.6)
xx = np.linspace(P.MN.min(), P.MN.max(), 10); axs[2].plot(xx, F['mag_tls_a'] + F['mag_tls_b'] * xx, color=CZ, lw=1.5)
axs[2].plot(xx, xx, ls=':', color=GRAY, lw=1); axs[2].set_xlabel('Magnitude, Niemz'); axs[2].set_ylabel('Magnitude, Zhu')
axs[2].text(0.04, 0.95, f"M$_Z$ = {F['mag_tls_a']:.2f} + {F['mag_tls_b']:.2f} M$_N$\nr = {F['mag_r']:.2f}", transform=axs[2].transAxes, va='top', fontsize=7)
for a, t in zip(axs, 'abc'): tag(a, t)
fig.tight_layout(); save(fig, 'Fig03_matching')

# ---------------------------------------------------------- Fig 4 discrepancy
fig = plt.figure(figsize=(W2, 4.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
a0 = fig.add_subplot(gs[0, :2]); a1 = fig.add_subplot(gs[0, 2]); a2 = fig.add_subplot(gs[1, :])
rng = np.random.default_rng(1); sub = rng.choice(len(P16), 500, replace=False); S = P16.iloc[sub]
a0.quiver(S.xN, S.yN, S.dE, S.dN, angles='xy', scale_units='xy', scale=1, width=0.0025, color=INK, alpha=0.7)
a0.scatter(S.xN, S.yN, s=2, color=CN, lw=0, zorder=3)
a0.set_aspect('equal'); a0.set_xlabel('Easting (m), Niemz location'); a0.set_ylabel('Northing (m)')
E = np.r_[S.xN, S.xN + S.dE]; Nn = np.r_[S.yN, S.yN + S.dN]
a0.set_xlim(np.percentile(E, 1) - 50, np.percentile(E, 99) + 50); a0.set_ylim(np.percentile(Nn, 1) - 50, np.percentile(Nn, 99) + 50)
bins = np.arange(-600, 601, 20)
for c, lab in [('dE', 'East'), ('dN', 'North'), ('dZ', 'Depth')]:
    a1.hist(P16[c], bins=bins, histtype='step', lw=1.3, label=lab,
            color={'dE': CN, 'dN': CZ, 'dZ': '#1baf7a'}[c])
a1.axvspan(-F['zhu_sigma_y_med'], F['zhu_sigma_y_med'], color=GRAY, alpha=0.5, lw=0, label='Zhu formal ±1σ (N)')
a1.set_xlabel('Discrepancy (m)'); a1.set_ylabel('Events'); a1.legend(frameon=False, loc='upper left', fontsize=6.3, bbox_to_anchor=(0.0, 1.12)); a1.set_xlim(-600, 600)
bs = pd.read_csv('results/discrepancy_by_stage.csv').set_index('stage').reindex(STG)
x = np.arange(len(STG)); w = 0.26
for i, (c, col, lab) in enumerate([('dE_med', CN, 'East'), ('dN_med', CZ, 'North'), ('dZ_med', '#1baf7a', 'Depth')]):
    a2.bar(x + (i - 1) * w, bs[c], w * 0.9, color=col, label=lab)
a2.plot(x, bs.d3_med, 'o', color=INK, ms=4, label='Median 3-D distance')
a2.axhline(0, color=INK, lw=0.6); a2.set_xticks(x, SHORT); a2.set_xlabel('16A stage'); a2.set_ylabel('Median discrepancy (m)')
a2.set_ylim(-250, 360); a2.legend(frameon=False, ncol=4, loc='upper left')
for a, t in zip([a0, a1, a2], 'abc'): tag(a, t)
fig.tight_layout(); save(fig, 'Fig04_discrepancy')

# ---------------------------------------------------------- stereonet helpers
def proj(nx, ny, nz):
    tr = np.arctan2(nx, ny); pl = np.arcsin(np.clip(np.abs(nz), 0, 1))
    r = np.sqrt(2) * np.sin((np.pi / 2 - pl) / 2)
    return r * np.sin(tr), r * np.cos(tr)
def net(ax):
    t = np.linspace(0, 2 * np.pi, 361); ax.plot(np.cos(t), np.sin(t), color=INK, lw=0.7)
    ax.plot([0, 0], [-1, 1], color=GRAY, lw=0.4); ax.plot([-1, 1], [0, 0], color=GRAY, lw=0.4)
    ax.text(0, 1.04, 'N', ha='center', va='bottom', fontsize=7); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.15, 1.15)
    a = np.radians(20); ax.plot([-np.sin(a), np.sin(a)], [-np.cos(a), np.cos(a)], ls='--', color=INK, lw=0.7)
    ax.text(np.sin(a) * 1.05, np.cos(a) * 1.05, 'S$_{Hmax}$', fontsize=6.5, ha='left')

# ---------------------------------------------------------- Fig 5 stage planes
T = pd.read_csv('results/stage_planes.csv'); T = T[(T.events == 'matched') & T.stage.isin(STG)]
fig = plt.figure(figsize=(W2, 2.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1])
a0 = fig.add_subplot(gs[0]); net(a0)
for c, col, mk in [('Niemz', CN, 'o'), ('Zhu', CZ, 's')]:
    t = T[T.cat == c]; X, Y = proj(t.nx.values, t.ny.values, t.nz.values)
    a0.scatter(X, Y, s=22, marker=mk, color=col, ec='white', lw=0.6, zorder=3, label=c)
    for xi, yi, s in zip(X, Y, t.stage): a0.text(xi + 0.04, yi + 0.03, s.replace('16A-', ''), fontsize=5.5, color=INK)
a0.legend(frameon=False, loc='lower left', bbox_to_anchor=(-0.05, -0.12), ncol=2)
a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2]); x = np.arange(len(STG))
for c, col, d in [('Niemz', CN, -0.15), ('Zhu', CZ, 0.15)]:
    t = T[T.cat == c].set_index('stage').reindex(STG)
    a1.errorbar(x + d, t.P, yerr=[t.P - t.P_lo, t.P_hi - t.P], fmt='o', ms=4, color=col, lw=1, capsize=0, label=c)
    a2.bar(x + d, t.l3, 0.28, color=col, label=c)
a1.set_ylabel('Planarity 1 − 2λ$_3$/(λ$_1$+λ$_2$)'); a1.set_xticks(x, SHORT, rotation=0); a1.set_ylim(0, 1.03); a1.set_xlabel('16A stage')
a2.set_ylabel('Thickness √λ$_3$ (m)'); a2.set_xticks(x, SHORT); a2.set_xlabel('16A stage'); a1.legend(frameon=False, loc='lower left')
for a, t in zip([a0, a1, a2], 'abc'): tag(a, t)
fig.tight_layout(); save(fig, 'Fig05_stage_planes')

# ---------------------------------------------------------- Fig 6 cross-sections of the same events
show = ['16A-3R', '16A-6B', '16A-8']
fig, axs = plt.subplots(2, 3, figsize=(W2, 4.4))
TN = T[T.cat == 'Niemz'].set_index('stage')
for j, s in enumerate(show):
    q = P16[P16.stage == s]; r = TN.loc[s]
    nrm = np.array([r.nx, r.ny, 0.0]); nrm /= np.linalg.norm(nrm); stk = np.array([-nrm[1], nrm[0], 0])
    for i, (k, cols, col) in enumerate([('N', ['xN', 'yN', 'zN'], CN), ('Z', ['xZ', 'yZ', 'zZ'], CZ)]):
        X = q[cols].values - q[cols].values.mean(0)
        u = X @ nrm; z = X[:, 2]
        axs[i, j].scatter(u, z, s=4, color=col, lw=0)
        axs[i, j].set_xlim(-300, 300); axs[i, j].set_ylim(300, -300); axs[i, j].set_aspect('equal')
        if i == 0: axs[i, j].set_title(f'Stage {s.replace("16A-","")}, n = {len(q)}', color=INK)
        axs[i, j].set_xlabel('Horizontal, normal to strike (m)' if i == 1 else '')
        if j == 0: axs[i, j].set_ylabel(f'{"Niemz" if k=="N" else "Zhu"}\nRelative depth (m)')
for a, t in zip(axs.flat, 'abcdef'): tag(a, t, -0.3)
fig.tight_layout(); save(fig, 'Fig06_cross_sections')

# ---------------------------------------------------------- Fig 7 clustering
labN, labZ = np.load('results/labels_hdb_N.npy'), np.load('results/labels_hdb_Z.npy')
fig = plt.figure(figsize=(W2, 5.0)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
for j, (k, lab, cols, col) in enumerate([('N', labN, ['xN', 'yN', 'zN'], CN), ('Z', labZ, ['xZ', 'yZ', 'zZ'], CZ)]):
    a = fig.add_subplot(gs[0, j]); X = P16[cols].values
    a.scatter(X[lab < 0, 0], X[lab < 0, 1], s=2, color=GRAY, lw=0, label='Noise')
    a.scatter(X[lab >= 0, 0], X[lab >= 0, 1], s=2, color=col, lw=0, label='In a cluster')
    for c in np.unique(lab[lab >= 0]):
        Y = X[lab == c, :2]; cv = np.cov(Y.T); w_, v_ = np.linalg.eigh(cv)
        ang = np.degrees(np.arctan2(v_[1, 1], v_[0, 1]))
        a.add_patch(Ellipse(Y.mean(0), 4 * np.sqrt(w_[1]), 4 * np.sqrt(w_[0]), angle=ang, fill=False, ec=INK, lw=0.6))
    a.set_aspect('equal'); a.set_xlabel('Easting (m)'); a.set_ylabel('Northing (m)')
    a.set_title(f'{"Niemz" if k=="N" else "Zhu"}: {lab.max()+1} HDBSCAN clusters, {np.mean(lab<0):.0%} noise', color=INK)
    a.legend(frameon=False, markerscale=4, loc='upper right', fontsize=6.3); tag(a, 'ab'[j])
R = pd.read_csv('results/perturbation_ari.csv'); a = fig.add_subplot(gs[1, 0])
bins = np.linspace(0, 1, 41)
a.hist(R.ari_formal_Z, bins, color=CZ, alpha=0.85, label='Zhu + its formal errors', lw=0)
a.hist(R.ari_formal_N, bins, color=CN, alpha=0.85, label='Niemz + Zhu-size formal errors', lw=0)
a.hist(R.ari_emp_N, bins, color=INK, alpha=0.85, label='Niemz + observed discrepancy', lw=0)
a.axvline(F['hdb_ARI_cl'], color=INK, ls='--', lw=1); a.text(F['hdb_ARI_cl'] + 0.02, a.get_ylim()[1] * 0.45, 'Niemz vs\nZhu', fontsize=6.5)
a.set_xlabel('Adjusted Rand index with unperturbed clusters'); a.set_ylabel('Realisations'); a.legend(frameon=False, fontsize=6.3, loc='upper left', bbox_to_anchor=(0.22, 1.0)); tag(a, 'c')
Hs = pd.read_csv('results/hdb_sensitivity.csv'); a = fig.add_subplot(gs[1, 1]); B = pd.read_csv('results/bgmm_runs.csv')
for ms, mk in [(3, 'o'), (5, 's'), (10, '^')]:
    h = Hs[Hs.ms == ms]; a.plot(h.mcs, h.ARI_cl, '-' + mk, color=INK, ms=4, lw=1, label=f'HDBSCAN, min_samples={ms}')
a.axhspan(B.ARI.min(), B.ARI.max(), color=GRAY, alpha=0.6, lw=0, label='Gaussian mixture, 10 seeds')
a.set_xlabel('HDBSCAN min_cluster_size'); a.set_ylabel('Niemz–Zhu ARI'); a.set_ylim(0, 1); a.legend(frameon=False, fontsize=6.3); tag(a, 'd')
fig.tight_layout(); save(fig, 'Fig07_clusters')

# ---------------------------------------------------------- Fig 8 stress
St = pd.read_csv('results/stress_tendencies.csv')
fig = plt.figure(figsize=(W2, 2.9)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
a0 = fig.add_subplot(gs[0])
gx, gy = np.meshgrid(np.linspace(-1, 1, 241), np.linspace(-1, 1, 241)); rr = np.hypot(gx, gy)
pl = np.pi / 2 - 2 * np.arcsin(np.clip(rr / np.sqrt(2), 0, 1)); tr = np.arctan2(gx, gy)
nn = np.stack([np.sin(tr) * np.cos(pl), np.cos(tr) * np.cos(pl), np.sin(pl)], -1)
Sm, _ = fz.stress_tensor(2400.0); Ts = np.full(rr.shape, np.nan)
for i in range(rr.shape[0]):
    for j in range(rr.shape[1]):
        if rr[i, j] <= 1: Ts[i, j] = fz.tendencies(nn[i, j], Sm)[0]
cs = a0.contourf(gx, gy, Ts, levels=np.linspace(0, 0.4, 9), cmap='Greys', alpha=0.75)
net(a0)
for c, col, mk in [('N', CN, 'o'), ('Z', CZ, 's')]:
    h = St[(St.set == 'hdbscan') & (St.cat == c)]
    tt = np.radians((h.dipdir.values + 180) % 360); pp = np.radians(90 - h.dip.values)
    X, Y = proj(np.sin(tt) * np.cos(pp), np.cos(tt) * np.cos(pp), np.sin(pp))
    a0.scatter(X, Y, s=10, marker=mk, color=col, ec='white', lw=0.4, zorder=3, label=f'{"Niemz" if c=="N" else "Zhu"} cluster poles')
cb = fig.colorbar(cs, ax=a0, fraction=0.04, pad=0.02); cb.set_label('Slip tendency (median stress)', fontsize=7)
a0.legend(frameon=False, loc='lower left', bbox_to_anchor=(-0.08, -0.14), fontsize=6.3); tag(a0, 'a')
a1 = fig.add_subplot(gs[1]); s = St[St.set == 'stage']; x = np.arange(len(STG))
for c, col, d in [('N', CN, -0.15), ('Z', CZ, 0.15)]:
    t = s[s.cat == c].set_index('id').reindex(STG)
    a1.errorbar(x + d, t.dPc, yerr=[t.dPc - t.dPc_lo, t.dPc_hi - t.dPc], fmt='o', ms=4, color=col, lw=1, label='Niemz' if c == 'N' else 'Zhu')
a1.axhline(F['dPc_opt'], color=INK, ls=':', lw=1); a1.text(8.4, F['dPc_opt'] - 0.4, 'optimally\noriented', fontsize=6, va='top', ha='right')
a1.set_xticks(x, SHORT); a1.set_xlabel('16A stage'); a1.set_ylabel('Pore-pressure increase for slip, ΔP$_c$ (MPa)')
a1.legend(frameon=False); tag(a1, 'b')
fig.tight_layout(); save(fig, 'Fig08_stress')

# ---------------------------------------------------------- Fig 9 overlap
O = pd.read_csv('results/stage_overlap.csv')
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3)); x = np.arange(len(STG) - 1)
for a, d in zip(axs, [25, 50]):
    for c, col, dd in [('N', CN, -0.17), ('Z', CZ, 0.17)]:
        t = O[O.cat == c].set_index('stage').reindex(STG[1:])
        a.bar(x + dd, t[f'f{d}'], 0.32, color=col, label='Niemz' if c == 'N' else 'Zhu')
    a.set_xticks(x, SHORT[1:]); a.set_ylim(0, 1); a.set_xlabel('16A stage')
    a.set_ylabel(f'Fraction within {d} m of\nan earlier stage\'s event')
axs[0].legend(frameon=False, ncol=2, loc='upper right')
for a, t in zip(axs, 'ab'): tag(a, t)
fig.tight_layout(); save(fig, 'Fig09_overlap')

# ---------------------------------------------------------- Fig S1 tolerance & magnitude independence
fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.2))
tl = pd.read_csv('results/match_tolerance.csv'); axs[0].plot(tl.tol, tl.n, '-o', color=INK, ms=4)
axs[0].set_xscale('log'); axs[0].set_xlabel('Matching tolerance (s)'); axs[0].set_ylabel('Matched pairs')
axs[1].scatter(P16.MN, P16.d3, s=2, color=INK, lw=0, alpha=0.5); axs[1].set_yscale('log')
axs[1].set_xlabel('Magnitude, Niemz'); axs[1].set_ylabel('3-D discrepancy (m)')
for a, t in zip(axs, 'ab'): tag(a, t)
fig.tight_layout(); save(fig, 'FigS1_matching_checks')

# ---------------------------------------------------------- Graphical abstract
s = '16A-6B'; q = P16[P16.stage == s]; r = TN.loc[s]
nrm = np.array([r.nx, r.ny, 0.0]); nrm /= np.linalg.norm(nrm)
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.3))
for a, (k, cols, col) in zip(axs, [('Niemz', ['xN', 'yN', 'zN'], CN), ('Zhu', ['xZ', 'yZ', 'zZ'], CZ)]):
    X = q[cols].values - q[cols].values.mean(0)
    a.scatter(X @ nrm, X[:, 2], s=7, color=col, lw=0)
    a.set_xlim(-250, 250); a.set_ylim(250, -250); a.set_aspect('equal')
    a.set_title(f'{"Surface network (Niemz)" if k == "Niemz" else "ML phase picking (Zhu)"}', color=INK, fontsize=8)
    a.set_xlabel('Normal to strike (m)'); a.set_ylabel('Relative depth (m)' if k == 'Niemz' else '')
fig.suptitle(f'Same {len(q)} earthquakes (stage 6B), two catalogues: a thin plane or a diffuse cloud', fontsize=8.5, y=0.99)
fig.text(0.5, 0.005, f'2,024 shared events · median location difference {F["shift_d3_med"]:.0f} m (stated error 9–20 m) · '
         f'critical pore pressure changes by up to {F["stage_dPc_absdiff_max"]:.0f} MPa', ha='center', fontsize=7, color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.95)); save(fig, 'Graphical_abstract')
