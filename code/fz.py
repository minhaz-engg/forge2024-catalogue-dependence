"""Shared functions: catalogue-dependence of fracture interpretation, Utah FORGE 2024.
Frame: x east, y north of the 16A wellhead (m); z depth below the 16A wellhead (m, positive down).
Times: s since 2024-04-03 00:00 UTC."""
import numpy as np, pandas as pd

LAT0, LON0, ELEV0, R = 38.50402147, -112.8963897, 1650.0, 6371008.8   # 16A wellhead, mean Earth radius
T0 = pd.Timestamp('2024-04-03 00:00:00')
SEED = 2024

# ------------------------------------------------------------------ data
def local_xy(lat, lon):
    x = np.radians(np.asarray(lon) - LON0) * R * np.cos(np.radians(LAT0))
    y = np.radians(np.asarray(lat) - LAT0) * R
    return x, y

def read_niemz(path='data/raw/FORGE2024_Niemz2024_surface_catalog.csv'):
    c = pd.read_csv(path)
    t = (pd.to_datetime(c['time(UTC)']) - T0).dt.total_seconds().values
    x, y = local_xy(c['lat'], c['lon'])
    return pd.DataFrame(dict(t=t, x=x, y=y, z=c['depth[m_local]'].values, M=c['magnitude'].values))

def read_zhu(path='data/raw/FORGE2024_Zhu2026_phasepick_catalog.csv'):
    c = pd.read_csv(path)
    t = (pd.to_datetime(c['time']) - T0).dt.total_seconds().values
    x, y = local_xy(c['latitude'], c['longitude'])
    return pd.DataFrame(dict(t=t, x=x, y=y, z=1000 * c['depth_km'].values + ELEV0, M=c['magnitude'].values,
                             sx=1000 * c['sigma_x_km'].values, sy=1000 * c['sigma_y_km'].values,
                             sz=1000 * c['sigma_z_km'].values))

def spatial_filter(c, rmax=1500.0, zlim=(1500.0, 3600.0)):
    ref = c[['x', 'y']].median().values
    r = np.hypot(c.x - ref[0], c.y - ref[1])
    return c[(r < rmax) & (c.z > zlim[0]) & (c.z < zlim[1])].reset_index(drop=True)

BBL = 0.158987294928
def stage_table(path='data/injection_2024_stages.csv', q_on_bpm=5.0):
    """Same stage definition as the companion benchmark study (pump-on/shut-in at 5 bbl/min)."""
    inj = pd.read_csv(path)
    rows = []
    for s, g in inj.groupby('stage'):
        on = g[g.q_bpm > q_on_bpm]
        q = g.q_bpm.clip(0, 100).fillna(0).values * BBL / 60
        rows.append(dict(stage=s, well=g.well.iloc[0], t_on=on.t_s.min(), t_off=on.t_s.max(),
                         V=np.trapezoid(q, g.t_s.values)))
    st = pd.DataFrame(rows).sort_values('t_on').reset_index(drop=True)
    nxt = st.t_on.shift(-1).fillna(st.t_on.iloc[-1] + 86400).values
    st['t_end'] = np.minimum(nxt, st.t_off.values + 24 * 3600)
    return st

def assign_stage(c, st):
    lab = np.full(len(c), '', dtype=object)
    for _, s in st.iterrows():
        lab[((c.t >= s.t_on) & (c.t < s.t_end)).values] = s.stage
    c = c.copy(); c['stage'] = lab
    return c

STAGES_16A = ['16A-3R', '16A-4', '16A-5', '16A-6A', '16A-6B', '16A-7', '16A-8', '16A-9', '16A-10']

# ------------------------------------------------------------------ geometry
def pca(X):
    """Eigen-decomposition of the location covariance. Returns eigenvalues l1>=l2>=l3 and vectors (columns)."""
    X = np.asarray(X, float); X = X - X.mean(0)
    w, v = np.linalg.eigh(np.cov(X.T))
    return w[::-1], v[:, ::-1]

def planarity(l):
    """Gammans (2013) planarity 1 - 2*l3/(l1+l2), as used for FORGE by Van Fleet et al. (2026)."""
    return 1 - 2 * l[2] / (l[0] + l[1])

def pole(v3):
    """Lower-hemisphere pole (x east, y north, z down): returns unit vector with z >= 0."""
    n = v3 / np.linalg.norm(v3)
    return -n if n[2] < 0 else n

def pole_to_dd(n):
    """Pole (lower hemisphere) -> plane dip direction and dip (deg). Pole trend = dip direction + 180."""
    n = pole(n)
    dip = np.degrees(np.arccos(np.clip(n[2], -1, 1)))          # plane dip = 90 - pole plunge
    trend = np.degrees(np.arctan2(n[0], n[1])) % 360            # pole trend
    return (trend + 180) % 360, dip

def axial_angle(a, b):
    """Angle (deg, 0-90) between two axes (sign-insensitive)."""
    c = abs(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))

def plane_fit(X):
    l, v = pca(X)
    n = pole(v[:, 2])
    dd, dip = pole_to_dd(n)
    return dict(n=len(X), l1=np.sqrt(l[0]), l2=np.sqrt(l[1]), l3=np.sqrt(max(l[2], 0)), P=planarity(l),
                dipdir=dd, dip=dip, nx=n[0], ny=n[1], nz=n[2], cx=X[:, 0].mean(), cy=X[:, 1].mean(), cz=X[:, 2].mean())

def bootstrap_plane(X, B=1000, rng=None):
    """Bootstrap the pole: returns the 95th-percentile angular deviation (deg) from the full-sample pole and P CI."""
    rng = rng or np.random.default_rng(SEED)
    n0 = plane_fit(X); p0 = np.array([n0['nx'], n0['ny'], n0['nz']])
    ang, P = np.empty(B), np.empty(B)
    for b in range(B):
        f = plane_fit(X[rng.integers(0, len(X), len(X))])
        ang[b] = axial_angle(p0, [f['nx'], f['ny'], f['nz']]); P[b] = f['P']
    return np.percentile(ang, 95), np.percentile(P, [2.5, 97.5])

def axial_mean(N, w=None):
    """Mean axis and dispersion of a set of axes via the orientation tensor (Scheidegger 1965; Woodcock 1977).
    Returns mean axis, S1 (largest normalized eigenvalue; 1 = all parallel) and S1/S2 strength ln(S1/S2)."""
    N = np.asarray(N, float); w = np.ones(len(N)) if w is None else np.asarray(w, float)
    T = (N * w[:, None]).T @ N / w.sum()
    e, v = np.linalg.eigh(T)
    return pole(v[:, 2]), e[2], np.log(e[2] / max(e[1], 1e-12))

# ------------------------------------------------------------------ stress
def stress_tensor(depth, Sv_g=25.6e-3, Sh_g=17.15e-3, SH_g=21.15e-3, az_SH=20.0, Pp_g=9.3e-3):
    """Effective principal stresses (MPa) at depth (m below surface), tensor in (east, north, down) frame.
    Gradients in MPa/m. Normal-faulting: Sv > SH > Sh. az_SH: azimuth of SHmax (deg east of north)."""
    Pp = Pp_g * depth
    a = np.radians(az_SH)
    eH = np.array([np.sin(a), np.cos(a), 0.0]); eh = np.array([np.cos(a), -np.sin(a), 0.0]); ev = np.array([0, 0, 1.0])
    S = (SH_g * depth - Pp) * np.outer(eH, eH) + (Sh_g * depth - Pp) * np.outer(eh, eh) + (Sv_g * depth - Pp) * np.outer(ev, ev)
    return S, Pp

def tendencies(n, S, mu=0.6):
    """Slip tendency Ts = tau/sigma_n' (Morris et al., 1996), dilation tendency Td = (s1-sn)/(s1-s3)
    (Ferrill et al., 1999) and the pore-pressure increase needed for Coulomb failure dPc = sigma_n' - tau/mu."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    t = S @ n; sn = n @ t; tau = np.sqrt(max(t @ t - sn**2, 0.0))
    e = np.linalg.eigvalsh(S)
    return tau / sn, (e[2] - sn) / (e[2] - e[0]), sn - tau / mu

def stress_mc(K=2000, rng=None):
    """Monte-Carlo stress states: Sh gradient U[16.7,17.6] kPa/m (Moore et al., 2020), SH gradient U[18.3,24.0] kPa/m
    (Ye & Ghassemi), SH azimuth U[N5E,N30E] (Ye & Ghassemi), Sv 25.6 kPa/m (Moore et al., 2020), Pp 9.3 kPa/m."""
    rng = rng or np.random.default_rng(SEED)
    return pd.DataFrame(dict(Sh=rng.uniform(16.7e-3, 17.6e-3, K), SH=rng.uniform(18.3e-3, 24.0e-3, K),
                             az=rng.uniform(5, 30, K)))
