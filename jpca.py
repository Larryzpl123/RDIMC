"""jPCA on two datasets, following the conventions of the official jPCA.m (Churchland lab codepack).

Usage:
    python jpca.py exampleData.mat sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb real.json

Monkey N: exampleData.mat from the codepack (trial-averaged PSTHs, 108 conditions, 218 neurons,
          -50..550 ms around neural movement onset, 10 ms bins). Used for Fig. 3f of Churchland et al. 2012.
Monkey J: MC_Maze full session, DANDI 000128 (raw spike times). We build PSTHs ourselves,
          aligned to (hand) movement onset.

Conventions copied from jPCA.m:
  - normalise each neuron by (firing-rate range over all supplied data + softenNorm), softenNorm = 5
  - subtract the cross-condition mean (computed over all supplied times)
  - PCA on the analysed window (-50..+150 ms), keep 6 PCs
  - dState = x(t+1) - x(t), preState = x(t)
  - R² = (Σ dState² - Σ error²) / Σ dState²   (no mean removal in the denominator)
"""
import json, sys
import numpy as np, h5py, scipy.io as sio
from scipy.ndimage import gaussian_filter1d

SOFT, K, WIN, DISP, DT = 5.0, 6, (-0.05, 0.15), (-0.05, 0.25), 0.010


def load_example(path):
    D = sio.loadmat(path, simplify_cells=True)["Data"]
    R = np.stack([d["A"] for d in D]).astype(float)            # conds x time x neurons
    tax = np.array(D[0]["times"]) / 1000.0
    return R, tax, dict(monkey="N", source="exampleData.mat (jPCA codepack)", alignment="neural movement onset",
                        n_conds=len(D), n_units=R.shape[2], n_trials=None)


def load_nwb(path, sig=0.030, pre=0.05, post=0.55):
    f = h5py.File(path, "r")
    st, idx = f["units/spike_times"][:], f["units/spike_times_index"][:]
    units = np.split(st, idx[:-1])
    tr = f["intervals/trials"]
    mo = tr["move_onset_time"][:]
    cond = list(zip(tr["trial_type"][:], tr["trial_version"][:]))
    conds = sorted(set(cond))
    pad = 0.15                                                    # extra time so smoothing has no edge effects
    T = int(round((pre + post + 2 * pad) * 1000))
    R = np.zeros((len(conds), T, len(units))); n = np.zeros(len(conds))
    for i, t0 in enumerate(mo):
        c = conds.index(cond[i]); b = np.zeros((T, len(units)))
        for u, s in enumerate(units):
            rel = s[(s >= t0 - pre - pad) & (s < t0 + post + pad)] - (t0 - pre - pad)
            np.add.at(b[:, u], (rel * 1000).astype(int).clip(0, T - 1), 1)
        R[c] += gaussian_filter1d(b * 1000.0, sig * 1000, axis=0); n[c] += 1
    R /= n[:, None, None]
    tax = np.arange(T) / 1000.0 - pre - pad
    keep = np.isclose(np.round(tax * 1000) % 10, 0) & (tax >= -pre - 1e-9) & (tax <= post + 1e-9)
    return R[:, keep, :], tax[keep], dict(monkey="J", source="MC_Maze full session, DANDI 000128",
                                          alignment="hand movement onset", n_conds=len(conds),
                                          n_units=len(units), n_trials=int(len(mo)))


def jpca(R, tax):
    C, T, N = R.shape
    big = R.reshape(-1, N)
    Rn = R / (big.max(0) - big.min(0) + SOFT)
    Rn = Rn - Rn.mean(0, keepdims=True)
    win = (tax >= WIN[0] - 1e-9) & (tax <= WIN[1] + 1e-9)
    small = Rn[:, win].reshape(-1, N); mu = small.mean(0)
    ev, V = np.linalg.eigh(np.cov((small - mu).T))               # PCA = eigenvectors of the covariance matrix
    o = np.argsort(ev)[::-1]; P = V[:, o[:K]]
    A = (Rn[:, win] - mu) @ P
    pre = A[:, :-1].reshape(-1, K); dS = (A[:, 1:] - A[:, :-1]).reshape(-1, K)
    Mfull = np.linalg.lstsq(pre, dS, rcond=None)[0].T          # least squares, any M
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    cols = []
    for i, j in pairs:
        E = np.zeros((K, K)); E[i, j] = 1; E[j, i] = -1; cols.append((pre @ E.T).ravel())
    h = np.linalg.lstsq(np.stack(cols, 1), dS.ravel(), rcond=None)[0]   # least squares, skew-symmetric M
    Ms = np.zeros((K, K))
    for hk, (i, j) in zip(h, pairs):
        Ms[i, j] = hk; Ms[j, i] = -hk
    vd = (dS ** 2).sum()
    r2 = lambda M: (vd - ((dS - pre @ M.T) ** 2).sum()) / vd
    e, Ve = np.linalg.eig(Ms)
    planes = []
    for k in np.argsort(-e.imag):
        if e[k].imag <= 0: continue
        a, b = Ve[:, k].real.copy(), Ve[:, k].imag.copy()
        a /= np.linalg.norm(a); b -= a * (a @ b); b /= np.linalg.norm(b)
        planes.append((a, b, e[k].imag / DT))
    a1, b1, w1 = planes[0]; a2 = planes[1][0]
    B = np.stack([a1, b1, a2], 1)
    if (B.T @ Ms @ B)[1, 0] < 0: B[:, 1] *= -1                    # make the top plane rotate counter-clockwise
    origVar = ((small - mu) ** 2).sum()
    plane_var = [float(((A.reshape(-1, K) @ np.stack([p[0], p[1]], 1)) ** 2).sum() / origVar) for p in planes]
    Y = (Rn - mu) @ P @ B
    disp = (tax >= DISP[0] - 1e-9) & (tax <= DISP[1] + 1e-9)
    Ys = Y[:, disp] * (2.0 / np.abs(Y[:, win, :2]).max())
    return dict(t=[round(x, 3) for x in tax[disp].tolist()], win=list(WIN),
                traj=[[[round(v, 3) for v in p] for p in c] for c in Ys.tolist()],
                omega_rad_s=[round(p[2], 3) for p in planes], freq_hz=[round(p[2] / (2 * np.pi), 3) for p in planes],
                eig_skew=[[0.0, round(float(x.imag) / DT, 3)] for x in e],
                eig_full=[[round(float(x.real) / DT, 3), round(float(x.imag) / DT, 3)] for x in np.linalg.eigvals(Mfull)],
                R2_full=round(float(r2(Mfull)), 4), R2_skew=round(float(r2(Ms)), 4),
                pca_var6=round(float(ev[o[:K]].sum() / ev.sum()), 4), plane_var=[round(v, 4) for v in plane_var])


if __name__ == "__main__":
    out = {}
    for key, (R, tax, meta) in (("N", load_example(sys.argv[1])), ("J", load_nwb(sys.argv[2]))):
        res = jpca(R, tax); res.update(meta); out[key] = res
        print(key, {k: v for k, v in res.items() if k not in ("traj", "t", "eig_skew", "eig_full")})
    json.dump(out, open(sys.argv[3], "w"), separators=(",", ":"))
