"""
ct_tools.py — functions used in the nano-CT tutorial (nanoCT_tutorial.ipynb).

Pipeline for a thin, freestanding electrode film imaged by X-ray microscopy:
    stack -> crop -> denoise -> find faces / tilt-correct -> air-referenced thresholds
    -> porosity, depth profiles, dense layer -> connectivity -> S-rich clusters
    -> pore size -> geometric + diffusive tortuosity -> export

Conventions
    volume axes : (z = slice, y = row along the film, x = column across the film)
    aligned axes: (z, y, depth)   depth index D_AIR = sharp (smooth) face, increasing into the film
    grey values : 8-bit, as exported from the reconstruction software
"""
import glob, json, os, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi, sparse
from scipy.sparse.linalg import cg

PX = 1.226424      # voxel size (µm) from the scan header
D_AIR = 40         # number of air voxels kept in front of the sharp face after alignment
COLORS = dict(pore='#7B5CC8', matrix='#16A08C', srich='#D98E04', transport='#C8402A', geometric='#1F4E8C')


# ---------------------------------------------------------------- 1. read data
def stack(files):
    """Read a list of 2D TIFF slices into one 3D array (z, y, x)."""
    first = np.asarray(Image.open(files[0]))
    vol = np.zeros((len(files),) + first.shape, dtype=first.dtype)
    for i, f in enumerate(files):
        vol[i] = np.asarray(Image.open(f))
    return vol


# ---------------------------------------------------------------- 2. pre-processing
def denoise(vol, sigma=1.0):
    """3D Gaussian filter. sigma in voxels; 1 voxel removes pixel noise without merging pores."""
    return ndi.gaussian_filter(vol.astype(np.float32), sigma)


# ---------------------------------------------------------------- 3. faces and tilt
def find_faces(vol_s, band=25, air_cols=(5, 40), k_sd=3, smooth_px=9, frac=0.5):
    """Locate the two film faces in every (slice, band of rows).

    In each band of `band` rows, count the fraction of voxels per column that are clearly solid
    (grey > air mean + k_sd * air SD). Smooth this solid-fraction profile over `smooth_px` columns.
    A face is where it first / last exceeds `frac` x the median solid fraction inside the film
    (half-maximum when frac = 0.5). Outlier bands are removed with a 3 x 3 median filter over
    (slice, band). Returns integer columns x1 (sharp/smooth face) and x2 (rough face), shape (n_slices, n_bands).
    """
    nz, ny, nx = vol_s.shape; nb = ny // band
    air = vol_s[:, :, air_cols[0]:air_cols[1]]; cut = air.mean() + k_sd * air.std()
    x1 = np.zeros((nz, nb)); x2 = np.zeros((nz, nb))
    for z in range(nz):
        for b in range(nb):
            f = (vol_s[z, b * band:(b + 1) * band] > cut).mean(0)
            fs = ndi.uniform_filter1d(f, smooth_px)
            core = np.where(fs > 0.15)[0]                       # provisional film region
            level = np.median(f[core[0]:core[-1] + 1])
            above = np.where(fs > frac * level)[0]
            x1[z, b], x2[z, b] = above[0], above[-1]
    x1 = ndi.median_filter(x1, 3, mode='nearest'); x2 = ndi.median_filter(x2, 3, mode='nearest')
    return x1.astype(int), x2.astype(int)


def load_faces(csv, n_slices, n_bands):
    """Load a face table (columns x_sharp_face, x_rough_face, ordered by slice then band)."""
    import pandas as pd
    f = pd.read_csv(csv)
    return (f.x_sharp_face.values.reshape(n_slices, n_bands).astype(int),
            f.x_rough_face.values.reshape(n_slices, n_bands).astype(int))


def tilt_angle(x1, band=25):
    """Fit a plane x1 = a*z + b*y + c to the sharp-face positions; return the tilt (deg) along z and y."""
    z, b = np.indices(x1.shape); y = b * band + band / 2
    A = np.c_[z.ravel(), y.ravel(), np.ones(z.size)]
    (a, bb, c), *_ = np.linalg.lstsq(A, x1.ravel().astype(float), rcond=None)
    return np.degrees(np.arctan(a)), np.degrees(np.arctan(bb))


def align(vol_s, x1, x2, band=25, depth=280):
    """Shift every band so its sharp face sits at depth index D_AIR.

    Returns the aligned volume (z, y, depth) and `rough` (z, y): the local depth (voxels) of the
    rough face measured from the sharp face. Voxels deeper than `rough` are outside the film.
    """
    nz, nb = x1.shape
    A = np.zeros((nz, nb * band, depth), np.float32); rough = np.zeros((nz, nb * band), int)
    for z in range(nz):
        for b in range(nb):
            s = x1[z, b] - D_AIR
            A[z, b * band:(b + 1) * band] = vol_s[z, b * band:(b + 1) * band, s:s + depth]
            rough[z, b * band:(b + 1) * band] = x2[z, b] - x1[z, b]
    return A, rough


def film_mask(A, rough, h=163):
    """Film slab (from the sharp face, h voxels deep; 163 voxels = 200 µm here) and the mask of voxels inside the film."""
    film = A[:, :, D_AIR:D_AIR + h]
    inside = np.arange(h)[None, None, :] < rough[:, :, None]
    return film, inside


# ---------------------------------------------------------------- 4. thresholds
def air_stats(A, depth_range=(5, 30)):
    """Mean and SD of the air in front of the sharp face (aligned depth indices 5-30)."""
    air = A[:, :, depth_range[0]:depth_range[1]]
    return float(air.mean()), float(air.std())


def thresholds(air_mean, air_sd, ks=(2, 3, 4)):
    """Pore cut-offs air mean + k*SD."""
    return {f'{k}sd': air_mean + k * air_sd for k in ks}


def segment(film, inside, t_pore, t_s=75):
    """Three phases: 0 = outside film, 1 = pore, 2 = matrix, 3 = S-rich."""
    lab = np.zeros(film.shape, np.uint8)
    lab[inside & (film <= t_pore)] = 1
    lab[inside & (film > t_pore) & (film < t_s)] = 2
    lab[inside & (film >= t_s)] = 3
    return lab


# ---------------------------------------------------------------- 5. depth profiles
def depth_profiles(film, inside, thr, t_s=75, air_mean=None):
    """Porosity (per threshold), S-rich fraction and film fraction versus depth."""
    n_in = inside.sum((0, 1)).astype(float); ok = n_in > 0
    out = {'depth_um': np.arange(film.shape[2]) * PX, 'film_fraction': inside.mean((0, 1))}
    for k, t in thr.items():
        v = ((film <= t) & inside).sum((0, 1)) / np.where(ok, n_in, 1)
        out[f'porosity_{k}'] = np.where(ok, 100 * v, np.nan)
    out['S_rich_pct'] = np.where(ok, 100 * ((film >= t_s) & inside).sum((0, 1)) / np.where(ok, n_in, 1), np.nan)
    if air_mean is not None:
        out['mean_excess_grey'] = np.where(ok, (np.where(inside, film, 0).sum((0, 1)) / np.where(ok, n_in, 1)) - air_mean, np.nan)
    return out


def dense_layer(depth_um, porosity, bulk=(50, 170), n_sd=2, search_to=80):
    """Depths (µm) where porosity (3-pt mean) falls below bulk mean - n_sd*SD, within the first `search_to` µm."""
    sm = np.convolve(np.nan_to_num(porosity, nan=np.nanmean(porosity)), np.ones(3) / 3, mode='same')
    b = (depth_um > bulk[0]) & (depth_um < bulk[1])
    cut = np.nanmean(porosity[b]) - n_sd * np.nanstd(porosity[b])
    low = depth_um[(sm < cut) & (depth_um < search_to)]
    return (float(low.min()), float(low.max())) if len(low) else None, float(cut)


# ---------------------------------------------------------------- 6. connectivity
def connectivity(pore, inside):
    """Classify pore voxels as through (touch both faces), dead-end (one face) or isolated (none).

    `pore` should include voxels beyond the rough face (they are open electrolyte). Faces are
    modelled as reservoirs by padding one fully open layer on each side; 6-connectivity.
    """
    P = np.pad(pore, ((0, 0), (0, 0), (1, 1)), constant_values=True)
    lab, _ = ndi.label(P)
    a = set(np.unique(lab[:, :, 0])) - {0}; b = set(np.unique(lab[:, :, -1])) - {0}
    lab = lab[:, :, 1:-1]; pin = pore & inside
    through = np.isin(lab, list(a & b)) & pin
    isolated = pin & ~np.isin(lab, list(a | b))
    dead = pin & ~through & ~isolated
    return dict(through=through, dead_end=dead, isolated=isolated,
                through_pct=100 * through.sum() / pin.sum(), dead_end_pct=100 * dead.sum() / pin.sum(),
                isolated_pct=100 * isolated.sum() / pin.sum())


# ---------------------------------------------------------------- 7. S-rich clusters
def s_clusters(film, inside, t_s=75, min_vox=4):
    """26-connected S-rich clusters >= min_vox voxels: equivalent diameter (µm) and centroid depth (µm)."""
    bright = (film >= t_s) & inside
    lab, n = ndi.label(bright, structure=np.ones((3, 3, 3)))
    cnt = np.bincount(lab.ravel())[1:]; keep = np.where(cnt >= min_vox)[0] + 1
    dia = (6 * cnt[keep - 1] * PX ** 3 / np.pi) ** (1 / 3)
    depth = np.array(ndi.center_of_mass(bright, lab, keep))[:, 2] * PX
    return dict(diameter_um=dia, depth_um=depth, vol_pct=100 * bright.sum() / inside.sum())


# ---------------------------------------------------------------- 8. pore size
def pore_diameters(pore, inside):
    """Local pore diameter = 2 x distance to the nearest solid, sampled on the pore skeleton (µm)."""
    from skimage.morphology import skeletonize
    sk = skeletonize(pore) > 0
    dt = ndi.distance_transform_edt(pore) * PX
    return 2 * dt[sk & inside]


# ---------------------------------------------------------------- 9. tortuosity
def geometric_tortuosity(pore, start_step=5):
    """Shortest pore paths from the sharp face (depth 0) to the far face; tau_geo = median length / thickness.

    Minimum-cost paths with 26-neighbour moves (skimage MCP_Geometric). Every `start_step`-th
    open voxel on the sharp face is used as a start point.
    """
    from skimage.graph import MCP_Geometric
    mcp = MCP_Geometric(np.where(pore, 1.0, np.inf), sampling=(PX, PX, PX))
    starts = np.argwhere(pore[:, :, 0])[::start_step]
    cost, _ = mcp.find_costs([(int(a), int(b), 0) for a, b in starts])
    end = cost[:, :, -1][pore[:, :, -1]]; end = end[np.isfinite(end)]
    H = (pore.shape[2] - 1) * PX
    return float(np.median(end) / H), mcp, cost


def path_tortuosity(mcp, cost, pore, n=400, bin_vox=8, seed=1):
    """Trace n shortest paths back from random end voxels; whole-path and local (per depth bin) tortuosity."""
    H = (pore.shape[2] - 1) * PX
    ends = np.argwhere(pore[:, :, -1] & np.isfinite(cost[:, :, -1]))
    sel = np.random.default_rng(seed).choice(len(ends), min(n, len(ends)), replace=False)
    taus, local, paths = [], [], []
    for k in sel:
        p = np.array(mcp.traceback((int(ends[k, 0]), int(ends[k, 1]), pore.shape[2] - 1)))
        seg = np.linalg.norm(np.diff(p * PX, axis=0), axis=1)
        taus.append(seg.sum() / H); paths.append(p)
        b = np.floor(((p[1:, 2] + p[:-1, 2]) / 2) / bin_vox).astype(int)
        for j in range((pore.shape[2] - 1) // bin_vox):
            local.append(((j + 0.5) * bin_vox * PX, seg[b == j].sum() / (bin_vox * PX)))
    return np.array(taus), np.array(local), paths


def diffusion_tortuosity(pore, rtol=1e-8, maxiter=50000):
    """Steady-state diffusion through the face-to-face connected pores (Laplace equation, D0 = 1).

    c = 1 on the sharp face (depth 0), c = 0 on the far face (half-voxel Dirichlet), no flux on the
    side walls. Solved with Jacobi-preconditioned conjugate gradients.
    Returns D_eff/D0, tau_f = eps / (D_eff/D0), the concentration field, and a flux-balance check.
    """
    conn = connectivity(pore, np.ones_like(pore))['through']
    nid = -np.ones(pore.shape, np.int64); nid[conn] = np.arange(conn.sum()); N = int(conn.sum())
    rows, cols, diag = [], [], np.zeros(N); rhs = np.zeros(N)
    for ax in range(3):
        a_ = [slice(None)] * 3; b_ = [slice(None)] * 3; a_[ax] = slice(0, -1); b_[ax] = slice(1, None)
        m = conn[tuple(a_)] & conn[tuple(b_)]; i = nid[tuple(a_)][m]; j = nid[tuple(b_)][m]
        rows += [i, j]; cols += [j, i]; np.add.at(diag, i, 1.0); np.add.at(diag, j, 1.0)
    g = 2.0                                                 # half-voxel conductance to the boundary
    b0 = nid[:, :, 0][conn[:, :, 0]]; np.add.at(diag, b0, g); np.add.at(rhs, b0, g)
    b1 = nid[:, :, -1][conn[:, :, -1]]; np.add.at(diag, b1, g)
    r_, c_ = np.concatenate(rows), np.concatenate(cols)
    M = sparse.csr_matrix((-np.ones(len(r_)), (r_, c_)), shape=(N, N)) + sparse.diags(diag)
    c, info = cg(M, rhs, M=sparse.diags(1 / diag), rtol=rtol, maxiter=maxiter)
    J_in = (g * (1 - c[b0])).sum(); J_out = (g * c[b1]).sum()
    area = pore.shape[0] * pore.shape[1]
    Deff = J_in * pore.shape[2] / area
    eps = float(pore.mean())
    C = np.full(pore.shape, np.nan, np.float32); C[conn] = c
    return dict(Deff=float(Deff), tau_f=eps / Deff, eps=eps, C=C, J=J_in / area,
                flux_balance=float(abs(J_in - J_out) / J_in), cg_info=int(info))


def local_tau_f(C, pore, J, window=9):
    """Local tortuosity factor tau_f(z) = eps(z) * |d c_mean/dz| / J (slope from a `window`-voxel linear fit)."""
    cbar = np.nanmean(C, axis=(0, 1)); eps = pore.mean((0, 1)); n = len(cbar); h = window // 2
    dc = np.full(n, np.nan); zz = np.arange(n)
    for z in range(h, n - h):
        dc[z] = -np.polyfit(zz[z - h:z + h + 1], cbar[z - h:z + h + 1], 1)[0]
    return cbar, eps * dc / J


# ---------------------------------------------------------------- 10. save
def save_slices(lab, folder, prefix='seg'):
    """Save a labelled volume as 8-bit TIFF slices (0 outside, 85 pore, 170 matrix, 255 S-rich)."""
    os.makedirs(folder, exist_ok=True)
    for i, s in enumerate(lab):
        Image.fromarray((s * 85).astype(np.uint8)).save(os.path.join(folder, f'{prefix}_{i:04d}.tif'))


# ---------------------------------------------------------------- 11. automation
def analyse_stack(pattern, margin=8, faces='detect', air=None, sigma=1.0, ks=(2, 3, 4), t_s=75,
                  transport_rows=(250, 500), out='results', verbose=True):
    """Run the whole pipeline on a folder of TIFF slices and write results/ (JSON + CSV).

    pattern        glob pattern of the slices, e.g. 'data/slices/*.tif'
    margin         border (voxels) cropped away after filtering, so the filter sees real neighbours
    faces          'detect' (find_faces) or the path of a face table CSV (load_faces)
    air            None = measure in front of the sharp face (air_stats), or (mean, SD) to fix them
    transport_rows rows of the region used for the (slower) transport calculations
    """
    import pandas as pd
    t0 = time.time(); log = (lambda *a: print(*a, '[%.0f s]' % (time.time() - t0))) if verbose else (lambda *a: None)
    files = sorted(glob.glob(pattern)); vol = stack(files)
    vol_s = denoise(vol, sigma)
    if margin: vol_s = vol_s[:, margin:-margin, margin:-margin]
    log('loaded + filtered', vol_s.shape)
    nb = vol_s.shape[1] // 25
    x1, x2 = find_faces(vol_s) if faces == 'detect' else load_faces(faces, len(files), nb)
    A, rough = align(vol_s, x1, x2); film, inside = film_mask(A, rough)
    air_m, air_sd = air_stats(A) if air is None else air; thr = thresholds(air_m, air_sd, ks); log('air %.2f ± %.2f' % (air_m, air_sd))
    prof = depth_profiles(film, inside, thr, t_s, air_m)
    dl, _ = dense_layer(prof['depth_um'], prof['porosity_3sd'])
    cl = s_clusters(film, inside, t_s)
    res = dict(n_slices=len(files), faces=faces, air_mean=air_m, air_sd=air_sd, tilt_deg=tilt_angle(x1)[0],
               thickness_um_median=float(np.median(rough) * PX), dense_layer_um=dl,
               S_rich_vol_pct=cl['vol_pct'], n_clusters=int(len(cl['diameter_um'])),
               cluster_d_median_um=float(np.median(cl['diameter_um'])), thresholds={})
    tprof = {'depth_um': np.arange(film.shape[2]) * PX}
    r0, r1 = transport_rows
    for k, t in thr.items():
        pore = (film <= t) | ~inside
        con = connectivity(pore, inside)
        sub = pore[:, r0:r1]
        tg, _, _ = geometric_tortuosity(sub)
        d = diffusion_tortuosity(sub)
        cbar, tl = local_tau_f(d['C'], sub, d['J'])
        tprof[f'c_mean_{k}'] = cbar; tprof[f'tau_local_{k}'] = tl
        res['thresholds'][k] = dict(cutoff=t, porosity_pct=float(100 * ((film <= t) & inside).sum() / inside.sum()),
                                    through_pct=con['through_pct'], isolated_pct=con['isolated_pct'],
                                    tau_geo=tg, Deff=d['Deff'], tau_f=d['tau_f'], eps_transport=d['eps'],
                                    pore_d_median_um=float(np.median(pore_diameters(sub, inside[:, r0:r1]))),
                                    flux_balance=d['flux_balance'])
        log(k, {kk: round(v, 4) for kk, v in res['thresholds'][k].items()})
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1, default=float)
    pd.DataFrame(prof).to_csv(os.path.join(out, 'depth_profiles.csv'), index=False)
    pd.DataFrame(tprof).to_csv(os.path.join(out, 'transport_profiles.csv'), index=False)
    pd.DataFrame({'diameter_um': cl['diameter_um'], 'depth_um': cl['depth_um']}).to_csv(os.path.join(out, 'S_clusters.csv'), index=False)
    log('written to', out)
    return res
