"""The field of an embryo and the volume sampled from it (docs/st_3d_alignment_strategies.md §1c, §1d).

One field per (embryo, pair features, seed), in the frame of the registered stack (`pipelines/stacked`):
x = (x, y, z) -> (f(x), o(x)), f the pair's ortholog embedding (the features of the cross cost), o the tissue
occupancy in [0, 1]. A small MLP on anisotropic Fourier features: frequencies ~ N(0, sigma_xy^2) in-plane and
N(0, sigma_z^2) in depth (lower: the data are coarser along z), plus the raw coordinates.

Losses (§1c), on the real spots only (never an interpolated point as a feature target, never held-out genes or labels):
1. features: |f(x_i) - e_i|^2 over minibatches of real spots;
2. depth: for a displacement pair (i in section k, j in k + 1; the peaks of the rows and columns of the stack's
   neighbour coupling, sampled by mass, every gap the same total weight) and x(t) = (1 - t) x_i + t x_j,
   |J_f(x(t)) (x_j - x_i) - (e_j - e_i)|^2, the derivative along the segment as one JVP (no full Jacobian);
3. occupancy (binary cross-entropy): positives are the real spots and points on the segments, jittered in-plane by
   occ_jitter spot spacings; negatives are in-plane points of each section farther than occ_out spacings from its
   tissue, and gap points farther than occ_out spacings from every segment (the segments' in-plane positions on 9
   levels of t per gap);
4. smoothness: |J_f|_F^2 at the spot and segment points (three JVPs).
Coordinates are centred and divided by the stack's in-plane RMS radius (one scale for all axes, so Jacobians and
spacings keep their meaning); features are centred and divided by their RMS per dimension.

Sampling (§1d): scrambled Sobol points in the box [in-plane bounding box of the spots + 2 spacings] x [z_first,
z_last] (the domain is clipped to the outermost sections), kept where o > 0.5, the first N in Sobol order. The box
is in stack units, in which z (the variogram gaps) and xy share one scale, so the density is isotropic: the sample
spacing h = (box volume x acceptance / N)^(1/3) is the same along every axis. Masses are equal; features are f(x).

Caches under the bench root: cache/fields/<pair>/<embryo>/seed<s>_<field hash>_<variant>.pt (weights and
normalization) and cache/volumes/<pair>/<embryo>/seed<s>_<field hash>_<variant>_n<N>.npz (the sampled volume). The
variant names the stack the field was fitted to: `full`, `loo<k>` (G1), `odd_common`, `even_reest` (G2).
"""
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from mgw.bench import store


@dataclass(frozen=True)
class FieldParams:
    width: int = 128
    depth: int = 3
    n_freq: int = 64
    sigma_xy: float = 4.0
    sigma_z: float = 1.0
    steps: int = 3000
    batch: int = 2048
    lr: float = 2e-3
    lambda_depth: float = 1.0
    lambda_occ: float = 0.5
    lambda_smooth: float = 1e-3
    occ_out: float = 2.0
    occ_jitter: float = 0.5
    device: str = "auto"

    @classmethod
    def from_mapping(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in names})


def torch_device(name="auto"):
    """cuda if available (or $MGW_FIELD_DEVICE), else cpu."""
    name = os.environ.get("MGW_FIELD_DEVICE") or name
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class FieldNet(torch.nn.Module):
    """x (normalized, n x 3) -> (f (n x d), occupancy logit (n,)) on anisotropic Fourier features."""

    def __init__(self, d_out, p: FieldParams, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        scale = torch.tensor([p.sigma_xy, p.sigma_xy, p.sigma_z], dtype=torch.float32)[:, None]
        self.register_buffer("B", torch.randn(3, p.n_freq, generator=g) * scale)
        torch.manual_seed(int(seed))
        layers, d_in = [], 3 + 2 * p.n_freq
        for _ in range(p.depth):
            layers += [torch.nn.Linear(d_in, p.width), torch.nn.SiLU()]
            d_in = p.width
        self.trunk = torch.nn.Sequential(*layers)
        self.f_head, self.o_head = torch.nn.Linear(p.width, d_out), torch.nn.Linear(p.width, 1)

    def encode(self, x):
        xb = 2 * math.pi * x @ self.B
        return torch.cat([x, torch.sin(xb), torch.cos(xb)], -1)

    def forward(self, x):
        h = self.trunk(self.encode(x))
        return self.f_head(h), self.o_head(h).squeeze(-1)

    def feat(self, x):
        return self.f_head(self.trunk(self.encode(x)))

    def occ(self, x):
        return self.o_head(self.trunk(self.encode(x))).squeeze(-1)


# ---------- the training data of one stack
def spot_spacing(ST):
    """Median over sections of the median in-plane distance to the nearest spot (stack units)."""
    out = []
    for k in range(len(ST["names"])):
        xy = ST["xyz"][ST["offsets"][k]:ST["offsets"][k + 1], :2]
        if len(xy) > 1:
            out.append(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    return float(np.median(out))


def segments(ST):
    """The displacement pairs of every gap as (i, j, w), w scaled so every gap carries the same total weight."""
    I, J, W = [], [], []
    for i, j, w in ST["pairs"]:
        I.append(np.asarray(i)); J.append(np.asarray(j)); W.append(np.asarray(w) / max(np.sum(w), 1e-300))
    n = max(len(I), 1)
    return np.concatenate(I).astype(np.int64), np.concatenate(J).astype(np.int64), np.concatenate(W) / n


def occupancy_negatives(ST, h, occ_out, n_target=20000, levels=9, seed=0):
    """Negative occupancy points (stack units): in-plane points of each section farther than occ_out * h from its
    spots, and gap points farther than occ_out * h from every segment of their gap (in-plane, at the nearest of
    `levels` values of t)."""
    rng = np.random.default_rng(seed)
    xyz, off, z = ST["xyz"], ST["offsets"], np.asarray(ST["z"], dtype=np.float64)
    lo, hi = xyz[:, :2].min(0) - 3 * h, xyz[:, :2].max(0) + 3 * h
    n_sec, r = len(ST["names"]), occ_out * h
    per = max(n_target // max(2 * n_sec - 1, 1), 64)
    plane, gap = [], []
    for k in range(n_sec):
        c = rng.uniform(lo, hi, (per, 2))
        d = cKDTree(xyz[off[k]:off[k + 1], :2]).query(c)[0]
        plane.append(np.c_[c[d > r], np.full((d > r).sum(), z[k])])
    for k, (i, j, _) in enumerate(ST["pairs"]):
        dz = z[k + 1] - z[k]
        if dz <= 0:
            continue
        ts = np.linspace(0, 1, levels)
        trees = [cKDTree((1 - t) * xyz[i, :2] + t * xyz[j, :2]) for t in ts]
        c = np.c_[rng.uniform(lo, hi, (per, 2)), rng.uniform(z[k], z[k + 1], per)]
        lev = np.rint((c[:, 2] - z[k]) / dz * (levels - 1)).astype(int)
        d = np.empty(per)
        for m in range(levels):
            sel = lev == m
            if sel.any():
                d[sel] = trees[m].query(c[sel, :2])[0]
        gap.append(c[d > r])
    cat = lambda L: np.vstack(L) if L else np.zeros((0, 3))
    return cat(plane), cat(gap)


# ---------- the field
class Field:
    """A trained field with its normalization; features() and occupancy() take stack coordinates (n x 3)."""

    def __init__(self, net, meta):
        self.net, self.meta = net, meta
        self.centre, self.scale = np.asarray(meta["centre"]), float(meta["scale"])
        self.mu, self.sd = np.asarray(meta["mu"]), float(meta["sd"])

    @property
    def device(self):
        return self.net.B.device

    def _x(self, xyz):
        return torch.as_tensor((np.asarray(xyz, dtype=np.float64) - self.centre) / self.scale, dtype=torch.float32,
                               device=self.device)

    @torch.no_grad()
    def _eval(self, xyz, fn, batch=65536):
        out = [fn(self._x(xyz[s:s + batch])).cpu().numpy() for s in range(0, len(xyz), batch)]
        return np.concatenate(out) if out else np.zeros(0)

    def features(self, xyz):
        return self._eval(xyz, self.net.feat).astype(np.float64) * self.sd + self.mu

    def occupancy(self, xyz):
        return 1 / (1 + np.exp(-self._eval(xyz, self.net.occ).astype(np.float64)))

    def box(self):
        """(lo, hi) of the sampling box in stack units: the spots' in-plane bounding box + 2 spacings, z clipped
        to [z_first, z_last]."""
        return np.asarray(self.meta["lo"]), np.asarray(self.meta["hi"])

    # ---------- persistence
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + f".tmp{os.getpid()}.pt")
        torch.save(dict(state=self.net.state_dict(), meta=self.meta), tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, device="auto"):
        z = torch.load(path, map_location="cpu", weights_only=False)
        meta = z["meta"]
        net = FieldNet(meta["d_out"], FieldParams.from_mapping(meta["params"]), meta["seed"])
        net.load_state_dict(z["state"])
        return cls(net.to(torch_device(device)).eval(), meta)


def fit(ST, F, p: FieldParams, seed=0, log=None):
    """Train the field on a stack-like dict ST (xyz, offsets, z, pairs, names; stack units) and the features F of
    its points (stack order). Returns a Field."""
    dev = torch_device(p.device)
    xyz = np.asarray(ST["xyz"], dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    z = np.asarray(ST["z"], dtype=np.float64)
    if len(z) < 2 or z[-1] <= z[0]:
        raise ValueError("a field needs >= 2 sections at distinct depths")
    h = spot_spacing(ST)
    centre = np.r_[xyz[:, :2].mean(0), 0.5 * (z[0] + z[-1])]
    scale = float(np.sqrt(((xyz[:, :2] - centre[:2]) ** 2).sum(1).mean()))
    mu = F.mean(0)
    sd = float(np.sqrt(((F - mu) ** 2).mean()))
    lo, hi = np.r_[xyz[:, :2].min(0) - 2 * h, z[0]], np.r_[xyz[:, :2].max(0) + 2 * h, z[-1]]
    neg_plane, neg_gap = occupancy_negatives(ST, h, p.occ_out, seed=seed)
    I, J, W = segments(ST)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    X, Fn = T((xyz - centre) / scale), T((F - mu) / sd)
    NP, NG = T((neg_plane - centre) / scale), T((neg_gap - centre) / scale)
    It, Jt, Wt = (torch.as_tensor(a, device=dev) for a in (I, J, W))
    d = F.shape[1]
    net = FieldNet(d, p, seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=p.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p.steps, eta_min=0.05 * p.lr)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    B, jit = p.batch, p.occ_jitter * h / scale
    eye = torch.eye(3, device=dev)
    bce = torch.nn.BCEWithLogitsLoss()
    hist = []
    for step in range(p.steps):
        ib = torch.randint(0, len(X), (B,), device=dev, generator=g)
        f_s, _ = net(X[ib])
        l_feat = ((f_s - Fn[ib]) ** 2).mean()
        terms = dict(feat=l_feat)
        loss = l_feat
        if len(It):
            pb = torch.multinomial(Wt, B, replacement=True, generator=g)
            t = torch.rand(B, 1, device=dev, generator=g)
            xa, xb = X[It[pb]], X[Jt[pb]]
            xt, dx = xa + t * (xb - xa), xb - xa
            _, jv = torch.func.jvp(net.feat, (xt,), (dx,))
            terms["depth"] = ((jv - (Fn[Jt[pb]] - Fn[It[pb]])) ** 2).mean()
            loss = loss + p.lambda_depth * terms["depth"]
        else:
            xt = X[ib]
        # occupancy: spots and jittered segment points against the two kinds of negatives
        half = B // 2
        jitter = torch.cat([torch.randn(half, 2, device=dev, generator=g) * jit, torch.zeros(half, 1, device=dev)], 1)
        pos = torch.cat([X[ib[:half]], xt[:half] + jitter])
        negs = [N[torch.randint(0, len(N), (half,), device=dev, generator=g)] for N in (NP, NG) if len(N)]
        if negs:
            neg = torch.cat(negs)
            logit = net.occ(torch.cat([pos, neg]))
            terms["occ"] = bce(logit, torch.cat([torch.ones(len(pos), device=dev), torch.zeros(len(neg), device=dev)]))
            loss = loss + p.lambda_occ * terms["occ"]
        if p.lambda_smooth > 0:
            xs = torch.cat([X[ib[:half]], xt[:half]])
            terms["smooth"] = sum((torch.func.jvp(net.feat, (xs,), (eye[c].expand_as(xs),))[1] ** 2).sum(1).mean()
                                  for c in range(3)) / d
            loss = loss + p.lambda_smooth * terms["smooth"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(p.steps // 10, 1) == 0 or step == p.steps - 1:
            hist.append(dict(step=step, **{k: v.item() for k, v in terms.items()}))
            if log:
                log(f"    field step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in hist[-1].items() if k != "step"))
    net.eval()
    meta = dict(centre=centre.tolist(), scale=scale, mu=mu.tolist(), sd=sd, lo=lo.tolist(), hi=hi.tolist(),
                spot_spacing=h, z=z.tolist(), d_out=d, seed=int(seed), params=dict(p.__dict__), history=hist,
                n_neg_plane=len(neg_plane), n_neg_gap=len(neg_gap))
    return Field(net, meta)


# ---------- sampling (§1d)
def sample(field, n, seed=0, max_draws=1 << 24):
    """N quasi-uniform points of the field's volume: scrambled Sobol in the box, kept where o > 0.5, the first N in
    Sobol order. Returns dict(xyz, F, occ, spacing, accept, n_draws)."""
    from scipy.stats import qmc
    lo, hi = field.box()
    sob = qmc.Sobol(3, scramble=True, seed=seed)
    U = sob.random_base2(max(int(np.ceil(np.log2(4 * n))), 8))
    kept, drawn = [], 0
    while True:
        X = lo + U * (hi - lo)
        o = field.occupancy(X)
        kept.append((X[o > 0.5], o[o > 0.5]))
        drawn += len(X)
        if sum(len(k[0]) for k in kept) >= n or drawn >= max_draws:
            break
        U = sob.random_base2(int(np.log2(sob.num_generated)))      # doubles the total: balance is kept
    xyz, occ = np.vstack([k[0] for k in kept]), np.concatenate([k[1] for k in kept])
    accept = len(xyz) / drawn
    if len(xyz) < n:
        raise RuntimeError(f"only {len(xyz)} of {n} samples inside the volume after {drawn} draws")
    xyz, occ = xyz[:n], occ[:n]
    return dict(xyz=xyz, F=field.features(xyz), occ=occ, accept=accept, n_draws=drawn,
                spacing=float((np.prod(hi - lo) * accept / n) ** (1 / 3)))


# ---------- caches
def field_path(root, cfg, seed, pair_id, embryo, variant="full"):
    return (Path(root) / "cache" / "fields" / store.safe(pair_id) / embryo /
            f"seed{seed}_{cfg.field_hash()}_{variant}.pt")


def volume_path(root, cfg, seed, pair_id, embryo, n, variant="full"):
    return (Path(root) / "cache" / "volumes" / store.safe(pair_id) / embryo /
            f"seed{seed}_{cfg.field_hash()}_{variant}_n{n}.npz")


def get_field(cfg, root, seed, pair_id, embryo, ST, F, variant="full", log=print):
    """The cached field of a stack (trained and saved on the first call)."""
    f = field_path(root, cfg, seed, pair_id, embryo, variant)
    p = FieldParams.from_mapping(cfg.raw["field"])
    if f.exists():
        return Field.load(f, p.device)
    log(f"  field {embryo} ({variant}, {p.steps} steps, {torch_device(p.device)})")
    fld = fit(ST, F, p, seed=seed, log=None)
    log(f"    losses at the end: " + ", ".join(f"{k} {v:.4f}" for k, v in fld.meta["history"][-1].items()
                                                if k != "step"))
    fld.save(f)
    return fld


def get_volume(cfg, root, seed, pair_id, embryo, field, n, variant="full"):
    """The cached volume of a field: dict(xyz, F, occ, spacing, accept, n_draws)."""
    f = volume_path(root, cfg, seed, pair_id, embryo, n, variant)
    if f.exists():
        z = np.load(f)
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    vol = sample(field, n, seed=seed)
    store.savez(f, **vol)
    return vol
