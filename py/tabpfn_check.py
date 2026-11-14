# tabpfn_check.py -- TabPFN variants for the check of FLAGS.md item 21 (TabPFN underpredicts the mean of spending
# and counts), called from R/07_tabpfn_check.R through reticulate. Each call fits one variant and returns:
#   mean   for the raw variants, the mean of the predictive distribution (what the IRT run used); for the log
#          variants, the mean of expm1 over the 999 quantiles k / 1000 of the predicted distribution of log1p(y)
#          (a 0.05%-trimmed mean: the exact E[expm1(Z)] is useless here, since TabPFN's bins reach 98 on the
#          log scale and probabilities of 1e-9 there dominate it)
#   qmean  for the raw variants, the mean over the same 999 quantiles on the outcome's own scale, to see whether
#          the tails of the distribution account for the gap between its mean and the training mean
#   transforms  the target transform of each ensemble member
# Variants: n_estimators 1 (as in the IRT run) or 8 (TabPFN's default for regression); raw or log1p outcome.
# Same settings as py/learners_py.py otherwise (CPU, ignore_pretraining_limits, test rows 1,000 at a time).
# With cache (a path prefix), each block of 1,000 test rows is saved as it finishes and reloaded on the next call,
# so a fit longer than one session's time limit (8 members with 45-70 features) finishes over several runs. The
# fitted model is the same on every call (same data and random_state), and a test row's prediction does not
# depend on the other test rows, so the blocks are those of one uninterrupted fit.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def fit_variant(xtr, ytr, xte, seed, n_est, log, cache=None):
    import torch
    torch.set_num_threads(int(os.environ.get("IRT_TABPFN_THREADS", "1")))
    from tabpfn import TabPFNRegressor
    xtr = np.asarray(xtr, dtype=np.float32)
    xte = np.asarray(xte, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.float64).ravel()
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if xtr.shape[0] > 10000:
        i = rng.choice(xtr.shape[0], 10000, replace=False)
        xtr, ytr = xtr[i], ytr[i]
    t0 = time.time()
    m = TabPFNRegressor(device="cpu", n_estimators=int(n_est), random_state=seed, ignore_pretraining_limits=True)
    m.fit(xtr, np.log1p(ytr) if log else ytr)
    secs = time.time() - t0
    mean, qmean = [], []
    for i in range(0, xte.shape[0], 1000):
        f = None if cache is None else f"{cache}_{i // 1000}.npz"
        if f is not None and os.path.exists(f):
            z = np.load(f); mean.append(z["mean"]); qmean.append(z["qmean"]); secs += float(z["secs"]); continue
        t1 = time.time()
        o = m.predict(xte[i:i + 1000], output_type="full")
        q = quantiles(o["criterion"], o["logits"])                     # rows x 999
        mean.append(np.expm1(q).mean(axis=1) if log else np.asarray(o["mean"], dtype=float))
        qmean.append(np.full(q.shape[0], np.nan) if log else q.mean(axis=1))
        secs += time.time() - t1
        if f is not None:
            np.savez(f, mean=mean[-1], qmean=qmean[-1], secs=time.time() - t1)
    tr = "; ".join(transform_name(c.target_transform) for c in m.executor_.ensemble_configs)
    return {"mean": np.concatenate(mean), "qmean": np.concatenate(qmean), "transforms": tr, "secs": secs}


def transform_name(t):
    if t is None:
        return "none"
    steps = getattr(t, "steps", None)
    return "+".join(type(st[1]).__name__ for st in steps) if steps else type(t).__name__


def quantiles(crit, logits, k=1000):
    """The quantiles j / k (j = 1, ..., k - 1) of TabPFN's bar distribution, linear within each bin; a quantile
    that falls in one of the two outer (half-normal) bins is set to that bin's inner border (they carry about
    1e-7 of the probability here)."""
    import torch
    b = crit.borders.detach().cpu().double().numpy()
    w = np.diff(b)
    p = torch.softmax(torch.as_tensor(logits).double(), -1).cpu().numpy()
    qs = np.arange(1, k) / k
    out = np.empty((p.shape[0], k - 1))
    nb = len(w)
    for r in range(p.shape[0]):
        c = np.cumsum(p[r])
        j = np.minimum(np.searchsorted(c, qs), nb - 1)
        lo = np.where(j > 0, c[np.maximum(j - 1, 0)], 0.0)
        z = b[j] + np.clip((qs - lo) / np.maximum(p[r, j], 1e-300), 0, 1) * w[j]
        out[r] = np.clip(z, b[1], b[-2])
    return out
