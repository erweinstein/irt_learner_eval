# tabpfn_dist_check.py -- for R/07c_tabpfn_dist_check.R (FLAGS.md item 21): where in its predictive distribution
# does TabPFN lose the mean on counts? Fits one ensemble member on the outcome exactly as py/tabpfn_check.py does
# (same seed and settings) and returns, for each test row (and, separately, each training row predicted in-sample),
# the predicted mean, median, P(y < 0.5) (a zero, for counts) and P(y > t) at the thresholds t passed in (the
# training outcome's 90th and 99th percentiles), with the partial means E[Y 1(Y > t)]. Probabilities come from TabPFN's bar distribution, uniform within
# each bin (the two outer half-normal bins hold about 1e-7 of the probability here and are treated the same way).
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def cdf_at(b, p, v):
    """P(Y <= v) for each row of bin probabilities p, bins with borders b, uniform within a bin."""
    w = np.diff(b)
    frac = np.clip((v - b[:-1]) / w, 0, 1)                         # share of each bin below v
    return (p * frac).sum(axis=1)


def partial_mean_above(b, p, v):
    """E[Y 1(Y > v)] for each row, uniform within a bin: the part of each bin above v, times that part's midpoint."""
    lo = np.maximum(b[:-1], v)
    frac = np.clip((b[1:] - lo) / np.diff(b), 0, 1)
    return (p * frac * (lo + b[1:]) / 2).sum(axis=1)


def summarize(m, x, thresholds):
    import torch
    out = {"mean": [], "median": [], "p0": []}
    for t in range(len(thresholds)):
        out[f"exceed{t}"] = []
        out[f"pmean{t}"] = []
    for i in range(0, x.shape[0], 1000):
        o = m.predict(x[i:i + 1000], output_type="full")
        b = o["criterion"].borders.detach().cpu().double().numpy()
        p = torch.softmax(torch.as_tensor(o["logits"]).double(), -1).cpu().numpy()
        out["mean"].append(np.asarray(o["mean"], dtype=float))
        out["median"].append(np.asarray(o["median"], dtype=float))
        out["p0"].append(cdf_at(b, p, 0.5))
        for t, v in enumerate(thresholds):
            out[f"exceed{t}"].append(1 - cdf_at(b, p, float(v)))
            out[f"pmean{t}"].append(partial_mean_above(b, p, float(v)))
    return {k: np.concatenate(v) for k, v in out.items()}


def dist_check(xtr, ytr, xte, seed, thresholds, max_test=2000):
    import torch
    torch.set_num_threads(int(os.environ.get("IRT_TABPFN_THREADS", "1")))
    from tabpfn import TabPFNRegressor
    xtr = np.asarray(xtr, dtype=np.float32)
    xte = np.asarray(xte, dtype=np.float32)[:int(max_test)]
    ytr = np.asarray(ytr, dtype=np.float64).ravel()
    seed = int(seed)
    m = TabPFNRegressor(device="cpu", n_estimators=1, random_state=seed, ignore_pretraining_limits=True)
    m.fit(xtr, ytr)
    th = [float(v) for v in np.atleast_1d(thresholds)]
    return {"test": summarize(m, xte, th), "train": summarize(m, xtr, th)}
