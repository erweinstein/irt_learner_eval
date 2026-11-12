# learners_py.py -- The Python learners of the IRT evaluation, called from R/00_learners.R through reticulate.
#   catboost  CatBoost with its defaults (1,000 rounds, learning rate set automatically), loss matched to the
#             outcome (log loss, Poisson, Tweedie 1.5, squared error)
#   realmlp   RealMLP with its tuned defaults (Holzmueller, Grinsztajn and Steinwart 2024; pytabkit's
#             RealMLP_TD), squared error for every non-binary outcome
#   tabpfn    TabPFN v2 (Hollmann et al. 2025), a pretrained transformer that predicts by in-context learning:
#             no training, the training data are its context; the regressor's mean for non-binary outcomes;
#             at most 10,000 training rows (a random 10,000 beyond that). One ensemble member (n_estimators = 1;
#             the package default is 4 for classification and 8 for regression), because on one CPU thread a
#             single fit and prediction already takes 45-95 seconds at 1,000 training and 5,000 test rows
# All run single-threaded on the CPU, seeded.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def fit_predict(name, type_, xtr, ytr, xte, seed=1):
    xtr = np.asarray(xtr, dtype=np.float32)
    xte = np.asarray(xte, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.float64).ravel()
    seed = int(seed)
    if name == "catboost":
        import catboost as cb
        if type_ == "binary":
            m = cb.CatBoostClassifier(random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
            m.fit(xtr, ytr.astype(int))
            return m.predict_proba(xte)[:, 1]
        loss = {"count": "Poisson", "cost": "Tweedie:variance_power=1.5"}.get(type_, "RMSE")
        m = cb.CatBoostRegressor(loss_function=loss, random_seed=seed, thread_count=1, verbose=False,
                                 allow_writing_files=False)
        m.fit(xtr, ytr)
        p = m.predict(xte, prediction_type="Exponent") if type_ in ("count", "cost") else m.predict(xte)
        return np.asarray(p, dtype=float)
    if name == "realmlp":
        import torch
        torch.set_num_threads(1)
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
        if type_ == "binary":
            m = RealMLP_TD_Classifier(random_state=seed, n_threads=1, device="cpu", verbosity=0)
            m.fit(xtr, ytr.astype(int))
            return m.predict_proba(xte)[:, 1]
        m = RealMLP_TD_Regressor(random_state=seed, n_threads=1, device="cpu", verbosity=0)
        m.fit(xtr, ytr)
        return np.asarray(m.predict(xte), dtype=float).ravel()
    if name == "tabpfn":
        import torch
        torch.set_num_threads(1)
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        rng = np.random.default_rng(seed)
        if xtr.shape[0] > 10000:
            i = rng.choice(xtr.shape[0], 10000, replace=False)
            xtr, ytr = xtr[i], ytr[i]
        if type_ == "binary":
            m = TabPFNClassifier(device="cpu", n_estimators=1, random_state=seed, ignore_pretraining_limits=True)
            m.fit(xtr, ytr.astype(int))
            return m.predict_proba(xte)[:, 1]
        m = TabPFNRegressor(device="cpu", n_estimators=1, random_state=seed, ignore_pretraining_limits=True)
        m.fit(xtr, ytr)
        return np.asarray(m.predict(xte), dtype=float).ravel()
    raise ValueError(name)
