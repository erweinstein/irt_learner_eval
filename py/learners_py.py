# learners_py.py -- The Python learners of the IRT evaluation, called from R/00_learners.R through reticulate.
#   catboost  CatBoost with its defaults (1,000 rounds, learning rate set automatically), loss matched to the
#             outcome (log loss, Poisson, Tweedie 1.5, squared error)
#   catboost_tuned  CatBoost tuned as xgb_tuned is: learning rate 0.05, depth 4, rounds (at most 1,000) by 5-fold
#             cross-validated early stopping (added 13 November 2026)
#   realmlp   RealMLP with its tuned defaults (Holzmueller, Grinsztajn and Steinwart 2024; pytabkit's
#             RealMLP_TD), squared error for every non-binary outcome
#   tabpfn    TabPFN v2 (Hollmann et al. 2025), a pretrained transformer that predicts by in-context learning:
#             no training, the training data are its context; the regressor's mean for non-binary outcomes;
#             at most 10,000 training rows (a random 10,000 beyond that). One ensemble member (n_estimators = 1;
#             the package default is 4 for classification and 8 for regression), because on one CPU thread a
#             single fit and prediction already takes 45-95 seconds at 1,000 training and 5,000 test rows (and about
#             20 minutes at 2,000 training rows and 71 predictors)
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
    if name == "catboost_tuned":
        # the same tuning as xgb_tuned: a smaller learning rate (0.05) and shallower trees (depth 4), with the
        # number of rounds (at most 1,000) chosen by 5-fold cross-validated early stopping (30 rounds). Unlike the
        # defaults, the boosting starts from the training mean (as xgboost and LightGBM do): boost_from_average for
        # log loss and squared error, and a constant baseline of log(mean y) for Poisson and Tweedie, for which
        # CatBoost 1.2 does not allow boost_from_average (from 0, on the log scale, 1,000 small steps can fall well
        # short of the mean of spending in dollars)
        import catboost as cb
        loss = {"binary": "Logloss", "count": "Poisson", "cost": "Tweedie:variance_power=1.5"}.get(type_, "RMSE")
        par = {"loss_function": loss, "learning_rate": 0.05, "depth": 4, "random_seed": seed, "thread_count": 1,
               "logging_level": "Silent", "allow_writing_files": False}
        lab = ytr.astype(int) if type_ == "binary" else ytr
        logscale = type_ in ("count", "cost")
        if logscale:
            b0 = float(np.log(max(np.mean(ytr), 1e-8)))
            ptr, pte = cb.Pool(xtr, lab, baseline=np.full(len(lab), b0)), cb.Pool(xte, baseline=np.full(len(xte), b0))
        else:
            par["boost_from_average"] = True
            ptr, pte = cb.Pool(xtr, lab), cb.Pool(xte)
        cv = cb.cv(ptr, dict(par, iterations=1000), fold_count=5, early_stopping_rounds=30,
                   seed=seed, stratified=(type_ == "binary"), logging_level="Silent", as_pandas=True)
        col = [c for c in cv.columns if c.startswith("test-") and c.endswith("-mean")][0]
        best = max(10, int(cv[col].idxmin()) + 1)
        if type_ == "binary":
            m = cb.CatBoostClassifier(iterations=best, **par)
            m.fit(ptr)
            return m.predict_proba(pte)[:, 1]
        m = cb.CatBoostRegressor(iterations=best, **par)
        m.fit(ptr)
        p = m.predict(pte, prediction_type="Exponent") if logscale else m.predict(pte)
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
        # one thread unless IRT_TABPFN_THREADS says otherwise (2 was used for the last large first replicates of
        # 12 November 2026, run one at a time; the predictions agree to floating-point rounding)
        torch.set_num_threads(int(os.environ.get("IRT_TABPFN_THREADS", "1")))
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        rng = np.random.default_rng(seed)
        if xtr.shape[0] > 10000:
            i = rng.choice(xtr.shape[0], 10000, replace=False)
            xtr, ytr = xtr[i], ytr[i]
        # the test rows are predicted 1,000 at a time: each test row attends only to the training rows, so this
        # gives the same predictions (to float32 rounding, about 1e-6) at a third of the peak memory (3.1 GB at
        # 2,000 training rows, 71 predictors and 3,000 test rows, unbatched)
        batched = lambda f: np.concatenate([np.asarray(f(xte[i:i + 1000]), dtype=float).ravel()
                                            for i in range(0, xte.shape[0], 1000)])
        if type_ == "binary":
            m = TabPFNClassifier(device="cpu", n_estimators=1, random_state=seed, ignore_pretraining_limits=True)
            m.fit(xtr, ytr.astype(int))
            return batched(lambda a: m.predict_proba(a)[:, 1])
        m = TabPFNRegressor(device="cpu", n_estimators=1, random_state=seed, ignore_pretraining_limits=True)
        m.fit(xtr, ytr)
        return batched(m.predict)
    raise ValueError(name)
