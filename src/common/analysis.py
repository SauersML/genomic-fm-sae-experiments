"""Shared statistical analysis & evaluation harness for the genomics-SAE project.

This module operates on *generic* inputs so all three aims can reuse it:

    X          (n_samples, n_features) feature matrix
               e.g. SAE feature vectors, or ref/alt feature deltas, or
               per-locus haplotype feature profiles.
    y          (n_samples,) labels: binary, multiclass, or continuous.
    groups     (n_samples,) optional group ids for leakage-safe splitting
               (e.g. sample/haplotype/locus/contig id). No group may appear
               in both train and test.
    covariates (n_samples, n_cov) optional "obvious" confounders
               (e.g. SV length, GC content) used for confound control.

Design principles (this harness is built to survive adversarial scrutiny):
  * Never report a number without a CI or a null comparison.
  * Always compare against a shuffled-label baseline (permutation test).
  * Always compare against a covariate-only baseline + residualized-feature
    model so we know features add signal beyond the obvious confounds.
  * Group-aware splitting everywhere; no group leaks across train/test.
  * Functions are pure (no global state) and importable.

Pure CPU, headless (matplotlib Agg). UMAP is optional and skipped gracefully.

Public API
----------
  evaluate_separation(X, y, groups=None, covariates=None,
                      task='classification', seed=0, n_splits=5,
                      n_boot=1000) -> dict
  permutation_test(X, y, groups=None, covariates=None,
                   task='classification', n_perm=1000, seed=0) -> dict
  differential_features(X, y, task='classification',
                        feature_names=None) -> pandas.DataFrame
  make_plots(X, y, outdir, eval_result=None, diff_table=None,
             task='classification', seed=0) -> list[str]
  run_report(X, y, groups=None, covariates=None, outdir='.',
             title='analysis', task='classification', seed=0,
             feature_names=None, n_perm=1000) -> dict
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless before pyplot import
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    r2_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

try:  # optional dependency, skipped gracefully
    import umap  # noqa: F401
    _HAVE_UMAP = True
except Exception:  # pragma: no cover - import guard
    _HAVE_UMAP = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _is_binary(y) -> bool:
    u = np.unique(np.asarray(y))
    return u.size == 2


def _make_splitter(y, groups, n_splits, seed, stratify):
    """Return a list of (train_idx, test_idx) honoring groups when present."""
    n = len(y)
    n_splits = int(min(n_splits, _max_splits(y, groups)))
    if n_splits < 2:
        raise ValueError(
            "Not enough samples/groups for >=2 CV folds "
            f"(got effective n_splits={n_splits})."
        )
    if groups is not None:
        # GroupKFold guarantees no group spans train/test.
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(np.zeros(n), y, groups)), n_splits
    if stratify:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(n), y)), n_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n))), n_splits


def _max_splits(y, groups) -> int:
    if groups is not None:
        return int(np.unique(groups).size)
    return len(y)


def _classification_metrics(y_true, y_score) -> dict:
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["auprc"] = float("nan")
    return out


def _regression_metrics(y_true, y_pred) -> dict:
    out = {"r2": float(r2_score(y_true, y_pred))}
    rho, _ = stats.spearmanr(y_true, y_pred)
    out["spearman"] = float(rho)
    return out


def _bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=0, groups=None):
    """Bootstrap a metric's 95% CI.

    Resamples groups (cluster bootstrap) when groups given, else samples.
    Returns (lo, hi). NaN if metric undefined.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    vals = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_boot):
        if groups is not None:
            chosen = rng.choice(uniq, size=uniq.size, replace=True)
            idx = np.concatenate([idx_by_group[g] for g in chosen])
        else:
            idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def _build_models(task: str, seed: int):
    """Return dict name -> sklearn estimator/pipeline for the task."""
    if task == "classification":
        linear = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced",
                random_state=seed)),  # default is L2 penalty
        ])
        gbt = HistGradientBoostingClassifier(random_state=seed)
        return {"l2_logreg": linear, "hist_gbt": gbt}
    else:
        linear = Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=seed)),
        ])
        gbt = HistGradientBoostingRegressor(random_state=seed)
        return {"ridge": linear, "hist_gbt": gbt}


def _cv_predictions(model, X, y, splits, task):
    """Out-of-fold predictions. Returns (y_true_oof, y_pred_oof) aligned to
    the concatenation of test folds (order = fold order)."""
    yt, yp = [], []
    for tr, te in splits:
        m = _clone_fit(model, X[tr], y[tr])
        if task == "classification":
            score = _proba(m, X[te])
        else:
            score = m.predict(X[te])
        yt.append(y[te])
        yp.append(score)
    return np.concatenate(yt), np.concatenate(yp)


def _clone_fit(model, X, y):
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, y)
    return m


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def _residualize(X, covariates, splits):
    """Out-of-fold residualization: for each feature, regress it on the
    covariates using train folds only, then subtract the prediction on the
    test fold. Returns a residualized copy of X (same shape) with values
    filled at each sample's test-fold position. This removes linear covariate
    signal from X without leaking test info."""
    X = _as_2d(X)
    C = _as_2d(covariates)
    Xr = np.full_like(X, np.nan)
    for tr, te in splits:
        Cb = np.column_stack([np.ones(len(tr)), C[tr]])
        # least squares beta per feature: (C'C)^-1 C' X
        beta, *_ = np.linalg.lstsq(Cb, X[tr], rcond=None)
        Cte = np.column_stack([np.ones(len(te)), C[te]])
        Xr[te] = X[te] - Cte @ beta
    return Xr


def _ordered_groups(splits, groups):
    if groups is None:
        return None
    groups = np.asarray(groups)
    return np.concatenate([groups[te] for _, te in splits])


# --------------------------------------------------------------------------- #
# 1. evaluate_separation
# --------------------------------------------------------------------------- #
def evaluate_separation(
    X,
    y,
    groups=None,
    covariates=None,
    task: str = "classification",
    seed: int = 0,
    n_splits: int = 5,
    n_boot: int = 1000,
) -> dict:
    """Group-aware CV evaluation of how well X separates / predicts y.

    Fits baseline models (L2 logistic regression + HistGradientBoosting for
    classification; Ridge + HistGradientBoosting for regression) with
    leakage-safe CV, and reports held-out metrics with bootstrap 95% CIs.

    Confound control (when ``covariates`` is given):
      * ``covariates_only``  -- model trained on covariates alone (the
        "obvious" baseline a feature set must beat).
      * ``features_residualized`` -- features after out-of-fold linear
        residualization of the covariates; tells us whether features carry
        signal *beyond* the covariates.

    Returns a nested dict of metrics, each with point estimate and ci95.
    """
    X = _as_2d(X)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
    stratify = (task == "classification") and _is_binary(y)
    splits, used_splits = _make_splitter(y, groups, n_splits, seed, stratify)
    groups_oof = _ordered_groups(splits, groups)

    if task == "classification" and not _is_binary(y):
        warnings.warn(
            "Multiclass detected; AUROC/AUPRC reported via one-vs-rest macro "
            "is not implemented in this harness. Binarizing is recommended. "
            "Proceeding with binary path will fail.")

    def eval_block(Xb, cov_block=None):
        block = {}
        models = _build_models(task, seed)
        for name, model in models.items():
            yt, yp = _cv_predictions(model, Xb, y, splits, task)
            if task == "classification":
                metrics = _classification_metrics(yt, yp)
                ci_au = _bootstrap_ci(yt, yp, lambda a, b: roc_auc_score(a, b),
                                      n_boot, seed, groups_oof)
                ci_ap = _bootstrap_ci(yt, yp,
                                      lambda a, b: average_precision_score(a, b),
                                      n_boot, seed + 1, groups_oof)
                block[name] = {
                    "auroc": metrics["auroc"], "auroc_ci95": ci_au,
                    "auprc": metrics["auprc"], "auprc_ci95": ci_ap,
                }
            else:
                metrics = _regression_metrics(yt, yp)
                ci_r2 = _bootstrap_ci(yt, yp, r2_score, n_boot, seed, groups_oof)
                ci_sp = _bootstrap_ci(
                    yt, yp,
                    lambda a, b: stats.spearmanr(a, b)[0],
                    n_boot, seed + 1, groups_oof)
                block[name] = {
                    "r2": metrics["r2"], "r2_ci95": ci_r2,
                    "spearman": metrics["spearman"], "spearman_ci95": ci_sp,
                }
        return block

    result = {
        "task": task,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_groups": int(np.unique(groups).size) if groups is not None else None,
        "n_splits": int(used_splits),
        "split_strategy": ("GroupKFold" if groups is not None
                           else ("StratifiedKFold" if stratify else "KFold")),
        "features": eval_block(X),
    }

    if covariates is not None:
        C = _as_2d(covariates)
        result["covariates_only"] = eval_block(C)
        Xr = _residualize(X, C, splits)
        result["features_residualized"] = eval_block(Xr)

    return result


# --------------------------------------------------------------------------- #
# 2. permutation_test
# --------------------------------------------------------------------------- #
def permutation_test(
    X,
    y,
    groups=None,
    covariates=None,
    task: str = "classification",
    metric: Optional[str] = None,
    model: Optional[str] = None,
    n_perm: int = 1000,
    seed: int = 0,
    n_splits: int = 5,
) -> dict:
    """Label-permutation null for the held-out metric.

    The observed statistic is the out-of-fold metric of ``model``. Labels are
    permuted **within groups' structure** (permuting group->label assignment
    when groups are given, so the group block structure of the null matches
    the observed analysis and we don't break leakage control).

    Returns dict with observed value, null mean/std, and a one-sided p-value
    (P(null >= observed)) with the standard +1 correction.
    """
    X = _as_2d(X)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
    stratify = (task == "classification") and _is_binary(y)
    splits, _ = _make_splitter(y, groups, n_splits, seed, stratify)

    if metric is None:
        metric = "auroc" if task == "classification" else "spearman"
    if model is None:
        model = "l2_logreg" if task == "classification" else "ridge"

    def metric_fn(yt, yp):
        if metric == "auroc":
            return roc_auc_score(yt, yp)
        if metric == "auprc":
            return average_precision_score(yt, yp)
        if metric == "r2":
            return r2_score(yt, yp)
        if metric == "spearman":
            return stats.spearmanr(yt, yp)[0]
        raise ValueError(f"unknown metric {metric}")

    def score_for(y_use):
        m = _build_models(task, seed)[model]
        yt, yp = _cv_predictions(m, X, y_use, splits, task)
        return metric_fn(yt, yp)

    observed = float(score_for(y))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    if groups is not None:
        uniq = np.unique(groups)
        # one label per group (use first occurrence) so permutation preserves
        # the group block structure.
        first_idx = {g: np.where(groups == g)[0][0] for g in uniq}
        group_label = np.array([y[first_idx[g]] for g in uniq])
        gpos = {g: i for i, g in enumerate(uniq)}
        idx_map = np.array([gpos[g] for g in groups])
    for i in range(n_perm):
        if groups is not None:
            perm = rng.permutation(group_label)
            y_perm = perm[idx_map]
        else:
            y_perm = rng.permutation(y)
        try:
            null[i] = score_for(y_perm)
        except ValueError:
            null[i] = np.nan

    null = null[np.isfinite(null)]
    p = (np.sum(null >= observed) + 1) / (null.size + 1)
    return {
        "metric": metric,
        "model": model,
        "observed": observed,
        "null_mean": float(np.mean(null)) if null.size else float("nan"),
        "null_std": float(np.std(null)) if null.size else float("nan"),
        "n_perm_effective": int(null.size),
        "p_value": float(p),
        "permute_within_groups": groups is not None,
    }


# --------------------------------------------------------------------------- #
# 3. differential_features
# --------------------------------------------------------------------------- #
def differential_features(
    X,
    y,
    task: str = "classification",
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-feature univariate association tests with BH-FDR.

    Binary classification: Mann-Whitney U (rank, robust) and Welch t per
    feature; effect = mean(pos) - mean(neg). Continuous: Spearman rho.

    Returns a DataFrame ranked by adjusted p-value (ascending), with columns:
      feature, stat, effect, p_value, p_adj (BH-FDR), direction.
    """
    from statsmodels.stats.multitest import multipletests

    X = _as_2d(X)
    y = np.asarray(y)
    n_feat = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_feat)]
    feature_names = list(feature_names)

    rows = []
    if task == "classification" and _is_binary(y):
        classes = np.unique(y)
        pos, neg = classes[1], classes[0]
        mpos = y == pos
        mneg = y == neg
        for j in range(n_feat):
            a = X[mpos, j]
            b = X[mneg, j]
            effect = float(np.mean(a) - np.mean(b))
            try:
                u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                u, p_mw = np.nan, 1.0
            rows.append((feature_names[j], float(u), effect, float(p_mw)))
        cols = ["feature", "stat_mannwhitney_u", "effect_mean_diff", "p_value"]
    else:
        for j in range(n_feat):
            rho, p = stats.spearmanr(X[:, j], y)
            rows.append((feature_names[j], float(rho), float(rho), float(p)))
        cols = ["feature", "stat_spearman_rho", "effect_spearman_rho", "p_value"]

    df = pd.DataFrame(rows, columns=cols)
    pvals = df["p_value"].to_numpy()
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    df["p_adj_bh"] = p_adj
    df["direction"] = np.where(df[cols[2]] >= 0, "up", "down")
    df = df.sort_values("p_adj_bh", kind="mergesort").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 4. Plots
# --------------------------------------------------------------------------- #
def make_plots(
    X,
    y,
    outdir: str,
    eval_result: Optional[dict] = None,
    diff_table: Optional[pd.DataFrame] = None,
    task: str = "classification",
    seed: int = 0,
) -> list:
    """Save diagnostic plots to ``outdir``. Returns list of file paths."""
    os.makedirs(outdir, exist_ok=True)
    X = _as_2d(X)
    y = np.asarray(y)
    paths = []

    # ---- PCA scatter colored by y -------------------------------------- #
    try:
        from sklearn.decomposition import PCA
        Xs = StandardScaler().fit_transform(X)
        ncomp = min(2, Xs.shape[1])
        pc = PCA(n_components=ncomp, random_state=seed).fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(5, 4))
        if ncomp == 2:
            sc = ax.scatter(pc[:, 0], pc[:, 1], c=y, cmap="viridis", s=12, alpha=0.7)
        else:
            sc = ax.scatter(pc[:, 0], np.zeros_like(pc[:, 0]), c=y,
                            cmap="viridis", s=12, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="y")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2" if ncomp == 2 else "")
        ax.set_title("PCA of X colored by y")
        p = os.path.join(outdir, "pca.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        paths.append(p)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"PCA plot skipped: {e}")

    # ---- UMAP (optional) ----------------------------------------------- #
    if _HAVE_UMAP and X.shape[1] >= 2 and X.shape[0] >= 5:
        try:
            import umap
            Xs = StandardScaler().fit_transform(X)
            emb = umap.UMAP(random_state=seed, n_neighbors=min(15, X.shape[0] - 1)
                            ).fit_transform(Xs)
            fig, ax = plt.subplots(figsize=(5, 4))
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="viridis", s=12, alpha=0.7)
            fig.colorbar(sc, ax=ax, label="y")
            ax.set_title("UMAP of X colored by y")
            p = os.path.join(outdir, "umap.png")
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            paths.append(p)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"UMAP plot skipped: {e}")

    # ---- ROC curve (classification only) ------------------------------- #
    if task == "classification" and _is_binary(y):
        try:
            model = _build_models("classification", seed)["l2_logreg"]
            splits, _ = _make_splitter(y, None, 5, seed, stratify=True)
            yt, yp = _cv_predictions(model, X, y, splits, "classification")
            fpr, tpr, _ = roc_curve(yt, yp)
            auc = roc_auc_score(yt, yp)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, label=f"L2 logreg (AUROC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("Held-out ROC"); ax.legend(loc="lower right")
            p = os.path.join(outdir, "roc.png")
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            paths.append(p)

            # ---- Calibration ------------------------------------------- #
            span = float(np.ptp(yp))
            ps = (yp - yp.min()) / (span + 1e-12) if span > 0 else yp
            frac_pos, mean_pred = calibration_curve(yt, ps, n_bins=min(10, len(yt) // 5 or 2))
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(mean_pred, frac_pos, "o-", label="model")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
            ax.set_xlabel("Mean predicted"); ax.set_ylabel("Fraction positive")
            ax.set_title("Calibration"); ax.legend(loc="upper left")
            p = os.path.join(outdir, "calibration.png")
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            paths.append(p)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"ROC/calibration plot skipped: {e}")

    # ---- Volcano / top-feature bar ------------------------------------- #
    if diff_table is not None and len(diff_table):
        try:
            effect_col = [c for c in diff_table.columns if c.startswith("effect")][0]
            fig, ax = plt.subplots(figsize=(5, 4))
            eff = diff_table[effect_col].to_numpy()
            neglogp = -np.log10(np.clip(diff_table["p_adj_bh"].to_numpy(), 1e-300, 1))
            ax.scatter(eff, neglogp, s=12, alpha=0.6)
            ax.axhline(-np.log10(0.05), color="r", ls="--", lw=0.8, label="FDR 0.05")
            ax.set_xlabel(effect_col); ax.set_ylabel("-log10(FDR)")
            ax.set_title("Volcano: differential features"); ax.legend()
            p = os.path.join(outdir, "volcano.png")
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            paths.append(p)

            top = diff_table.head(min(20, len(diff_table)))
            fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(top))))
            ax.barh(top["feature"].astype(str)[::-1],
                    top[effect_col].to_numpy()[::-1])
            ax.set_xlabel(effect_col)
            ax.set_title("Top differential features")
            p = os.path.join(outdir, "top_features.png")
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            paths.append(p)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"Volcano/bar plot skipped: {e}")

    return paths


# --------------------------------------------------------------------------- #
# 5. run_report
# --------------------------------------------------------------------------- #
def _fmt_ci(d, key):
    v = d.get(key)
    lo, hi = d.get(f"{key}_ci95", (float("nan"), float("nan")))
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def run_report(
    X,
    y,
    groups=None,
    covariates=None,
    outdir: str = ".",
    title: str = "analysis",
    task: str = "classification",
    seed: int = 0,
    feature_names: Optional[Sequence[str]] = None,
    n_perm: int = 1000,
    n_splits: int = 5,
    n_boot: int = 1000,
) -> dict:
    """Run the full harness and write markdown + plots + results JSON.

    Returns the results dict (also written to ``<outdir>/results.json``).
    """
    os.makedirs(outdir, exist_ok=True)
    X = _as_2d(X)
    y = np.asarray(y)

    sep = evaluate_separation(X, y, groups, covariates, task, seed,
                              n_splits=n_splits, n_boot=n_boot)
    perm = permutation_test(X, y, groups, covariates, task,
                            n_perm=n_perm, seed=seed, n_splits=n_splits)
    diff = differential_features(X, y, task, feature_names)
    plots = make_plots(X, y, outdir, sep, diff, task, seed)

    results = {
        "title": title,
        "separation": sep,
        "permutation_test": perm,
        "n_significant_features_fdr05": int((diff["p_adj_bh"] < 0.05).sum()),
        "plots": [os.path.abspath(p) for p in plots],
    }
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    diff.to_csv(os.path.join(outdir, "differential_features.csv"), index=False)

    # ---- markdown ------------------------------------------------------ #
    primary = "auroc" if task == "classification" else "spearman"
    secondary = "auprc" if task == "classification" else "r2"
    lines = []
    A = lines.append
    A(f"# {title}\n")
    A(f"- task: **{task}**, samples: {sep['n_samples']}, features: "
      f"{sep['n_features']}, groups: {sep['n_groups']}")
    A(f"- split: **{sep['split_strategy']}** ({sep['n_splits']} folds), seed {seed}\n")

    A("## Held-out performance (point [95% CI])\n")
    A(f"| model | {primary} | {secondary} |")
    A("|---|---|---|")
    for name, m in sep["features"].items():
        A(f"| features / {name} | {_fmt_ci(m, primary)} | {_fmt_ci(m, secondary)} |")
    if "covariates_only" in sep:
        A("\n### Confound control\n")
        A(f"| model | {primary} | {secondary} |")
        A("|---|---|---|")
        for name, m in sep["covariates_only"].items():
            A(f"| covariates-only / {name} | {_fmt_ci(m, primary)} | {_fmt_ci(m, secondary)} |")
        for name, m in sep["features_residualized"].items():
            A(f"| features-residualized / {name} | {_fmt_ci(m, primary)} | {_fmt_ci(m, secondary)} |")
        A("\n*Interpretation:* features add signal beyond the covariates only "
          "if **features-residualized** stays above chance and the raw "
          "**features** model beats **covariates-only**.\n")

    A("## Permutation test (label-shuffle null)\n")
    A(f"- metric: **{perm['metric']}** ({perm['model']}); permute within "
      f"groups: {perm['permute_within_groups']}")
    A(f"- observed = **{perm['observed']:.3f}**, null = "
      f"{perm['null_mean']:.3f} ± {perm['null_std']:.3f} "
      f"(n={perm['n_perm_effective']})")
    A(f"- **p-value = {perm['p_value']:.4g}**\n")

    A("## Differential features (BH-FDR)\n")
    A(f"- significant at FDR<0.05: **{results['n_significant_features_fdr05']}** "
      f"of {sep['n_features']}\n")
    A(diff.head(15).to_markdown(index=False))
    A("")

    A("## Plots\n")
    for p in plots:
        A(f"- ![{os.path.basename(p)}]({os.path.basename(p)})")

    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return results


# --------------------------------------------------------------------------- #
# Self-test (synthetic data with known signal + a confound)
# --------------------------------------------------------------------------- #
def _selftest():  # pragma: no cover - exercised via __main__
    import tempfile
    rng = np.random.default_rng(42)
    n, p = 400, 30
    n_groups = 40
    groups = rng.integers(0, n_groups, size=n)
    # group-level random effect so naive splits would leak
    g_effect = rng.normal(0, 1, size=n_groups)[groups]

    def report(name, X, y, cov, task="classification"):
        d = tempfile.mkdtemp(prefix=f"selftest_{name}_")
        r = run_report(X, y, groups=groups, covariates=cov, outdir=d,
                       title=name, task=task, seed=0, n_perm=300, n_splits=5)
        return r, d

    print("=" * 70)
    print("CASE 1: real signal in features (should beat null; low p)")
    y = (g_effect + rng.normal(0, 1, n) > 0).astype(int)
    signal = np.where(y[:, None] == 1, 1.2, -1.2) * np.ones((n, 3))
    X = np.column_stack([signal + rng.normal(0, 1, (n, 3)),
                         rng.normal(0, 1, (n, p - 3))])
    r1, _ = report("signal", X, y, cov=None)
    auc1 = r1["separation"]["features"]["l2_logreg"]["auroc"]
    p1 = r1["permutation_test"]["p_value"]
    print(f"  AUROC={auc1:.3f}  perm p={p1:.4g}  sig_feats={r1['n_significant_features_fdr05']}")
    assert auc1 > 0.65, f"expected signal AUROC>0.65, got {auc1}"
    assert p1 < 0.05, f"expected low p, got {p1}"

    print("CASE 2: pure noise (should NOT beat null; high p)")
    y2 = rng.integers(0, 2, n)
    Xn = rng.normal(0, 1, (n, p))
    r2, _ = report("noise", Xn, y2, cov=None)
    auc2 = r2["separation"]["features"]["l2_logreg"]["auroc"]
    p2 = r2["permutation_test"]["p_value"]
    print(f"  AUROC={auc2:.3f}  perm p={p2:.4g}  sig_feats={r2['n_significant_features_fdr05']}")
    assert p2 > 0.05, f"expected high p for noise, got {p2}"

    print("CASE 3: confounded (label driven by covariate; covariate-only "
          "baseline should explain it, residualized features should NOT)")
    cov = (g_effect + rng.normal(0, 0.5, n)).reshape(-1, 1)  # the confound
    y3 = (cov[:, 0] > np.median(cov[:, 0])).astype(int)
    # X is just a noisy copy of the covariate -> looks predictive but isn't
    # independent signal
    Xc = np.column_stack([cov[:, 0:1] + rng.normal(0, 0.3, (n, 3)),
                          rng.normal(0, 1, (n, p - 3))])
    r3, _ = report("confound", Xc, y3, cov=cov)
    sep3 = r3["separation"]
    auc_feat = sep3["features"]["l2_logreg"]["auroc"]
    auc_cov = sep3["covariates_only"]["l2_logreg"]["auroc"]
    auc_resid = sep3["features_residualized"]["l2_logreg"]["auroc"]
    print(f"  features AUROC      ={auc_feat:.3f}")
    print(f"  covariate-only AUROC={auc_cov:.3f}")
    print(f"  residualized AUROC  ={auc_resid:.3f}")
    assert auc_cov > 0.8, f"covariate-only should be high, got {auc_cov}"
    assert auc_resid < auc_feat - 0.1, (
        "residualizing the confound should collapse the apparent signal; "
        f"feat={auc_feat}, resid={auc_resid}")

    print("CASE 4: regression sanity (continuous y with signal)")
    yc = X[:, 0] * 1.5 + rng.normal(0, 0.5, n)
    r4, _ = report("regression", X, yc, cov=None, task="regression")
    sp = r4["separation"]["features"]["ridge"]["spearman"]
    p4 = r4["permutation_test"]["p_value"]
    print(f"  Spearman={sp:.3f}  perm p={p4:.4g}")
    assert sp > 0.5 and p4 < 0.05

    print("=" * 70)
    print("ALL SELF-TESTS PASSED")
    print(f"  UMAP available: {_HAVE_UMAP}")


if __name__ == "__main__":
    _selftest()
