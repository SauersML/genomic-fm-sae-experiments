#!/usr/bin/env python3
"""Ancient-DNA selection analysis for Evo2 layer-26 Goodfire SAE features.

Primary split is fixed by the pilot table: train = non-chr1/chr2, test =
chr1/chr2. Models use train-only SVD projections of SAE features; differential
feature tests use the original SAE feature columns.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
DATADIR = ROOT / "data" / "ancient_selection"
RESDIR = ROOT / "results" / "ancient_selection"
PLOTDIR = ROOT / "plots" / "ancient"
DOC = ROOT / "docs" / "RESULTS_ANCIENT_SELECTION.md"

COVARIATES = [
    "gc",
    "repeat_frac",
    "gene_density",
    "recomb_rate_cm_per_mb",
    "b_statistic",
    "dist_nearest_tss",
]
MATCH_COVARIATES = [
    "b_statistic",
    "recomb_rate_cm_per_mb",
    "gc",
    "repeat_frac",
    "gene_density",
    "derived_allele_freq_for_match",
]
SEED = 20260603


@dataclass
class FitResult:
    model: str
    feature_set: str
    task: str
    n_train: int
    n_test: int
    train_pos: int | None
    test_pos: int | None
    metrics: dict
    predictions: np.ndarray
    y_test: np.ndarray
    perm: dict
    svd_components: int | None = None


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def clean_id(value: object, idx: int) -> str:
    sid = str(value).strip()
    if sid and sid.lower() != "nan" and sid != ".":
        return sid
    return f"ancient_snp_{idx:05d}"


def load_feature_set(prefix: str) -> tuple[np.ndarray, list[str], dict]:
    if prefix == "region":
        fpath = DATADIR / "region_features.npy"
        ipath = DATADIR / "region_ids.txt"
        mpath = DATADIR / "region_meta.json"
    elif prefix == "delta":
        fpath = DATADIR / "delta_features.npy"
        ipath = DATADIR / "delta_ids.txt"
        mpath = DATADIR / "delta_meta.json"
    else:
        raise ValueError(prefix)
    missing = [p for p in (fpath, ipath, mpath) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing feature artifacts: " + ", ".join(map(str, missing)))
    X = np.load(fpath, mmap_mode="r")
    ids = read_ids(ipath)
    meta = json.loads(mpath.read_text())
    if X.ndim != 2 or X.shape[0] != len(ids):
        raise ValueError(f"{prefix}: feature/id shape mismatch: {X.shape}, ids={len(ids)}")
    if X.shape[1] != 32768:
        raise ValueError(f"{prefix}: expected 32768 SAE features, got {X.shape[1]}")
    probe = np.asarray(X[: min(128, X.shape[0])])
    if not np.isfinite(probe).all() or np.any(np.linalg.norm(probe, axis=1) == 0):
        raise ValueError(f"{prefix}: non-finite or zero feature rows in leading sample")
    return X, ids, meta


def align_features(X: np.ndarray, ids: list[str], df: pd.DataFrame) -> np.ndarray:
    pos = {sid: i for i, sid in enumerate(ids)}
    wanted = [clean_id(v, i) for i, v in enumerate(df["rsid"].tolist())]
    missing = [sid for sid in wanted if sid not in pos]
    if missing:
        raise ValueError(f"{len(missing)} SNP ids missing from features; first={missing[:5]}")
    return np.asarray(X[[pos[sid] for sid in wanted]], dtype=np.float32)


def split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train = df["split"].to_numpy() == "train"
    test = df["split"].to_numpy() == "test"
    if not train.any() or not test.any():
        raise ValueError("pilot split must contain train and test rows")
    return train, test


def impute_scale_covariates(df: pd.DataFrame, train: np.ndarray) -> np.ndarray:
    C = df[COVARIATES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(C[train], axis=0)
    bad = ~np.isfinite(C)
    if bad.any():
        C[bad] = np.take(med, np.where(bad)[1])
    return C


def fit_svd(train_X: np.ndarray, test_X: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    k = min(int(n_components), train_X.shape[0] - 1, train_X.shape[1] - 1)
    if k < 2:
        raise ValueError(f"not enough rows/features for SVD: train shape={train_X.shape}")
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Ztr = svd.fit_transform(train_X)
    Zte = svd.transform(test_X)
    return Ztr, Zte, k


def residualize_train_test(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    Ctr: np.ndarray,
    Cte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Btr = np.column_stack([np.ones(len(Ctr)), Ctr])
    beta, *_ = np.linalg.lstsq(Btr, Xtr, rcond=None)
    Bte = np.column_stack([np.ones(len(Cte)), Cte])
    return Xtr - Btr @ beta, Xte - Bte @ beta


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    rho = stats.spearmanr(y, pred)[0]
    return {"spearman": float(rho), "r2": float(r2_score(y, pred))}


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    out = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "prevalence": float(np.mean(y)),
    }
    if np.unique(y).size == 2:
        out["auroc"] = float(roc_auc_score(y, score))
        out["auprc"] = float(average_precision_score(y, score))
    return out


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    metric: str,
    task: str,
    n_boot: int,
    seed: int,
) -> list[float | None]:
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pp = pred[idx]
        if task == "classification" and np.unique(yy).size < 2:
            continue
        try:
            if metric == "spearman":
                v = stats.spearmanr(yy, pp)[0]
            elif metric == "r2":
                v = r2_score(yy, pp)
            elif metric == "auroc":
                v = roc_auc_score(yy, pp)
            elif metric == "auprc":
                v = average_precision_score(yy, pp)
            else:
                raise ValueError(metric)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return [None, None]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def add_cis(metrics: dict, y: np.ndarray, pred: np.ndarray, task: str, n_boot: int, seed: int) -> dict:
    out = dict(metrics)
    keys = ("auroc", "auprc") if task == "classification" else ("spearman", "r2")
    for i, key in enumerate(keys):
        out[f"{key}_ci95"] = bootstrap_ci(y, pred, key, task, n_boot, seed + i)
    return out


def permutation_p(
    fit_fn,
    Xtr,
    Xte,
    ytr,
    yte,
    task: str,
    metric: str,
    observed: float,
    n_perm: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(ytr)
        pred = fit_fn(Xtr, Xte, yp)
        try:
            if task == "classification":
                if np.unique(yte).size < 2:
                    v = np.nan
                elif metric == "auroc":
                    v = roc_auc_score(yte, pred)
                else:
                    v = average_precision_score(yte, pred)
            elif metric == "spearman":
                v = stats.spearmanr(yte, pred)[0]
            else:
                v = r2_score(yte, pred)
        except ValueError:
            v = np.nan
        if np.isfinite(v):
            null.append(float(v))
    arr = np.asarray(null, dtype=float)
    if arr.size == 0 or not np.isfinite(observed):
        p = float("nan")
    else:
        p = float((np.sum(arr >= observed) + 1) / (arr.size + 1))
    return {
        "metric": metric,
        "observed": float(observed),
        "null_mean": float(np.mean(arr)) if arr.size else float("nan"),
        "null_std": float(np.std(arr)) if arr.size else float("nan"),
        "n_perm_effective": int(arr.size),
        "p_value_greater": p,
    }


def regression_fit_predict(model_name: str, Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    if model_name == "ridge":
        model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    elif model_name == "hist_gbt":
        model = HistGradientBoostingRegressor(random_state=SEED, l2_regularization=0.1)
    else:
        raise ValueError(model_name)
    model.fit(Xtr, ytr)
    return np.asarray(model.predict(Xte), dtype=float)


def classification_fit_predict(model_name: str, Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    if model_name == "l2_logreg":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.25,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    elif model_name == "hist_gbt":
        model = HistGradientBoostingClassifier(random_state=SEED, l2_regularization=0.1)
    else:
        raise ValueError(model_name)
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(Xte)[:, 1], dtype=float)
    return np.asarray(model.decision_function(Xte), dtype=float)


def run_regression_block(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    n_components: int,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> list[FitResult]:
    results: list[FitResult] = []
    Xtr_raw = np.asarray(X[train], dtype=np.float32)
    Xte_raw = np.asarray(X[test], dtype=np.float32)
    Ctr = C[train]
    Cte = C[test]
    ytr = y[train]
    yte = y[test]

    for feature_mode in ("features", "features_residualized"):
        if feature_mode == "features":
            Xtr_use, Xte_use = Xtr_raw, Xte_raw
        else:
            Xtr_use, Xte_use = residualize_train_test(Xtr_raw, Xte_raw, Ctr, Cte)
            Xtr_use = Xtr_use.astype(np.float32, copy=False)
            Xte_use = Xte_use.astype(np.float32, copy=False)
        Ztr, Zte, k = fit_svd(Xtr_use, Xte_use, n_components, seed)
        for model_name in ("ridge", "hist_gbt"):
            pred = regression_fit_predict(model_name, Ztr, Zte, ytr)
            metrics = add_cis(regression_metrics(yte, pred), yte, pred, "regression", n_boot, seed)
            perm_n = n_perm if model_name == "ridge" else 0
            perm = permutation_p(
                lambda a, b, yy: regression_fit_predict(model_name, a, b, yy),
                Ztr,
                Zte,
                ytr,
                yte,
                "regression",
                "spearman",
                metrics["spearman"],
                perm_n,
                seed + 17,
            )
            results.append(
                FitResult(model_name, f"{name}_{feature_mode}", "regression", int(train.sum()), int(test.sum()), None, None, metrics, pred, yte, perm, k)
            )

    for model_name in ("ridge", "hist_gbt"):
        pred = regression_fit_predict(model_name, Ctr, Cte, ytr)
        metrics = add_cis(regression_metrics(yte, pred), yte, pred, "regression", n_boot, seed)
        perm_n = n_perm if model_name == "ridge" else 0
        perm = permutation_p(
            lambda a, b, yy: regression_fit_predict(model_name, a, b, yy),
            Ctr,
            Cte,
            ytr,
            yte,
            "regression",
            "spearman",
            metrics["spearman"],
            perm_n,
            seed + 23,
        )
        results.append(FitResult(model_name, f"{name}_covariates_only", "regression", int(train.sum()), int(test.sum()), None, None, metrics, pred, yte, perm, None))
    return results


def run_classification_block(
    name: str,
    X: np.ndarray,
    y_all: np.ndarray,
    C: np.ndarray,
    mask: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    n_components: int,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> list[FitResult]:
    idx_train = np.where(mask & train)[0]
    idx_test = np.where(mask & test)[0]
    ytr = y_all[idx_train].astype(int)
    yte = y_all[idx_test].astype(int)
    if np.unique(ytr).size < 2 or len(idx_test) == 0:
        raise ValueError(f"{name}: classification split lacks both train classes or test rows")

    Xtr_raw = np.asarray(X[idx_train], dtype=np.float32)
    Xte_raw = np.asarray(X[idx_test], dtype=np.float32)
    Ctr = C[idx_train]
    Cte = C[idx_test]
    results: list[FitResult] = []

    for feature_mode in ("features", "features_residualized"):
        if feature_mode == "features":
            Xtr_use, Xte_use = Xtr_raw, Xte_raw
        else:
            Xtr_use, Xte_use = residualize_train_test(Xtr_raw, Xte_raw, Ctr, Cte)
            Xtr_use = Xtr_use.astype(np.float32, copy=False)
            Xte_use = Xte_use.astype(np.float32, copy=False)
        Ztr, Zte, k = fit_svd(Xtr_use, Xte_use, n_components, seed)
        for model_name in ("l2_logreg", "hist_gbt"):
            score = classification_fit_predict(model_name, Ztr, Zte, ytr)
            metrics = add_cis(classification_metrics(yte, score), yte, score, "classification", n_boot, seed)
            perm_n = n_perm if model_name == "l2_logreg" else 0
            perm = permutation_p(
                lambda a, b, yy: classification_fit_predict(model_name, a, b, yy),
                Ztr,
                Zte,
                ytr,
                yte,
                "classification",
                "auroc",
                metrics["auroc"],
                perm_n,
                seed + 31,
            )
            results.append(
                FitResult(
                    model_name,
                    f"{name}_{feature_mode}",
                    "classification",
                    int(len(idx_train)),
                    int(len(idx_test)),
                    int(ytr.sum()),
                    int(yte.sum()),
                    metrics,
                    score,
                    yte,
                    perm,
                    k,
                )
            )

    for model_name in ("l2_logreg", "hist_gbt"):
        score = classification_fit_predict(model_name, Ctr, Cte, ytr)
        metrics = add_cis(classification_metrics(yte, score), yte, score, "classification", n_boot, seed)
        perm_n = n_perm if model_name == "l2_logreg" else 0
        perm = permutation_p(
            lambda a, b, yy: classification_fit_predict(model_name, a, b, yy),
            Ctr,
            Cte,
            ytr,
            yte,
            "classification",
            "auroc",
            metrics["auroc"],
            perm_n,
            seed + 37,
        )
        results.append(
            FitResult(
                model_name,
                f"{name}_covariates_only",
                "classification",
                int(len(idx_train)),
                int(len(idx_test)),
                int(ytr.sum()),
                int(yte.sum()),
                metrics,
                score,
                yte,
                perm,
                None,
            )
        )
    return results


def matched_mask(df: pd.DataFrame, train: np.ndarray, test: np.ndarray, seed: int) -> np.ndarray:
    eligible = df["label_binary"].notna().to_numpy()
    y = df["label_binary"].fillna(-1).to_numpy()
    out = np.zeros(len(df), dtype=bool)
    rng = np.random.default_rng(seed)
    for split_mask in (train, test):
        pos = np.where(eligible & split_mask & (y == 1))[0]
        neg = np.where(eligible & split_mask & (y == 0))[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(len(pos), len(neg))
        cov = np.array(df[MATCH_COVARIATES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), copy=True)
        med = np.nanmedian(cov[eligible & split_mask], axis=0)
        bad = ~np.isfinite(cov)
        cov[bad] = np.take(med, np.where(bad)[1])
        scaler = StandardScaler().fit(cov[np.concatenate([pos, neg])])
        P = scaler.transform(cov[pos])
        N = scaler.transform(cov[neg])
        nn = NearestNeighbors(n_neighbors=min(len(neg), max(1, len(neg))), metric="euclidean").fit(N)
        order = rng.permutation(len(pos))
        used_neg: set[int] = set()
        used_pos: list[int] = []
        chosen_neg: list[int] = []
        neigh = nn.kneighbors(P, return_distance=False)
        for pi in order:
            for ni_local in neigh[pi]:
                ni = int(neg[ni_local])
                if ni not in used_neg:
                    used_neg.add(ni)
                    chosen_neg.append(ni)
                    used_pos.append(int(pos[pi]))
                    break
            if len(chosen_neg) == k:
                break
        out[used_pos] = True
        out[chosen_neg] = True
    return out


def fitresult_to_dict(r: FitResult) -> dict:
    return {
        "model": r.model,
        "feature_set": r.feature_set,
        "task": r.task,
        "n_train": r.n_train,
        "n_test": r.n_test,
        "train_pos": r.train_pos,
        "test_pos": r.test_pos,
        "metrics": r.metrics,
        "permutation": r.perm,
        "svd_components": r.svd_components,
    }


def json_clean(obj):
    if isinstance(obj, dict):
        return {k: json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def differential_features(
    X: np.ndarray,
    y_abs: np.ndarray,
    outpath: Path,
    prefix: str,
    chunk: int = 512,
) -> pd.DataFrame:
    yr = stats.rankdata(y_abs)
    yr = (yr - yr.mean()) / yr.std(ddof=1)
    rows = []
    n = len(y_abs)
    for start in range(0, X.shape[1], chunk):
        stop = min(start + chunk, X.shape[1])
        block = np.asarray(X[:, start:stop], dtype=float)
        ranks = np.apply_along_axis(stats.rankdata, 0, block)
        ranks -= ranks.mean(axis=0)
        std = ranks.std(axis=0, ddof=1)
        rho = np.divide(yr @ ranks, (n - 1) * std, out=np.zeros_like(std), where=std > 0)
        rho = np.clip(rho, -0.999999, 0.999999)
        t = rho * np.sqrt((n - 2) / np.maximum(1e-12, 1 - rho * rho))
        p = 2 * stats.t.sf(np.abs(t), df=n - 2)
        for j, (rr, pp) in enumerate(zip(rho, p)):
            rows.append((f"{prefix}_sae_{start + j}", int(start + j), float(rr), float(pp)))
    df = pd.DataFrame(rows, columns=["feature", "feature_index", "spearman_abs_selection", "p_value"])
    _, q, _, _ = multipletests(df["p_value"].to_numpy(), method="fdr_bh")
    df["p_adj_bh"] = q
    df = df.sort_values("p_adj_bh", kind="mergesort").reset_index(drop=True)
    df.to_csv(outpath, index=False)
    return df


def plot_predicted_vs_true(results: list[FitResult], outpath: Path) -> None:
    keep = [r for r in results if r.task == "regression" and r.model == "ridge" and r.feature_set.endswith("_features")]
    if not keep:
        return
    n = len(keep)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, r in zip(axes[0], keep):
        ax.scatter(r.y_test, r.predictions, s=18, alpha=0.75, color="#356f8c", edgecolor="none")
        ax.axhline(np.mean(r.predictions), color="#999999", lw=0.8)
        ax.axvline(0, color="#cccccc", lw=0.8)
        m = r.metrics
        ax.set_title(r.feature_set.replace("_features", "").replace("_", " "))
        ax.set_xlabel("True selection coefficient")
        ax.set_ylabel("Predicted")
        ax.text(
            0.03,
            0.97,
            f"Spearman {m['spearman']:.3f}\nR2 {m['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_metric_forest(results: list[FitResult], outpath: Path) -> None:
    rows = []
    for r in results:
        if r.model not in ("ridge", "l2_logreg"):
            continue
        if r.task == "regression":
            metric = "spearman"
            label = r.feature_set.replace("_", " ")
        else:
            metric = "auroc"
            label = r.feature_set.replace("_", " ")
        ci = r.metrics.get(f"{metric}_ci95", [None, None])
        rows.append((r.task, label, metric, r.metrics.get(metric), ci[0], ci[1]))
    if not rows:
        return
    fig_h = max(5, 0.25 * len(rows))
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    y = np.arange(len(rows))
    vals = np.array([r[3] for r in rows], dtype=float)
    lo = np.array([np.nan if r[4] is None else r[4] for r in rows], dtype=float)
    hi = np.array([np.nan if r[5] is None else r[5] for r in rows], dtype=float)
    colors = ["#356f8c" if r[0] == "regression" else "#9a5a2e" for r in rows]
    xerr = np.vstack([np.maximum(0, vals - lo), np.maximum(0, hi - vals)])
    ax.errorbar(vals, y, xerr=xerr, fmt="none", ecolor="#555555", elinewidth=1, capsize=2, alpha=0.8)
    ax.scatter(vals, y, c=colors, s=32, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]} | {r[1]}" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#bbbbbb", lw=0.8)
    ax.axvline(0.5, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_xlabel("Spearman for regression; AUROC for classification")
    ax.set_title("Held-out chr1/chr2 performance")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_matched_collapse(results: list[FitResult], outpath: Path) -> None:
    rows = []
    for r in results:
        if r.task == "classification" and r.model == "l2_logreg" and (
            r.feature_set.endswith("_features") or r.feature_set.endswith("_features_residualized")
        ):
            rows.append(r)
    if not rows:
        return
    labels = [r.feature_set.replace("_", " ") for r in rows]
    vals = [r.metrics["auroc"] for r in rows]
    colors = ["#356f8c" if "matched" not in r.feature_set else "#c77738" for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.32 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors)
    ax.axvline(0.5, color="#444444", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC")
    ax.set_title("Raw vs matched-control classification")
    for yi, v, r in zip(y, vals, rows):
        ax.text(min(0.98, v + 0.02), yi, f"{v:.3f} (test pos={r.test_pos})", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def fmt_ci(metrics: dict, key: str) -> str:
    v = metrics.get(key, float("nan"))
    lo, hi = metrics.get(f"{key}_ci95", [None, None])
    if lo is None or hi is None:
        return f"{v:.3f} [NA, NA]"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def primary(results: list[FitResult], feature_set: str, model: str) -> FitResult:
    for r in results:
        if r.feature_set == feature_set and r.model == model:
            return r
    raise KeyError((feature_set, model))


def write_doc(summary: dict, results: list[FitResult], diff_region: pd.DataFrame, diff_delta: pd.DataFrame) -> None:
    region_raw = primary(results, "region_features", "ridge")
    region_cov = primary(results, "region_covariates_only", "ridge")
    region_res = primary(results, "region_features_residualized", "ridge")
    delta_raw = primary(results, "delta_features", "ridge")
    delta_cov = primary(results, "delta_covariates_only", "ridge")
    delta_res = primary(results, "delta_features_residualized", "ridge")
    cls_raw = primary(results, "region_classification_features", "l2_logreg")
    cls_matched = primary(results, "region_matched_classification_features", "l2_logreg")
    cls_res = primary(results, "region_classification_features_residualized", "l2_logreg")

    def perm(r: FitResult) -> str:
        return f"{r.perm['p_value_greater']:.4g}"

    verdict = (
        "No robust evidence that these Evo2 SAE features predict ancient selection "
        "coefficients beyond the obvious covariates."
    )
    if (
        region_raw.metrics["spearman"] > region_cov.metrics["spearman"]
        and region_res.metrics["spearman_ci95"][0] is not None
        and region_res.metrics["spearman_ci95"][0] > 0
        and region_res.perm["p_value_greater"] < 0.05
    ):
        verdict = (
            "Region SAE features retain a positive held-out signal after covariate "
            "residualization, but this should be read against the classification "
            "imbalance on chr1/chr2."
        )

    lines = [
        "# Ancient-DNA Selection: Evo2 Layer-26 Goodfire SAE",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "The fixed held-out test set is `chr1`/`chr2` (502 SNPs). Classification is especially fragile because the held-out selected/control subset has only 2 selected positives.",
        "",
        "## Regression: selection coefficient",
        "",
        "| feature set | model | held-out Spearman | held-out R2 | permutation p |",
        "|---|---|---:|---:|---:|",
        f"| region SAE | ridge | {fmt_ci(region_raw.metrics, 'spearman')} | {fmt_ci(region_raw.metrics, 'r2')} | {perm(region_raw)} |",
        f"| region covariates only | ridge | {fmt_ci(region_cov.metrics, 'spearman')} | {fmt_ci(region_cov.metrics, 'r2')} | {perm(region_cov)} |",
        f"| region SAE residualized | ridge | {fmt_ci(region_res.metrics, 'spearman')} | {fmt_ci(region_res.metrics, 'r2')} | {perm(region_res)} |",
        f"| ref/alt delta SAE | ridge | {fmt_ci(delta_raw.metrics, 'spearman')} | {fmt_ci(delta_raw.metrics, 'r2')} | {perm(delta_raw)} |",
        f"| delta covariates only | ridge | {fmt_ci(delta_cov.metrics, 'spearman')} | {fmt_ci(delta_cov.metrics, 'r2')} | {perm(delta_cov)} |",
        f"| delta SAE residualized | ridge | {fmt_ci(delta_res.metrics, 'spearman')} | {fmt_ci(delta_res.metrics, 'r2')} | {perm(delta_res)} |",
        "",
        "Covariates are `gc`, `repeat_frac`, `gene_density`, `recomb_rate_cm_per_mb`, `b_statistic`, and `dist_nearest_tss`.",
        "",
        "## Classification: strongly selected vs controls",
        "",
        "| contrast | model | held-out AUROC | held-out AUPRC | train/test positives | permutation p |",
        "|---|---|---:|---:|---:|---:|",
        f"| raw selected/control, region SAE | l2 logreg | {fmt_ci(cls_raw.metrics, 'auroc')} | {fmt_ci(cls_raw.metrics, 'auprc')} | {cls_raw.train_pos}/{cls_raw.test_pos} | {perm(cls_raw)} |",
        f"| raw selected/control, region SAE residualized | l2 logreg | {fmt_ci(cls_res.metrics, 'auroc')} | {fmt_ci(cls_res.metrics, 'auprc')} | {cls_res.train_pos}/{cls_res.test_pos} | {perm(cls_res)} |",
        f"| matched controls, region SAE | l2 logreg | {fmt_ci(cls_matched.metrics, 'auroc')} | {fmt_ci(cls_matched.metrics, 'auprc')} | {cls_matched.train_pos}/{cls_matched.test_pos} | {perm(cls_matched)} |",
        "",
        "The matched-control check uses nearest-neighbor matching within train and test splits on `b_statistic`, recombination, GC, repeat fraction, gene density, and derived allele frequency (falling back to `matching_af` only where DAF is missing).",
        "",
        "## Differential SAE features",
        "",
        f"- Region features associated with `|selection_coefficient|` at BH-FDR < 0.05: **{int((diff_region['p_adj_bh'] < 0.05).sum())}** / 32768.",
        f"- Delta features associated with `|selection_coefficient|` at BH-FDR < 0.05: **{int((diff_delta['p_adj_bh'] < 0.05).sum())}** / 32768.",
        "",
        "Top region associations:",
        "",
        diff_region.head(10).to_markdown(index=False),
        "",
        "Top delta associations:",
        "",
        diff_delta.head(10).to_markdown(index=False),
        "",
        "## Artifacts",
        "",
        "- `results/ancient_selection/summary.json`",
        "- `results/ancient_selection/differential_region_abs_selection.csv`",
        "- `results/ancient_selection/differential_delta_abs_selection.csv`",
        "- `plots/ancient/predicted_vs_true_selection.png`",
        "- `plots/ancient/metric_forest.png`",
        "- `plots/ancient/matched_control_collapse.png`",
        "",
        "## Feature Metadata",
        "",
        "```json",
        json.dumps(summary["feature_meta"], indent=2),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-components", type=int, default=128)
    ap.add_argument("--n-perm", type=int, default=250)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    RESDIR.mkdir(parents=True, exist_ok=True)
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATADIR / "snps_pilot.tsv", sep="\t")
    df["derived_allele_freq_for_match"] = pd.to_numeric(df["derived_allele_freq"], errors="coerce")
    fallback = pd.to_numeric(df["matching_af"], errors="coerce")
    df["derived_allele_freq_for_match"] = df["derived_allele_freq_for_match"].fillna(fallback)
    train, test = split_masks(df)
    C = impute_scale_covariates(df, train)
    y_reg = pd.to_numeric(df["selection_coefficient"], errors="raise").to_numpy(dtype=float)
    y_cls = df["label_binary"].to_numpy(dtype=float)
    cls_mask = np.isfinite(y_cls)

    region_X_raw, region_ids, region_meta = load_feature_set("region")
    delta_X_raw, delta_ids, delta_meta = load_feature_set("delta")
    region_X = align_features(region_X_raw, region_ids, df)
    delta_X = align_features(delta_X_raw, delta_ids, df)

    all_results: list[FitResult] = []
    all_results.extend(run_regression_block("region", region_X, y_reg, C, train, test, args.n_components, args.n_perm, args.n_boot, SEED))
    all_results.extend(run_regression_block("delta", delta_X, y_reg, C, train, test, args.n_components, args.n_perm, args.n_boot, SEED + 1))

    all_results.extend(run_classification_block("region_classification", region_X, y_cls, C, cls_mask, train, test, args.n_components, args.n_perm, args.n_boot, SEED + 2))
    all_results.extend(run_classification_block("delta_classification", delta_X, y_cls, C, cls_mask, train, test, args.n_components, args.n_perm, args.n_boot, SEED + 3))

    mmask = matched_mask(df, train, test, SEED + 4)
    all_results.extend(run_classification_block("region_matched_classification", region_X, y_cls, C, mmask, train, test, args.n_components, args.n_perm, args.n_boot, SEED + 5))
    all_results.extend(run_classification_block("delta_matched_classification", delta_X, y_cls, C, mmask, train, test, args.n_components, args.n_perm, args.n_boot, SEED + 6))

    diff_region = differential_features(region_X, np.abs(y_reg), RESDIR / "differential_region_abs_selection.csv", "region")
    diff_delta = differential_features(delta_X, np.abs(y_reg), RESDIR / "differential_delta_abs_selection.csv", "delta")

    plot_predicted_vs_true(all_results, PLOTDIR / "predicted_vs_true_selection.png")
    plot_metric_forest(all_results, PLOTDIR / "metric_forest.png")
    plot_matched_collapse(all_results, PLOTDIR / "matched_control_collapse.png")

    summary = {
        "split": {
            "train": int(train.sum()),
            "test": int(test.sum()),
            "test_chromosomes": sorted(df.loc[test, "chrom"].unique().tolist()),
            "classification_train_rows": int((cls_mask & train).sum()),
            "classification_test_rows": int((cls_mask & test).sum()),
            "classification_train_pos": int(np.nansum(y_cls[cls_mask & train])),
            "classification_test_pos": int(np.nansum(y_cls[cls_mask & test])),
            "matched_train_rows": int((mmask & train).sum()),
            "matched_test_rows": int((mmask & test).sum()),
            "matched_train_pos": int(np.nansum(y_cls[mmask & train])),
            "matched_test_pos": int(np.nansum(y_cls[mmask & test])),
        },
        "covariates": COVARIATES,
        "match_covariates": MATCH_COVARIATES,
        "feature_meta": {"region": region_meta, "delta": delta_meta},
        "results": [fitresult_to_dict(r) for r in all_results],
        "differential": {
            "region_fdr05": int((diff_region["p_adj_bh"] < 0.05).sum()),
            "delta_fdr05": int((diff_delta["p_adj_bh"] < 0.05).sum()),
        },
    }
    clean_summary = json_clean(summary)
    (RESDIR / "summary.json").write_text(json.dumps(clean_summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame([json_clean(fitresult_to_dict(r)) for r in all_results]).to_json(RESDIR / "model_results.json", orient="records", indent=2)
    write_doc(summary, all_results, diff_region, diff_delta)

    print(json.dumps({
        "region_regression_spearman": primary(all_results, "region_features", "ridge").metrics["spearman"],
        "region_covariate_spearman": primary(all_results, "region_covariates_only", "ridge").metrics["spearman"],
        "region_residualized_spearman": primary(all_results, "region_features_residualized", "ridge").metrics["spearman"],
        "classification_auroc": primary(all_results, "region_classification_features", "l2_logreg").metrics["auroc"],
        "matched_classification_auroc": primary(all_results, "region_matched_classification_features", "l2_logreg").metrics["auroc"],
    }, indent=2))


if __name__ == "__main__":
    main()
