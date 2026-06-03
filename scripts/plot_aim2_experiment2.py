#!/usr/bin/env python3
"""Explicit held-out analysis and summary plots for Aim 2.

Train on chr3-22 and test on the predeclared held-out chromosomes chr1/chr2.
This complements the GroupKFold-by-chromosome analysis in src/aim2_popgen/analyze.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/user/bio-interp-experiments")
DATA = ROOT / "data/aim2_popgen"
RES = ROOT / "results/aim2_popgen"
PLOTS = ROOT / "plots"
TASKS = ("sweeps", "introgression")
COV_NAMES = ("log_length", "gc", "repeat_frac", "mappability", "gene_density")


def load_inputs(task: str):
    X = np.load(DATA / "features.npy", mmap_mode="r")
    ids = [line.strip() for line in (DATA / "ids.txt").read_text().splitlines() if line.strip()]
    gc = np.load(DATA / "gc.npy")
    id_to_row = {rid: i for i, rid in enumerate(ids)}
    table = pd.read_csv(DATA / f"table_{task}.tsv", sep="\t")
    extra = pd.read_csv(DATA / "covariates_extra.tsv", sep="\t").drop_duplicates("id")
    table = table.merge(extra[["id", "repeat_frac", "mappability", "gene_density"]], on="id", how="left")
    missing = sorted(set(table["id"]) - set(id_to_row))
    if missing:
        raise ValueError(f"{task}: missing feature ids, first={missing[:3]}")
    rows = [id_to_row[rid] for rid in table["id"]]
    table["gc"] = gc[rows]
    cov = table.loc[:, COV_NAMES].to_numpy(float)
    if not np.isfinite(cov).all():
        raise ValueError(f"{task}: non-finite covariates")
    return np.asarray(X[rows], dtype=np.float32), table, cov


def auroc_ci(y, score, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    score = np.asarray(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            vals.append(roc_auc_score(y[idx], score[idx]))
    return np.percentile(vals, [2.5, 97.5]).tolist()


def fit_score(X_train, y_train, X_test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def residualize(X_train, X_test, cov_train, cov_test):
    reg = LinearRegression().fit(cov_train, X_train)
    return X_train - reg.predict(cov_train), X_test - reg.predict(cov_test)


def permutation_p(X_train, y_train, X_test, y_test, observed, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y_train)
        score = fit_score(X_train, yp, X_test)
        null.append(roc_auc_score(y_test, score))
    null = np.asarray(null)
    return {
        "p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_perm": int(len(null)),
    }


def explicit_test(task: str):
    X_raw, table, cov = load_inputs(task)
    y = table["y"].to_numpy(int)
    train = table["split"].eq("train").to_numpy()
    test = table["split"].eq("test").to_numpy()
    if set(table.loc[test, "chrom"]) != {"chr1", "chr2"}:
        raise ValueError(f"{task}: expected chr1/chr2 test set")
    svd = TruncatedSVD(n_components=128, random_state=0)
    X_train = svd.fit_transform(X_raw[train])
    X_test = svd.transform(X_raw[test])
    cov_train, cov_test = cov[train], cov[test]
    y_train, y_test = y[train], y[test]

    scores = {
        "features": fit_score(X_train, y_train, X_test),
        "covariates": fit_score(cov_train, y_train, cov_test),
    }
    Xr_train, Xr_test = residualize(X_train, X_test, cov_train, cov_test)
    scores["residualized"] = fit_score(Xr_train, y_train, Xr_test)

    out = {
        "task": task,
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "test_chromosomes": sorted(table.loc[test, "chrom"].unique().tolist()),
        "class_balance_train": {str(k): int(v) for k, v in table.loc[train, "y"].value_counts().sort_index().items()},
        "class_balance_test": {str(k): int(v) for k, v in table.loc[test, "y"].value_counts().sort_index().items()},
        "feature_preprocessing": "TruncatedSVD(128) fit on train chr3-22 only",
        "covariates": list(COV_NAMES),
        "models": {},
    }
    for name, score in scores.items():
        au = float(roc_auc_score(y_test, score))
        pr = float(average_precision_score(y_test, score))
        out["models"][name] = {
            "auroc": au,
            "auroc_ci95": auroc_ci(y_test, score, seed=hash((task, name)) % (2**32)),
            "auprc": pr,
        }
        if name in ("features", "residualized"):
            Xp_train = X_train if name == "features" else Xr_train
            Xp_test = X_test if name == "features" else Xr_test
            out["models"][name]["permutation"] = permutation_p(
                Xp_train, y_train, Xp_test, y_test, au, seed=17 + len(task) + len(name)
            )
    fpr_tpr = {}
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        prec, rec, _ = precision_recall_curve(y_test, score)
        fpr_tpr[name] = {
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
            "pr_recall": rec.tolist(),
            "pr_precision": prec.tolist(),
        }
    return out, fpr_tpr


def load_cv_summary():
    out = {}
    for task in TASKS:
        obj = json.loads((RES / task / "results.json").read_text())
        sep = obj["separation"]
        out[task] = {
            "features": sep["features"]["l2_logreg"]["auroc"],
            "features_ci": sep["features"]["l2_logreg"]["auroc_ci95"],
            "covariates": sep["covariates_only"]["l2_logreg"]["auroc"],
            "covariates_ci": sep["covariates_only"]["l2_logreg"]["auroc_ci95"],
            "residualized": sep["features_residualized"]["l2_logreg"]["auroc"],
            "residualized_ci": sep["features_residualized"]["l2_logreg"]["auroc_ci95"],
        }
    return out


def plot_summary(cv, explicit, curves):
    PLOTS.mkdir(exist_ok=True)
    colors = {"features": "#d94f3d", "covariates": "#6b7785", "residualized": "#087f8c"}
    labels = {"features": "SAE", "covariates": "covariates", "residualized": "residualized"}
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.2), dpi=220)
    fig.patch.set_facecolor("#fbfcfd")
    for ax in axs.ravel():
        ax.set_facecolor("#fbfcfd")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    ax = axs[0, 0]
    xbase = np.arange(len(TASKS))
    width = 0.22
    for i, key in enumerate(("features", "covariates", "residualized")):
        vals = [cv[t][key] for t in TASKS]
        lows = [vals[j] - cv[TASKS[j]][f"{key}_ci"][0] for j in range(len(TASKS))]
        highs = [cv[TASKS[j]][f"{key}_ci"][1] - vals[j] for j in range(len(TASKS))]
        ax.bar(xbase + (i - 1) * width, vals, width=width, color=colors[key], label=labels[key], alpha=0.92)
        ax.errorbar(xbase + (i - 1) * width, vals, yerr=[lows, highs], fmt="none", ecolor="#3f4851", lw=0.8, capsize=2)
    ax.axhline(0.5, color="#9aa6b2", lw=0.9, ls=(0, (4, 4)))
    ax.set_xticks(xbase)
    ax.set_xticklabels(["sweeps", "introgression"])
    ax.set_ylim(0.45, 0.72)
    ax.set_ylabel("AUROC")
    ax.set_title("Chromosome GroupKFold", loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")

    ax = axs[0, 1]
    for i, key in enumerate(("features", "covariates", "residualized")):
        vals = [explicit[t]["models"][key]["auroc"] for t in TASKS]
        lows = [vals[j] - explicit[TASKS[j]]["models"][key]["auroc_ci95"][0] for j in range(len(TASKS))]
        highs = [explicit[TASKS[j]]["models"][key]["auroc_ci95"][1] - vals[j] for j in range(len(TASKS))]
        ax.bar(xbase + (i - 1) * width, vals, width=width, color=colors[key], label=labels[key], alpha=0.92)
        ax.errorbar(xbase + (i - 1) * width, vals, yerr=[lows, highs], fmt="none", ecolor="#3f4851", lw=0.8, capsize=2)
    ax.axhline(0.5, color="#9aa6b2", lw=0.9, ls=(0, (4, 4)))
    ax.set_xticks(xbase)
    ax.set_xticklabels(["sweeps", "introgression"])
    ax.set_ylim(0.35, 0.78)
    ax.set_ylabel("AUROC")
    ax.set_title("Explicit test chromosomes chr1/chr2", loc="left", fontweight="semibold")

    for ax, task in zip(axs[1], TASKS):
        y_test_n = explicit[task]["test_n"]
        for key in ("features", "covariates", "residualized"):
            c = curves[task][key]
            ax.plot(c["roc_fpr"], c["roc_tpr"], color=colors[key], lw=1.8, label=f"{labels[key]} {explicit[task]['models'][key]['auroc']:.2f}")
        ax.plot([0, 1], [0, 1], color="#9aa6b2", lw=0.9, ls=(0, (4, 4)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        title = "Sweep ROC, chr1/chr2 test" if task == "sweeps" else "Introgression ROC, chr1/chr2 test"
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(pad=1.6, w_pad=2.2, h_pad=2.2)
    out = PLOTS / "aim2_experiment2_summary.png"
    fig.savefig(out, bbox_inches="tight")
    return out


def plot_umap():
    import umap

    X = np.load(DATA / "features.npy", mmap_mode="r")
    ids = [line.strip() for line in (DATA / "ids.txt").read_text().splitlines() if line.strip()]
    id_to_row = {rid: i for i, rid in enumerate(ids)}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=220)
    fig.patch.set_facecolor("#fbfcfd")
    for ax, task in zip(axs, TASKS):
        table = pd.read_csv(DATA / f"table_{task}.tsv", sep="\t")
        rows = [id_to_row[rid] for rid in table["id"]]
        svd = TruncatedSVD(n_components=50, random_state=0).fit_transform(np.asarray(X[rows], dtype=np.float32))
        emb = umap.UMAP(n_neighbors=25, min_dist=0.18, metric="euclidean", random_state=0).fit_transform(svd)
        y = table["y"].to_numpy(int)
        split = table["split"].to_numpy()
        ax.set_facecolor("#fbfcfd")
        for val, name, color in [(0, "control", "#7b8794"), (1, "positive", "#d94f3d" if task == "sweeps" else "#087f8c")]:
            m = y == val
            ax.scatter(emb[m, 0], emb[m, 1], s=16, c=color, alpha=0.58 if val else 0.32, linewidths=0, label=name)
        test = split == "test"
        ax.scatter(emb[test, 0], emb[test, 1], s=34, facecolors="none", edgecolors="#111827", linewidths=0.55, alpha=0.6, label="chr1/2 test")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9aa6b2")
        ax.spines["bottom"].set_color("#9aa6b2")
        ax.set_title("Sweeps vs controls" if task == "sweeps" else "Introgression vs controls", loc="left", fontweight="semibold")
        ax.legend(frameon=False, fontsize=8, loc="best")
    out = PLOTS / "aim2_experiment2_umap.png"
    fig.tight_layout(pad=1.4, w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    return out


def write_doc(explicit, summary_plot, umap_plot):
    doc = ROOT / "docs/RESULTS_AIM2.md"
    lines = doc.read_text().rstrip().splitlines()
    lines.extend([
        "",
        "## Explicit held-out chr1/chr2 test",
        "",
        "The fixed test set is chr1+chr2. The model is trained on chr3-22 only; SVD is fit only on the training chromosomes.",
        "",
    ])
    for task in TASKS:
        vals = explicit[task]["models"]
        lines.append(f"- {task}: SAE AUROC {vals['features']['auroc']:.3f} [{vals['features']['auroc_ci95'][0]:.3f}, {vals['features']['auroc_ci95'][1]:.3f}], "
                     f"covariates AUROC {vals['covariates']['auroc']:.3f}, residualized AUROC {vals['residualized']['auroc']:.3f}; "
                     f"feature permutation p={vals['features']['permutation']['p_value']:.4g}, residualized p={vals['residualized']['permutation']['p_value']:.4g}.")
    lines.extend([
        "",
        f"Summary plot: `{summary_plot.relative_to(ROOT)}`",
        f"UMAP plot: `{umap_plot.relative_to(ROOT)}`",
        "",
    ])
    doc.write_text("\n".join(lines) + "\n")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    explicit = {}
    curves = {}
    for task in TASKS:
        explicit[task], curves[task] = explicit_test(task)
    out_json = RES / "explicit_chr1_chr2_test.json"
    out_json.write_text(json.dumps(explicit, indent=2) + "\n")
    curve_json = RES / "explicit_chr1_chr2_curves.json"
    curve_json.write_text(json.dumps(curves) + "\n")
    summary_plot = plot_summary(load_cv_summary(), explicit, curves)
    umap_plot = plot_umap()
    write_doc(explicit, summary_plot, umap_plot)
    print(out_json)
    print(summary_plot)
    print(umap_plot)


if __name__ == "__main__":
    main()
