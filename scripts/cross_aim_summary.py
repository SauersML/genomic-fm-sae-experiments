#!/usr/bin/env python3
"""Build cross-aim summary tables and plots from completed pilot artifacts."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

ROOT = Path("/Users/user/bio-interp-experiments")
RES = ROOT / "results" / "cross_aim"
PLOTS = ROOT / "plots"


def ensure_dirs() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def metric_row(aim: str, contrast: str, metric: str, block: dict, model: str = "l2_logreg") -> dict:
    m = block[model]
    ci = m.get(f"{metric}_ci95")
    return {
        "aim": aim,
        "contrast": contrast,
        "metric": metric,
        "model": model,
        "value": m.get(metric),
        "ci_low": ci[0] if ci else "",
        "ci_high": ci[1] if ci else "",
    }


def master_table() -> pd.DataFrame:
    rows = []
    aim1_names = [
        ("Aim1", "coding-disrupting vs not", ROOT / "results/aim1_sv/a_coding_disrupting_vs_not/results.json"),
        ("Aim1", "coding/splice vs intergenic raw", ROOT / "results/aim1_sv/c_coding_vs_intergenic_raw/results.json"),
        ("Aim1", "coding/splice vs intergenic length-matched", ROOT / "results/aim1_sv/b_coding_vs_intergenic_lenmatched/results.json"),
        ("Aim1", "cds vs intronic length-matched", ROOT / "results/aim1_sv/pair_cds_vs_intronic_lenmatched/results.json"),
        ("Aim1", "splice vs intergenic length-matched", ROOT / "results/aim1_sv/pair_splice_vs_intergenic_lenmatched/results.json"),
        ("Aim1", "cds vs intergenic length-matched", ROOT / "results/aim1_sv/pair_cds_vs_intergenic_lenmatched/results.json"),
    ]
    for aim, contrast, path in aim1_names:
        sep = load_json(path)["separation"]
        for kind in ("features", "covariates_only", "features_residualized"):
            r = metric_row(aim, contrast, "auroc", sep[kind])
            r["series"] = kind
            rows.append(r)

    for contrast, path in [
        ("sweeps vs controls", ROOT / "results/aim2_popgen/sweeps/results.json"),
        ("introgression vs controls", ROOT / "results/aim2_popgen/introgression/results.json"),
    ]:
        sep = load_json(path)["separation"]
        for kind in ("features", "covariates_only", "features_residualized"):
            r = metric_row("Aim2", contrast, "auroc", sep[kind])
            r["series"] = kind
            rows.append(r)

    aim3 = load_json(ROOT / "results/aim3_assoc/aggregate.json")["all"]["per_gene"]
    for r0 in aim3:
        gene = r0["gene"]
        ci = r0["cv_features_spearman_ci95"]
        rows.append({
            "aim": "Aim3", "contrast": gene, "metric": "spearman", "model": "ridge",
            "series": "features", "value": r0["cv_features_spearman"],
            "ci_low": ci[0], "ci_high": ci[1],
        })
        rows.append({
            "aim": "Aim3", "contrast": gene, "metric": "spearman", "model": "ridge",
            "series": "covariates_only", "value": r0["cv_altcount_only_spearman"],
            "ci_low": "", "ci_high": "",
        })
        rows.append({
            "aim": "Aim3", "contrast": gene, "metric": "spearman", "model": "ridge",
            "series": "features_residualized", "value": r0["cv_resid_altcount_spearman"],
            "ci_low": "", "ci_high": "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "master_auroc.csv", index=False)
    return df


def plot_master(df: pd.DataFrame) -> None:
    show = df[df["series"].isin(["features", "covariates_only", "features_residualized"])].copy()
    show["label"] = show["aim"] + " | " + show["contrast"]
    labels = list(dict.fromkeys(show["label"]))
    ybase = np.arange(len(labels))
    offsets = {"features": -0.22, "covariates_only": 0.0, "features_residualized": 0.22}
    colors = {"features": "#2f6f8f", "covariates_only": "#8d5a2b", "features_residualized": "#5b8f3a"}
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.36)))
    for series, sub in show.groupby("series"):
        xs, ys, xerr = [], [], []
        for _, r in sub.iterrows():
            y = labels.index(r["label"]) + offsets[series]
            xs.append(float(r["value"]))
            ys.append(y)
            if r["ci_low"] != "" and not pd.isna(r["ci_low"]):
                xerr.append([[float(r["value"]) - float(r["ci_low"])], [float(r["ci_high"]) - float(r["value"])]])
            else:
                xerr.append([[0], [0]])
        xerr_arr = np.array(xerr).reshape(len(xerr), 2).T
        ax.errorbar(xs, ys, xerr=xerr_arr, fmt="o", ms=4, capsize=2, label=series, color=colors[series])
    ax.axvline(0.5, color="#888", lw=1, ls="--", label="AUROC chance")
    ax.axvline(0.0, color="#bbb", lw=1, ls=":", label="Spearman zero")
    ax.set_yticks(ybase)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("AUROC for Aim1/Aim2; Spearman for Aim3")
    ax.set_title("Cross-aim raw feature signal versus confound controls")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_auroc_forest.png", dpi=180)
    plt.close(fig)


def aim2_frame() -> pd.DataFrame:
    ids = [l.strip() for l in (ROOT / "data/aim2_popgen/ids.txt").read_text().splitlines() if l.strip()]
    X = np.load(ROOT / "data/aim2_popgen/features.npy", mmap_mode="r")
    sweeps = pd.read_csv(ROOT / "data/aim2_popgen/table_sweeps.tsv", sep="\t")
    intro = pd.read_csv(ROOT / "data/aim2_popgen/table_introgression.tsv", sep="\t")
    cov = pd.read_csv(ROOT / "data/aim2_popgen/covariates_extra.tsv", sep="\t")
    gc = np.load(ROOT / "data/aim2_popgen/gc.npy")
    base = pd.DataFrame({"id": ids, "row": np.arange(len(ids)), "gc": gc})
    tab = pd.concat([sweeps, intro], ignore_index=True).merge(base, on="id", how="inner").merge(cov, on="id", how="left")
    if "chrom" not in tab.columns and "chrom_x" in tab.columns:
        tab["chrom"] = tab["chrom_x"]
    return tab, X


def aim2_composition_and_structure() -> dict:
    tab, X = aim2_frame()
    rows = tab["row"].to_numpy()
    Xs = np.asarray(X[rows])
    svd = TruncatedSVD(n_components=10, random_state=0)
    Z = svd.fit_transform(Xs)
    comp = tab[["gc", "repeat_frac", "mappability", "gene_density"]].to_numpy(float)
    r2_rows = []
    for j in range(6):
        r2 = LinearRegression().fit(comp, Z[:, j]).score(comp, Z[:, j])
        r2_rows.append({"component": f"svd_{j}", "composition_r2": float(r2), "explained_variance_ratio": float(svd.explained_variance_ratio_[j])})
    r2df = pd.DataFrame(r2_rows)
    r2df.to_csv(RES / "aim2_svd_composition_r2.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].bar(r2df["component"], r2df["composition_r2"], color="#6b7f2a")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("R2 from GC/repeat/mappability/gene density")
    axes[0].set_title("Aim2 SVD components explained by composition")
    colors = tab["label"].map({"sweep": "#276fbf", "introgression": "#7a3e9d", "control": "#777"}).fillna("#777")
    axes[1].scatter(tab["gc"], Z[:, 1], c=colors, s=12, alpha=0.65)
    axes[1].set_xlabel("GC")
    axes[1].set_ylabel("SVD 1")
    axes[2].scatter(tab["repeat_frac"], Z[:, 2], c=colors, s=12, alpha=0.65)
    axes[2].set_xlabel("repeat fraction")
    axes[2].set_ylabel("SVD 2")
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_svd_vs_composition.png", dpi=180)
    plt.close(fig)

    return {"svd_composition": r2_rows, "svd_variance_sum10": float(svd.explained_variance_ratio_.sum())}


def pca_umap_plots() -> None:
    aim1_labels = pd.read_parquet(ROOT / "data/aim1_sv/labels.parquet")
    X1 = np.load(ROOT / "data/aim1_sv/features.npy", mmap_mode="r")
    idx1 = np.arange(len(aim1_labels))
    X1s = np.asarray(X1[idx1])
    X1z = StandardScaler().fit_transform(TruncatedSVD(n_components=50, random_state=0).fit_transform(X1s))

    tab2, X2 = aim2_frame()
    X2s = np.asarray(X2[tab2["row"].to_numpy()])
    X2z = StandardScaler().fit_transform(TruncatedSVD(n_components=50, random_state=0).fit_transform(X2s))

    def emb(X):
        p = PCA(n_components=2, random_state=0).fit_transform(X)
        if umap is not None:
            u = umap.UMAP(n_components=2, random_state=0, n_neighbors=20, min_dist=0.15).fit_transform(X)
        else:
            u = np.full_like(p, np.nan)
        return p, u

    p1, u1 = emb(X1z)
    p2, u2 = emb(X2z)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, data, title in [(axes[0, 0], p1, "Aim1 delta PCA"), (axes[0, 1], u1, "Aim1 delta UMAP")]:
        for lab, sub in aim1_labels.groupby("consequence_coarse"):
            ii = sub.index.to_numpy()
            ax.scatter(data[ii, 0], data[ii, 1], s=12, alpha=0.7, label=lab)
        ax.set_title(title)
    for ax, data, title in [(axes[1, 0], p2, "Aim2 region PCA"), (axes[1, 1], u2, "Aim2 region UMAP")]:
        for lab, sub in tab2.groupby("label"):
            ii = sub.index.to_numpy()
            ax.scatter(data[ii, 0], data[ii, 1], s=12, alpha=0.7, label=lab)
        ax.set_title(title)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 1].legend(fontsize=7, loc="best")
    axes[1, 1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_pca_umap.png", dpi=180)
    plt.close(fig)


def sae_activation_plots() -> dict:
    X1 = np.load(ROOT / "data/aim1_sv/features.npy", mmap_mode="r")
    X2 = np.load(ROOT / "data/aim2_popgen/features.npy", mmap_mode="r")
    X3 = np.load(ROOT / "data/aim3_assoc/features.npy", mmap_mode="r")
    mats = {"Aim1": np.asarray(X1), "Aim2": np.asarray(X2), "Aim3": np.asarray(X3)}
    sparsity_rows = []
    top_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, X in mats.items():
        nnz = (np.abs(X) > 0).sum(axis=1)
        sparsity_rows.append({"aim": name, "median_nonzero": float(np.median(nnz)), "mean_nonzero": float(np.mean(nnz))})
        axes[0].hist(nnz, bins=40, alpha=0.45, label=name)
        freq = (np.abs(X) > 0).mean(axis=0)
        mean_abs = np.mean(np.abs(X), axis=0)
        top = np.argsort(mean_abs)[-25:][::-1]
        for rank, j in enumerate(top[:10], 1):
            top_rows.append({"aim": name, "rank": rank, "feature": int(j), "active_fraction": float(freq[j]), "mean_abs": float(mean_abs[j])})
        axes[1].plot(np.arange(1, 26), mean_abs[top], marker="o", ms=3, label=name)
    axes[0].set_xlabel("nonzero SAE features per row")
    axes[0].set_ylabel("rows")
    axes[0].set_title("SAE activation sparsity")
    axes[1].set_xlabel("ranked feature")
    axes[1].set_ylabel("mean absolute activation")
    axes[1].set_title("Dominant SAE features")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_sae_activation.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(sparsity_rows).to_csv(RES / "sae_sparsity.csv", index=False)
    pd.DataFrame(top_rows).to_csv(RES / "top_active_features.csv", index=False)
    overlap = {}
    by = {aim: set(pd.DataFrame(top_rows).query("aim == @aim").head(10)["feature"]) for aim in mats}
    for a in by:
        for b in by:
            if a < b:
                overlap[f"{a}_{b}_top10_overlap"] = len(by[a] & by[b])
    return {"sparsity": sparsity_rows, "top_feature_overlap": overlap}


def extras() -> dict:
    out = {}
    labels = pd.read_parquet(ROOT / "data/aim1_sv/labels.parquet")
    X1 = np.load(ROOT / "data/aim1_sv/features.npy", mmap_mode="r")
    norms = np.linalg.norm(np.asarray(X1), axis=1)
    df = labels.assign(delta_l2=norms)
    groups = [g["delta_l2"].to_numpy() for _, g in df.groupby("consequence_coarse")]
    h, p = stats.kruskal(*groups)
    coding = df["consequence_coarse"].eq("coding")
    u = stats.mannwhitneyu(df.loc[coding, "delta_l2"], df.loc[~coding, "delta_l2"], alternative="two-sided")
    out["aim1_delta_magnitude"] = {"kruskal_p": float(p), "coding_vs_other_mwu_p": float(u.pvalue)}
    fig, ax = plt.subplots(figsize=(9, 4))
    order = sorted(df["consequence_coarse"].unique())
    ax.violinplot([df.loc[df["consequence_coarse"].eq(o), "delta_l2"] for o in order], showmeans=True)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("||alt - ref SAE delta||2")
    ax.set_title("Aim1 delta magnitude by consequence")
    fig.tight_layout()
    fig.savefig(PLOTS / "aim1_delta_by_consequence.png", dpi=180)
    plt.close(fig)

    tab, X2 = aim2_frame()
    sweeps = tab[tab["task"].eq("sweeps")].copy()
    Xs = np.asarray(X2[sweeps["row"].to_numpy()])
    Z = StandardScaler().fit_transform(TruncatedSVD(n_components=128, random_state=0).fit_transform(Xs))
    y = sweeps["y"].to_numpy()
    groups = sweeps["chrom"].to_numpy()
    cov_sets = [
        ("none", []),
        ("gc", ["gc"]),
        ("gc_repeat", ["gc", "repeat_frac"]),
        ("gc_repeat_gene", ["gc", "repeat_frac", "gene_density"]),
        ("gc_repeat_gene_mappability", ["gc", "repeat_frac", "gene_density", "mappability"]),
    ]
    rows = []
    for name, cols in cov_sets:
        Xcur = Z if not cols else np.c_[Z, sweeps[cols].to_numpy(float)]
        pred = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
            Xcur, y, cv=GroupKFold(n_splits=5), groups=groups, method="predict_proba"
        )[:, 1]
        auroc = float(__import__("sklearn.metrics").metrics.roc_auc_score(y, pred))
        rows.append({"covariates_added": name, "feature_plus_covariate_auroc": auroc})
    erode = pd.DataFrame(rows)
    erode.to_csv(RES / "aim2_sweeps_covariate_path.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(erode["covariates_added"], erode["feature_plus_covariate_auroc"], marker="o", color="#2f6f8f")
    ax.set_ylim(0.5, max(0.75, erode["feature_plus_covariate_auroc"].max() + 0.03))
    ax.set_ylabel("GroupKFold AUROC")
    ax.set_title("Aim2 sweeps: adding composition covariates")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_sweeps_covariate_path.png", dpi=180)
    plt.close(fig)
    out["aim2_sweeps_covariate_path"] = rows

    aim3 = load_json(ROOT / "results/aim3_assoc/aggregate.json")["all"]["per_gene"]
    diffs = []
    for r in aim3:
        diffs.append({
            "gene": r["gene"],
            "features_minus_altcount_heldout_spearman": (
                r["heldout_test_features_spearman"] - r["heldout_test_altcount_only_spearman"]
            ),
        })
    pd.DataFrame(diffs).to_csv(RES / "aim3_heldout_vs_altcount.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    dfd = pd.DataFrame(diffs).sort_values("features_minus_altcount_heldout_spearman")
    ax.barh(dfd["gene"], dfd["features_minus_altcount_heldout_spearman"], color="#5b8f3a")
    ax.axvline(0, color="#777", lw=1)
    ax.set_xlabel("held-out Spearman: features - ALT-count")
    ax.set_title("Aim3 feature model versus ALT-count baseline")
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_aim3_vs_altcount.png", dpi=180)
    plt.close(fig)
    out["aim3_features_vs_altcount"] = diffs
    return out


def main() -> None:
    ensure_dirs()
    df = master_table()
    plot_master(df)
    summary = {}
    summary.update(aim2_composition_and_structure())
    pca_umap_plots()
    summary.update(sae_activation_plots())
    summary.update(extras())
    with (RES / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {RES} and {PLOTS}/cross_*")


if __name__ == "__main__":
    main()
