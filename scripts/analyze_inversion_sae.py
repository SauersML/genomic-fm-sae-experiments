#!/usr/bin/env python3
"""SAE-only inversion analysis.

Inputs:
  data/inversions/features.npy + ids.txt + labels.parquet
  data/aim1_sv/features.npy + ids.txt + labels.parquet

Outputs:
  results/inversions/*.tsv|json
  plots/inversion_sae_specificity.png
  docs/RESULTS_INVERSION_SAE.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "data/inversions"
AIM1 = ROOT / "data/aim1_sv"
OUT = ROOT / "results/inversions"
PLOTS = ROOT / "plots"
DOCS = ROOT / "docs"
F15532 = 15532


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def welch_table(x: np.ndarray, y: np.ndarray, prefix: str) -> pd.DataFrame:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x.mean(axis=0) - y.mean(axis=0)
    vx = x.var(axis=0, ddof=1)
    vy = y.var(axis=0, ddof=1)
    nx, ny = x.shape[0], y.shape[0]
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    cohen = np.divide(diff, pooled, out=np.zeros_like(diff), where=pooled > 0)
    t, p = stats.ttest_ind(x, y, axis=0, equal_var=False, nan_policy="omit")
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    q = bh(p)
    return pd.DataFrame({
        "feature": [f"f{i}" for i in range(x.shape[1])],
        "feature_idx": np.arange(x.shape[1]),
        f"{prefix}_mean_diff_raw": diff,
        f"{prefix}_cohen_d": cohen,
        "p": p,
        "q": q,
    }).sort_values(["q", f"{prefix}_cohen_d"], ascending=[True, False])


def corr_table(x: np.ndarray, y: np.ndarray, name: str) -> pd.DataFrame:
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    rows = []
    for j in range(x.shape[1]):
        if ok.sum() < 20 or np.std(x[ok, j]) == 0:
            r, p = np.nan, 1.0
        else:
            r, p = stats.spearmanr(x[ok, j], y[ok])
        rows.append((f"f{j}", j, r, p))
    out = pd.DataFrame(rows, columns=["feature", "feature_idx", f"spearman_{name}", "p"])
    out["q"] = bh(out["p"].fillna(1).to_numpy())
    return out.sort_values(["q", f"spearman_{name}"], ascending=[True, False])


def cv_metric(x: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str) -> dict:
    uniq = np.unique(groups)
    n_splits = min(5, len(uniq))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return {"task": task, "metric": None}
    pred = np.full(len(y), np.nan)
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(x, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        if set(np.unique(y)).issubset({0, 1}):
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(64, x.shape[0] - 1, x.shape[1]), random_state=1),
                LogisticRegression(max_iter=2000, C=0.25, class_weight="balanced"),
            )
            model.fit(x[tr], y[tr])
            pred[te] = model.predict_proba(x[te])[:, 1]
        else:
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(64, x.shape[0] - 1, x.shape[1]), random_state=1),
                Ridge(alpha=10.0),
            )
            model.fit(x[tr], y[tr])
            pred[te] = model.predict(x[te])
    ok = np.isfinite(pred)
    if set(np.unique(y)).issubset({0, 1}):
        score = roc_auc_score(y[ok], pred[ok]) if len(np.unique(y[ok])) == 2 else np.nan
        metric = "AUROC"
    else:
        score = r2_score(y[ok], pred[ok])
        metric = "R2"
    return {"task": task, "metric": metric, "score": float(score), "n_scored": int(ok.sum()), "n_splits": int(n_splits)}


def load_feature_set(base: Path, labels_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    feats = np.load(base / "features.npy")
    ids = read_ids(base / "ids.txt")
    labels = pd.read_parquet(labels_path)
    labels = labels.set_index("id").loc[ids].reset_index()
    return feats, labels


def plot_all(inv_x: np.ndarray, inv_lab: pd.DataFrame, sv_x: np.ndarray, sv_lab: pd.DataFrame,
             inv_vs_sv: pd.DataFrame, coding: pd.DataFrame) -> None:
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })
    combined = np.vstack([inv_x, sv_x])
    z = StandardScaler().fit_transform(combined)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=35, min_dist=0.25, metric="cosine", random_state=3).fit_transform(z)
        emb_name = "UMAP"
    except Exception:
        emb = PCA(n_components=2, random_state=3).fit_transform(z)
        emb_name = "PCA"

    kind = np.array(["INV"] * len(inv_x) + sv_lab["svtype"].astype(str).to_list())
    colors = {"INV": "#8E24AA", "DEL": "#0B6E99", "INS": "#F26A4F"}
    fig = plt.figure(figsize=(17, 11), dpi=190)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.02, 1.05], hspace=0.36, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    for k in ["DEL", "INS", "INV"]:
        mask = kind == k
        ax.scatter(emb[mask, 0], emb[mask, 1], s=18 if k == "INV" else 12,
                   c=colors[k], alpha=0.74 if k == "INV" else 0.38,
                   linewidths=0, label=f"{k} n={mask.sum()}")
    ax.set_title(f"{emb_name}: inversion deltas vs indel deltas", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel(f"{emb_name} 1")
    ax.set_ylabel(f"{emb_name} 2")
    ax.legend(frameon=False, loc="best", markerscale=1.5)

    ax = fig.add_subplot(gs[0, 1])
    v = inv_vs_sv.copy()
    xcol = "inv_vs_indel_cohen_d"
    sig = v["q"] < 0.05
    y = -np.log10(v["q"].clip(lower=1e-300))
    ax.scatter(v.loc[~sig, xcol], y[~sig], s=7, c="#B7C1C8", alpha=0.16, linewidths=0)
    pos = sig & (v[xcol] > 0)
    neg = sig & (v[xcol] < 0)
    ax.scatter(v.loc[pos, xcol], y[pos], s=18, c="#8E24AA", alpha=0.72, linewidths=0)
    ax.scatter(v.loc[neg, xcol], y[neg], s=18, c="#0B6E99", alpha=0.72, linewidths=0)
    ax.axhline(-np.log10(0.05), color="#8A969E", lw=1)
    ax.axvline(0, color="#8A969E", lw=1)
    ax.set_title("SAE features specific to inversion alleles", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("standardized mean shift: inversion - indel")
    ax.set_ylabel("-log10 BH q")

    ax = fig.add_subplot(gs[1, 0])
    top = inv_vs_sv.head(12).iloc[::-1]
    vals = top[xcol].to_numpy()
    bar_colors = np.where(vals >= 0, "#8E24AA", "#0B6E99")
    ax.barh(top["feature"], vals, color=bar_colors, height=0.72)
    lim = max(0.85, float(np.nanmax(np.abs(vals))) * 1.12)
    ax.set_xlim(-lim, lim * 1.34)
    q_x = lim * 1.05
    ax.text(q_x, len(top) - 0.05, "BH q", ha="left", va="bottom",
            fontsize=8.5, color="#606A73", weight="bold")
    for yi, (_, r) in enumerate(top.iterrows()):
        ax.text(q_x, yi, f"{r['q']:.1e}", va="center", ha="left",
                fontsize=8.2, color="#424A52")
    ax.axvline(0, color="#8A969E", lw=1)
    ax.set_title("Top inversion-shifted SAE dimensions", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("standardized mean shift")

    ax = fig.add_subplot(gs[1, 1])
    f = pd.DataFrame({
        "value": np.r_[inv_x[:, F15532], sv_x[:, F15532]],
        "kind": kind,
        "coding": np.r_[inv_lab["y_binary"].to_numpy(), sv_lab["y_binary"].to_numpy()],
    })
    positions = {"DEL": 0, "INS": 1, "INV": 2}
    rng = np.random.default_rng(4)
    for k in ["DEL", "INS", "INV"]:
        sub = f[f["kind"] == k]
        xs = positions[k] + rng.normal(0, 0.055, len(sub))
        cs = np.where(sub["coding"].to_numpy() == 1, "#C2185B", colors[k])
        ax.scatter(xs, sub["value"], s=16, c=cs, alpha=0.42, linewidths=0)
        ax.hlines(sub["value"].median(), positions[k] - 0.26, positions[k] + 0.26, color="#111111", lw=2)
    ax.set_xticks([0, 1, 2], ["DEL", "INS", "INV"])
    ax.set_title("f15532 response across SV classes", loc="left", fontsize=14, weight="bold")
    ax.set_ylabel("raw SAE delta")

    fig.savefig(PLOTS / "inversion_sae_specificity.png", bbox_inches="tight")
    plt.close(fig)


def plot_associations(inv_x: np.ndarray, inv_lab: pd.DataFrame, coding: pd.DataFrame,
                      length: pd.DataFrame, af: pd.DataFrame) -> None:
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })

    fig = plt.figure(figsize=(16, 10), dpi=190)
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.30)

    def volcano(ax, df, xcol, title, pos_color="#8E24AA", neg_color="#0B6E99"):
        sig = df["q"] < 0.05
        y = -np.log10(df["q"].clip(lower=1e-300))
        ax.scatter(df.loc[~sig, xcol], y[~sig], s=7, c="#B7C1C8", alpha=0.16, linewidths=0)
        pos = sig & (df[xcol] > 0)
        neg = sig & (df[xcol] < 0)
        ax.scatter(df.loc[pos, xcol], y[pos], s=17, c=pos_color, alpha=0.70, linewidths=0)
        ax.scatter(df.loc[neg, xcol], y[neg], s=17, c=neg_color, alpha=0.70, linewidths=0)
        ax.axhline(-np.log10(0.05), color="#8A969E", lw=1)
        ax.axvline(0, color="#8A969E", lw=1)
        ax.set_title(title, loc="left", fontsize=14, weight="bold")
        ax.set_ylabel("-log10 BH q")

    ax = fig.add_subplot(gs[0, 0])
    volcano(ax, coding, "inv_coding_vs_other_cohen_d", "Coding-disrupting inversion features")
    ax.set_xlabel("standardized shift: coding - other")

    ax = fig.add_subplot(gs[0, 1])
    volcano(ax, length, "spearman_log_inv_len", "SAE features associated with inversion length")
    ax.set_xlabel("Spearman rho with log10(length)")

    ax = fig.add_subplot(gs[1, 0])
    volcano(ax, af, "spearman_af", "SAE features associated with allele frequency")
    ax.set_xlabel("Spearman rho with AF")

    ax = fig.add_subplot(gs[1, 1])
    fidx = int(length.iloc[0]["feature_idx"])
    colors = {"ins": "#8E24AA", "del": "#0B6E99", "mnp": "#F26A4F"}
    x = np.log10(inv_lab["inv_len"].clip(lower=1).to_numpy())
    y = inv_x[:, fidx]
    for typ, sub in inv_lab.groupby("type_allele"):
        idx = sub.index.to_numpy()
        ax.scatter(x[idx], y[idx], s=18, c=colors.get(str(typ), "#606A73"),
                   alpha=0.48, linewidths=0, label=f"{typ} n={len(idx)}")
    ax.set_title(f"{length.iloc[0]['feature']} tracks inversion length", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("log10 inversion length")
    ax.set_ylabel("raw SAE delta")
    ax.legend(frameon=False, loc="best")

    fig.savefig(PLOTS / "inversion_sae_associations.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    inv_x, inv_lab = load_feature_set(INV, INV / "labels.parquet")
    sv_x, sv_lab = load_feature_set(AIM1, AIM1 / "labels.parquet")
    sv_lab = sv_lab.copy()
    sv_lab["svtype"] = sv_lab["svtype"].astype(str)

    inv_vs_sv = welch_table(inv_x, sv_x, "inv_vs_indel")
    inv_vs_sv.to_csv(OUT / "inv_vs_indel_features.tsv", sep="\t", index=False)

    coding = welch_table(
        inv_x[inv_lab["y_binary"].to_numpy() == 1],
        inv_x[inv_lab["y_binary"].to_numpy() == 0],
        "inv_coding_vs_other",
    )
    coding.to_csv(OUT / "inv_coding_features.tsv", sep="\t", index=False)

    length = corr_table(inv_x, inv_lab["log_inv_len"].to_numpy(), "log_inv_len")
    length.to_csv(OUT / "inv_length_features.tsv", sep="\t", index=False)
    af = corr_table(inv_x, inv_lab["af"].to_numpy(), "af")
    af.to_csv(OUT / "inv_af_features.tsv", sep="\t", index=False)

    metrics = [
        cv_metric(np.vstack([inv_x, sv_x]), np.r_[np.ones(len(inv_x)), np.zeros(len(sv_x))],
                  np.r_[inv_lab["chrom"].to_numpy(), sv_lab["chrom"].to_numpy()], "inversion_vs_indel"),
        cv_metric(inv_x, inv_lab["y_binary"].to_numpy(), inv_lab["chrom"].to_numpy(), "inversion_coding_disrupting"),
        cv_metric(inv_x, np.log10(inv_lab["inv_len"].clip(lower=1).to_numpy()),
                  inv_lab["chrom"].to_numpy(), "inversion_log_length"),
    ]

    f15532 = {
        "inv_mean": float(inv_x[:, F15532].mean()),
        "inv_median": float(np.median(inv_x[:, F15532])),
        "del_mean": float(sv_x[sv_lab["svtype"].to_numpy() == "DEL", F15532].mean()),
        "ins_mean": float(sv_x[sv_lab["svtype"].to_numpy() == "INS", F15532].mean()),
        "inv_vs_indel_feature_row": inv_vs_sv[inv_vs_sv["feature_idx"] == F15532].iloc[0].to_dict(),
    }
    summary = {
        "n_inversion_alleles": int(len(inv_x)),
        "n_indel_aim1": int(len(sv_x)),
        "inversion_consequence_counts": inv_lab["consequence"].value_counts().to_dict(),
        "inversion_type_allele_counts": inv_lab["type_allele"].value_counts().head(20).to_dict(),
        "metrics": metrics,
        "n_inv_vs_indel_bh05": int((inv_vs_sv["q"] < 0.05).sum()),
        "n_inv_coding_bh05": int((coding["q"] < 0.05).sum()),
        "n_inv_length_bh05": int((length["q"] < 0.05).sum()),
        "n_inv_af_bh05": int((af["q"] < 0.05).sum()),
        "f15532": f15532,
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    plot_all(inv_x, inv_lab, sv_x, sv_lab, inv_vs_sv, coding)
    plot_associations(inv_x, inv_lab, coding, length, af)

    top_inv = inv_vs_sv.head(8)[["feature", "inv_vs_indel_cohen_d", "inv_vs_indel_mean_diff_raw", "q"]]
    top_code = coding.head(8)[["feature", "inv_coding_vs_other_cohen_d", "inv_coding_vs_other_mean_diff_raw", "q"]]
    top_len = length.head(8)[["feature", "spearman_log_inv_len", "q"]]
    lines = [
        "# Inversion SAE Analysis",
        "",
        "Scope: HPRC release2 `INV`-flagged alleles only, analyzed with Evo2 layer-26 Goodfire SAE deltas. No introgression or expression outcome is used here.",
        "",
        f"- Inversion alleles embedded: {len(inv_x)}",
        f"- Reference indel deltas compared from Aim1 pilot: {len(sv_x)}",
        f"- INV-vs-indel SAE dimensions at BH q<0.05: {summary['n_inv_vs_indel_bh05']}",
        f"- Within-inversion coding-vs-other SAE dimensions at BH q<0.05: {summary['n_inv_coding_bh05']}",
        f"- Inversion-length-associated SAE dimensions at BH q<0.05: {summary['n_inv_length_bh05']}",
        f"- AF-associated SAE dimensions at BH q<0.05: {summary['n_inv_af_bh05']}",
        "",
        "## Predictive SAE Checks",
    ]
    for m in metrics:
        if m.get("metric") is None:
            lines.append(f"- {m['task']}: not estimable")
        else:
            lines.append(f"- {m['task']}: {m['metric']}={m['score']:.3f} over {m['n_scored']} held-out rows by chromosome")
    lines += [
        "",
        "## f15532",
        f"- INV mean raw delta: {f15532['inv_mean']:.6g}; DEL mean: {f15532['del_mean']:.6g}; INS mean: {f15532['ins_mean']:.6g}.",
        f"- INV-vs-indel f15532 standardized shift: {f15532['inv_vs_indel_feature_row']['inv_vs_indel_cohen_d']:.3f}, q={f15532['inv_vs_indel_feature_row']['q']:.3g}.",
        "",
        "## Top INV-vs-Indel SAE Features",
        top_inv.to_markdown(index=False),
        "",
        "## Top Coding-vs-Other Features Within Inversions",
        top_code.to_markdown(index=False),
        "",
        "## Top Inversion-Length Features",
        top_len.to_markdown(index=False),
        "",
        "Plots: `plots/inversion_sae_specificity.png`, `plots/inversion_sae_associations.png`",
    ]
    (DOCS / "RESULTS_INVERSION_SAE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2)[:4000])
    print("wrote", PLOTS / "inversion_sae_specificity.png")


if __name__ == "__main__":
    main()
