#!/usr/bin/env python
"""Aim-1 analysis driver: do Evo2 SAE ref->alt feature deltas separate SVs by
functional consequence, beyond the length/GC confound and beyond chance?

Local / CPU only. Loads:
  data/aim1_sv/features.npy   (n_items, D)   ref->alt SAE delta, row order == ids.txt
  data/aim1_sv/ids.txt        one id per line
  data/aim1_sv/labels.parquet id-aligned labels/covariates/groups (build_manifest.py)

Runs the shared harness (src/common/analysis.py::run_report) for:
  (a) coding-disrupting (cds|splice) vs not        -- primary binary target
  (b) coding vs intergenic, LENGTH-MATCHED         -- critical adversarial control
  (c) coding (cds|splice) vs intergenic, raw       -- the headline contrast, unmatched
  (+) selected pairwise one-vs-one contrasts
Each with covariates = [log_svlen, gc_window, svtype_ins], groups = chrom,
and a permutation null. Writes results/aim1_sv/<contrast>/ and a top-level
results/aim1_sv/results.json summary.

Self-test (no features.npy needed):
    .venv/bin/python src/aim1_sv/analyze.py --selftest
Real run:
    .venv/bin/python src/aim1_sv/analyze.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import analysis  # noqa: E402

DATADIR = os.path.join(ROOT, "data/aim1_sv")
RESDIR = os.path.join(ROOT, "results/aim1_sv")
# Core confounds + (if present) extra window-level covariates shipped alongside
# the manifest (data/aim1_sv/covariates_extra.tsv): repeat content, mappability,
# gene density. These are merged by id in load_aligned and added to the covariate
# set so the "beyond confound" test also controls for repeat/mappability, not just
# length/GC. Absent columns are silently skipped.
COV_COLS = ["log_svlen", "gc_window", "svtype_ins",
            "repeat_frac", "mappability", "gene_density"]
EXTRA_COV_COLS = ["repeat_frac", "mappability", "gene_density"]


# ---------------------------------------------------------------- loading ---
def load_aligned(features_path: str, ids_path: str, labels_path: str):
    """Load features + ids + labels and align strictly by id (manifest order)."""
    X = np.load(features_path)
    with open(ids_path) as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]
    if X.shape[0] != len(ids):
        raise ValueError(f"features rows {X.shape[0]} != ids {len(ids)}")
    lab = pd.read_parquet(labels_path).set_index("id")
    missing = [i for i in ids if i not in lab.index]
    if missing:
        raise ValueError(f"{len(missing)} ids in features not in labels, e.g. {missing[:3]}")
    lab = lab.loc[ids].reset_index()  # align to feature row order

    # Optional: merge per-id GC% if azure-compute shipped it (gc_window.{parquet,csv}
    # keyed by id with a 'gc_window' or 'gc' column). Overwrites the NaN placeholder.
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        gpath = os.path.join(os.path.dirname(features_path), "gc_window" + ext)
        if os.path.exists(gpath):
            g = reader(gpath)
            gcol = "gc_window" if "gc_window" in g.columns else ("gc" if "gc" in g.columns else None)
            if "id" in g.columns and gcol:
                gmap = dict(zip(g["id"], g[gcol]))
                lab["gc_window"] = lab["id"].map(gmap).astype(float)
                print(f"[analyze] merged gc_window from {gpath} "
                      f"({lab['gc_window'].notna().sum()}/{len(lab)} filled)")
                break

    # Optional: merge extra window covariates (repeat_frac, mappability,
    # gene_density) keyed by id, if shipped with the manifest.
    epath = os.path.join(os.path.dirname(labels_path), "covariates_extra.tsv")
    if os.path.exists(epath):
        e = pd.read_csv(epath, sep="\t")
        have = [c for c in EXTRA_COV_COLS if c in e.columns]
        if "id" in e.columns and have:
            e = e[["id"] + have].drop_duplicates("id").set_index("id")
            for c in have:
                lab[c] = lab["id"].map(e[c]).astype(float)
            print(f"[analyze] merged extra covariates {have} from {epath}")
    return X, lab


def _covariates(sub: pd.DataFrame):
    """Build covariate matrix; drop gc_window if azure never filled it (all NaN)."""
    cols = [c for c in COV_COLS if c in sub.columns]
    cov = sub[cols].copy()
    # drop fully-NaN covariates (e.g. gc_window not provided), warn
    for c in list(cov.columns):
        if cov[c].isna().all():
            print(f"[analyze] covariate '{c}' is all-NaN (not provided) -> dropped")
            cov = cov.drop(columns=[c])
    # impute any residual NaN with column median (defensive)
    if cov.isna().any().any():
        cov = cov.fillna(cov.median(numeric_only=True))
    if cov.shape[1] == 0:
        return None, []
    return cov.to_numpy(dtype=float), list(cov.columns)


# ------------------------------------------------------- length matching ---
def length_match(sub: pd.DataFrame, label_col: str, pos_val, neg_val,
                 seed: int = 0, n_bins: int = 12):
    """Subsample two groups to match the log_svlen distribution. Within each
    quantile bin of log_svlen (bins from the pooled distribution), keep equal
    numbers of pos and neg by downsampling the larger side. Returns the matched
    sub-DataFrame (index preserved)."""
    rng = np.random.default_rng(seed)
    d = sub[sub[label_col].isin([pos_val, neg_val])].copy()
    edges = np.quantile(d["log_svlen"], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    d["_bin"] = pd.cut(d["log_svlen"], bins=np.unique(edges), include_lowest=True)
    keep_idx = []
    for _, g in d.groupby("_bin", observed=True):
        pos = g[g[label_col] == pos_val]
        neg = g[g[label_col] == neg_val]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep_idx += list(pos.sample(k, random_state=int(rng.integers(1 << 31))).index)
        keep_idx += list(neg.sample(k, random_state=int(rng.integers(1 << 31))).index)
    return sub.loc[keep_idx].copy()


# ---------------------------------------------------------- one contrast ---
def _svd_reduce(Xs: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Unsupervised TruncatedSVD to k components for the heavy CV path.

    NOTE (leakage caveat, documented): the SVD basis is fit on all rows of the
    contrast subset (uses test-fold X, but NOT y) before CV. This is the standard
    unsupervised-preprocessing tradeoff; it cannot leak label information and only
    mildly optimistically biases variance estimates. It bounds compute on the wide
    (32768-d) delta. Per-feature univariate stats (differential_features) are still
    computed on the FULL-width delta, so feature-level findings are not reduced.
    """
    from sklearn.decomposition import TruncatedSVD
    k = int(min(k, Xs.shape[1] - 1, max(2, Xs.shape[0] - 1)))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    return svd.fit_transform(Xs)


def run_contrast(X: np.ndarray, lab: pd.DataFrame, mask: np.ndarray,
                 y: np.ndarray, name: str, seed: int, n_perm: int,
                 n_splits: int, n_boot: int, svd_k: int = 0) -> dict:
    """Run run_report on the masked subset; return a compact summary.

    If svd_k>0 and the feature matrix is wider than svd_k, the heavy CV/permutation
    path runs on an unsupervised SVD reduction (see _svd_reduce); differential
    features are still computed on the full-width delta.
    """
    sub = lab.loc[mask].reset_index(drop=True)
    Xs_full = X[mask.values if hasattr(mask, "values") else mask]
    ys = np.asarray(y)
    cov, cov_names = _covariates(sub)
    groups = sub["chrom"].to_numpy()
    outdir = os.path.join(RESDIR, name)

    Xs = Xs_full
    reduced = False
    if svd_k and Xs_full.shape[1] > svd_k:
        Xs = _svd_reduce(Xs_full, svd_k, seed)
        reduced = True
    print(f"\n=== contrast: {name}  n={len(ys)}  pos={int(ys.sum())} "
          f"groups={sub['chrom'].nunique()} cov={cov_names} "
          f"X={Xs.shape}{' (SVD)' if reduced else ''} ===")
    res = analysis.run_report(
        Xs, ys, groups=groups, covariates=cov, outdir=outdir,
        title=f"aim1_sv :: {name}", task="classification", seed=seed,
        n_perm=n_perm, n_splits=n_splits, n_boot=n_boot,
    )
    if reduced:
        # full-width per-feature differential stats (univariate, cheap) overwrite
        # the reduced-component table so feature-level findings stay interpretable.
        diff_full = analysis.differential_features(Xs_full, ys, task="classification")
        diff_full.to_csv(os.path.join(outdir, "differential_features.csv"), index=False)
        res["n_significant_features_fdr05"] = int((diff_full["p_adj_bh"] < 0.05).sum())
    # compact summary
    sep = res["separation"]

    def best(block):
        if block not in sep:
            return None
        out = {}
        for model, m in sep[block].items():
            out[model] = {"auroc": m.get("auroc"), "auroc_ci95": m.get("auroc_ci95"),
                          "auprc": m.get("auprc"), "auprc_ci95": m.get("auprc_ci95")}
        return out

    return {
        "name": name, "n": int(len(ys)), "n_pos": int(ys.sum()),
        "n_groups": int(sub["chrom"].nunique()), "covariates": cov_names,
        "features": best("features"),
        "covariates_only": best("covariates_only"),
        "features_residualized": best("features_residualized"),
        "permutation_p": res["permutation_test"]["p_value"],
        "permutation_observed": res["permutation_test"]["observed"],
        "permutation_null_mean": res["permutation_test"]["null_mean"],
        "n_sig_features_fdr05": res["n_significant_features_fdr05"],
        "outdir": outdir,
    }


def analyze(features_path, ids_path, labels_path, seed, n_perm, n_splits, n_boot,
            svd_k=0):
    os.makedirs(RESDIR, exist_ok=True)
    X, lab = load_aligned(features_path, ids_path, labels_path)
    print(f"[analyze] X={X.shape}  labels={lab.shape}  svd_k={svd_k or 'off'}")
    summaries = []

    # (a) coding-disrupting (cds|splice) vs not -- primary binary target
    mask = pd.Series(True, index=lab.index)
    summaries.append(run_contrast(
        X, lab, mask, lab["y_binary"].to_numpy(),
        "a_coding_disrupting_vs_not", seed, n_perm, n_splits, n_boot, svd_k))

    # (c) coding (cds|splice) vs intergenic -- raw, unmatched headline contrast
    is_coding = lab["consequence"].isin(["cds", "splice"])
    is_inter = lab["consequence"] == "intergenic"
    m_ci = is_coding | is_inter
    summaries.append(run_contrast(
        X, lab, m_ci, is_coding[m_ci].astype(int).to_numpy(),
        "c_coding_vs_intergenic_raw", seed, n_perm, n_splits, n_boot, svd_k))

    # (b) coding vs intergenic, LENGTH-MATCHED -- critical adversarial control
    lab2 = lab.copy()
    lab2["_grp"] = np.where(lab2["consequence"].isin(["cds", "splice"]), "coding",
                            np.where(lab2["consequence"] == "intergenic", "intergenic", "other"))
    matched = length_match(lab2[lab2["_grp"] != "other"], "_grp", "coding", "intergenic", seed=seed)
    if len(matched) >= 20:
        mm = lab.index.isin(matched.index)
        mm = pd.Series(mm, index=lab.index)
        y_m = (lab.loc[mm, "consequence"].isin(["cds", "splice"])).astype(int).to_numpy()
        s = run_contrast(X, lab, mm, y_m,
                         "b_coding_vs_intergenic_lenmatched", seed, n_perm, n_splits, n_boot, svd_k)
        # record matching quality
        md = lab.loc[mm]
        s["lenmatch_median_log_svlen"] = {
            "coding": float(md.loc[md["consequence"].isin(["cds", "splice"]), "log_svlen"].median()),
            "intergenic": float(md.loc[md["consequence"] == "intergenic", "log_svlen"].median()),
        }
        summaries.append(s)
    else:
        print("[analyze] WARNING: length-matched subset too small, skipped")

    # (+) selected pairwise one-vs-one contrasts (each length-matched too)
    pairs = [("cds", "intronic"), ("splice", "intergenic"), ("cds", "intergenic")]
    for a, b in pairs:
        m = lab["consequence"].isin([a, b])
        if m.sum() < 20:
            continue
        sub = lab.loc[m].copy()
        matched = length_match(sub, "consequence", a, b, seed=seed)
        if len(matched) < 20:
            continue
        mm = pd.Series(lab.index.isin(matched.index), index=lab.index)
        y_p = (lab.loc[mm, "consequence"] == a).astype(int).to_numpy()
        summaries.append(run_contrast(
            X, lab, mm, y_p, f"pair_{a}_vs_{b}_lenmatched",
            seed, n_perm, n_splits, n_boot, svd_k))

    out = {"seed": seed, "n_items": int(X.shape[0]), "n_features": int(X.shape[1]),
           "contrasts": summaries}
    with open(os.path.join(RESDIR, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[analyze] wrote {os.path.join(RESDIR, 'results.json')} "
          f"({len(summaries)} contrasts)")
    return out


# ----------------------------------------------------------------- self-test
def selftest():
    """End-to-end on synthetic features with planted signal: a few feature dims
    carry the coding-vs-not label; length is correlated with the label (confound)
    but the planted signal is independent of length, so it should survive
    residualization. Writes to a temp dir, asserts the pipeline runs and that
    the planted contrast beats chance."""
    import tempfile
    rng = np.random.default_rng(0)
    D = 2000   # wide-ish to exercise the SVD reduction path realistically
    lab = pd.read_parquet(os.path.join(DATADIR, "labels.parquet")).copy()
    # use real ids/labels/covariates for realism; plant signal into X
    n = len(lab)
    X = rng.normal(size=(n, D)).astype(np.float32)
    y = lab["y_binary"].to_numpy()
    # plant: shift first 5 features by class (signal independent of length)
    X[y == 1, :5] += 1.3
    # also make a few features track log_svlen (a confound channel) so the
    # length-matched control has something to (correctly) discount
    z = (lab["log_svlen"] - lab["log_svlen"].mean()) / (lab["log_svlen"].std() + 1e-9)
    X[:, 5:8] += 0.8 * z.to_numpy()[:, None]

    tmp = tempfile.mkdtemp()
    fpath = os.path.join(tmp, "features.npy")
    ipath = os.path.join(tmp, "ids.txt")
    np.save(fpath, X)
    with open(ipath, "w") as fh:
        fh.write("\n".join(lab["id"].tolist()) + "\n")
    # cheap settings for the self-test
    global RESDIR
    saved = RESDIR
    RESDIR = os.path.join(tmp, "results")
    try:
        out = analyze(fpath, ipath, os.path.join(DATADIR, "labels.parquet"),
                      seed=0, n_perm=100, n_splits=5, n_boot=100, svd_k=64)
    finally:
        RESDIR = saved
    # assertions
    a = next(c for c in out["contrasts"] if c["name"] == "a_coding_disrupting_vs_not")
    auroc = a["features"]["logreg"]["auroc"] if "logreg" in a["features"] else \
        list(a["features"].values())[0]["auroc"]
    assert auroc > 0.7, f"planted signal not recovered: AUROC={auroc}"
    assert a["permutation_p"] < 0.05, f"perm p not significant: {a['permutation_p']}"
    # residualized should still beat chance (signal independent of length)
    rblock = a.get("features_residualized")
    assert rblock is not None, "no residualized block (covariates missing?)"
    print(f"\n[SELFTEST] contrast (a) features AUROC={auroc:.3f} "
          f"perm_p={a['permutation_p']:.4f}  resid={ {k: v['auroc'] for k,v in rblock.items()} }")
    print("[SELFTEST] PASSED — pipeline runs end-to-end and recovers planted signal.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(DATADIR, "features.npy"))
    ap.add_argument("--ids", default=os.path.join(DATADIR, "ids.txt"))
    ap.add_argument("--labels", default=os.path.join(DATADIR, "labels.parquet"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--svd-k", type=int, default=128,
                    help="reduce wide delta to K comps for the heavy CV path "
                         "(0=off; differential_features stays full-width). "
                         "Recommended for the 32768-d delta to bound CPU compute.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        analyze(args.features, args.ids, args.labels,
                args.seed, args.n_perm, args.n_splits, args.n_boot, args.svd_k)
