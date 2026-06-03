#!/usr/bin/env python3
"""Aim 2 analysis driver: does Evo2-SAE feature content of popgen-flagged
regions separate (A) selective sweeps and (B) archaic-introgression segments
from matched controls, beyond a length/GC/repeat/mappability/gene-density
confound baseline?

For each task it loads the aligned table + the azure-compute feature matrix and
calls ``src/common/analysis.py::run_report`` with:
  - X         = pooled Evo2 layer-26 SAE feature vector per region
  - y         = 1 positive / 0 control
  - groups    = chrom  (held out by whole chromosome -> GroupKFold, no leakage)
  - covariates= [log_length, GC, repeat_frac, mappability, gene_density]
  - task      = classification ; permutation null + differential features on.

Run modes:
  python -m src.aim2_popgen.analyze --selftest     # synthetic planted-signal test
  python -m src.aim2_popgen.analyze                 # real run (needs features.npy)

Real-run inputs (data/aim2_popgen/):
  features.npy  (n_rows, d) , ids.txt (row order) , gc.npy (n_rows,)
  table_sweeps.tsv , table_introgression.tsv
  covariates_extra.tsv (required for repeat/mappability/gene-density control)
Outputs:
  results/aim2_popgen/<task>/{results.json,report.md,differential_features.csv,*.png}
  results/aim2_popgen/summary.json
  docs/RESULTS_AIM2.md
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.common.analysis import run_report  # noqa: E402

DATADIR = os.path.join(ROOT, "data/aim2_popgen")
RESDIR = os.path.join(ROOT, "results/aim2_popgen")
TASKS = ("sweeps", "introgression")
REQUIRED_EXTRA_COV_COLS = ("repeat_frac", "mappability", "gene_density")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_table(task):
    rows = []
    path = os.path.join(DATADIR, f"table_{task}.tsv")
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(r)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def load_features(datadir):
    """Returns (X, ids, gc). Aligns by ids.txt row order."""
    fpath = os.path.join(datadir, "features.npy")
    ipath = os.path.join(datadir, "ids.txt")
    gpath = os.path.join(datadir, "gc.npy")
    missing = [p for p in (fpath, ipath, gpath) if not os.path.exists(p)]
    if missing:
        rel = [os.path.relpath(p, ROOT) for p in missing]
        raise FileNotFoundError(
            "Aim 2 real run requires GPU-produced features, ids, and GC: "
            + ", ".join(rel)
        )
    X = np.load(fpath)
    if X.ndim != 2:
        raise ValueError(f"features.npy must be 2-D, got shape {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"features.npy must be non-empty, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("features.npy contains NaN or infinite values")
    with open(os.path.join(datadir, "ids.txt")) as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError("ids.txt contains duplicate ids; feature alignment is ambiguous")
    gc = np.load(gpath)
    if gc.ndim != 1:
        raise ValueError(f"gc.npy must be 1-D, got shape {gc.shape}")
    if len(ids) != X.shape[0]:
        raise ValueError(f"ids.txt ({len(ids)}) != features.npy rows ({X.shape[0]})")
    if gc.shape[0] != X.shape[0]:
        raise ValueError(f"gc.npy rows ({gc.shape[0]}) != features.npy rows ({X.shape[0]})")
    if not np.isfinite(gc).all():
        raise ValueError("gc.npy contains NaN or infinite values")
    return X, ids, gc


def load_extra_covariates(datadir):
    """id -> extra covariates from covariates_extra.tsv.

    These are required for the real Aim 2 claim: repeat content, mappability,
    and gene density are obvious local confounds for population-genetic hits.
    """
    p = os.path.join(datadir, "covariates_extra.tsv")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "Aim 2 real run requires data/aim2_popgen/covariates_extra.tsv"
        )
    out = {}
    with open(p) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            missing = [c for c in REQUIRED_EXTRA_COV_COLS if r.get(c) in (None, "")]
            if missing:
                raise ValueError(
                    f"covariates_extra.tsv row {r.get('id', '<missing id>')} "
                    f"missing required columns: {missing}"
                )
            try:
                out[r["id"]] = {c: float(r[c]) for c in REQUIRED_EXTRA_COV_COLS}
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric covariates_extra.tsv values for id {r.get('id')}"
                ) from exc
    if not out:
        raise ValueError("covariates_extra.tsv has no usable data rows")
    return out


def align(task, X, ids, gc, extra=None):
    """Join feature rows (keyed by ids, possibly incl. __w32) to the table by id.

    Uses only the 8 kb rows (id == region_id, no __w32 suffix). Returns aligned
    X, y, groups(chrom), covariates, plus per-row meta for reporting.

    Covariates = [log_length, gc, repeat_frac, mappability, gene_density].
    """
    id2row = {i: k for k, i in enumerate(ids)}
    rows = load_table(task)
    keep_X, y, groups, log_len, gc_vec, meta = [], [], [], [], [], []
    extra_cols = list(REQUIRED_EXTRA_COV_COLS)
    extra_vec = {c: [] for c in extra_cols}
    n_missing = 0
    missing_extra = []
    for r in rows:
        fid = r["id"]  # 8 kb id == region_id
        if fid not in id2row:
            n_missing += 1
            continue
        if fid not in extra:
            missing_extra.append(fid)
            continue
        k = id2row[fid]
        keep_X.append(X[k])
        y.append(int(r["y"]))
        groups.append(r["chrom"])
        log_len.append(float(r["log_length"]))
        gc_vec.append(float(gc[k]))
        for c in extra_cols:
            extra_vec[c].append(extra[fid][c])
        meta.append(r)
    if missing_extra:
        examples = ", ".join(missing_extra[:5])
        raise ValueError(
            f"{task}: {len(missing_extra)} aligned rows missing extra covariates "
            f"(examples: {examples})"
        )
    Xa = np.asarray(keep_X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if Xa.ndim != 2 or Xa.shape[0] == 0:
        raise ValueError(f"{task}: no feature rows aligned to table_{task}.tsv")
    if np.unique(y).size != 2:
        raise ValueError(f"{task}: expected binary labels after alignment, got {np.unique(y)}")
    if np.unique(groups).size < 2:
        raise ValueError(f"{task}: need at least two chromosomes/groups after alignment")
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"{task}: both classes are required after alignment, got {counts.tolist()}")
    cov_cols = [np.asarray(log_len)]
    cov_names = ["log_length"]
    cov_cols.append(np.asarray(gc_vec))
    cov_names.append("gc")
    for c in extra_cols:
        cov_cols.append(np.asarray(extra_vec[c]))
        cov_names.append(c)
    cov = np.column_stack(cov_cols)
    if not np.isfinite(cov).all():
        raise ValueError(f"{task}: covariate matrix contains NaN or infinite values")
    return Xa, y, groups, cov, cov_names, meta, n_missing


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
MODEL = "l2_logreg"   # primary baseline model key in the harness


def _pick(block):
    """Return the metric dict for the primary model, else first available."""
    return block.get(MODEL, next(iter(block.values())))


def verdict(res):
    """Apply the harness 'real signal' rule (ANALYSIS_HARNESS.md).

    Metric blocks are flat: block[model]["auroc"], block[model]["auroc_ci95"].
    """
    sep = res["separation"]
    feat = _pick(sep["features"])
    auroc = feat["auroc"]
    lo, hi = feat["auroc_ci95"]
    perm_p = res["permutation_test"]["p_value"]
    beats_chance = lo is not None and lo > 0.5
    perm_sig = perm_p < 0.05
    notes = []
    survives_confound = None
    if "covariates_only" in sep and "features_residualized" in sep:
        cov_m = _pick(sep["covariates_only"])
        res_m = _pick(sep["features_residualized"])
        cov_auroc = cov_m["auroc"]
        res_lo = res_m["auroc_ci95"][0]
        beats_cov = auroc > cov_auroc
        residual_above_chance = res_lo is not None and res_lo > 0.5
        survives_confound = beats_cov and residual_above_chance
        notes.append(f"covariate-only AUROC={cov_auroc:.3f}; "
                     f"residualized AUROC={res_m['auroc']:.3f} "
                     f"(CI lo={res_lo:.3f})")
    real = beats_chance and perm_sig and (survives_confound in (None, True))
    return {
        "auroc": auroc, "auroc_ci95": [lo, hi], "perm_p": perm_p,
        "beats_chance": bool(beats_chance), "perm_significant": bool(perm_sig),
        "survives_confound": survives_confound, "real_signal": bool(real),
        "n_sig_features_fdr05": res["n_significant_features_fdr05"],
        "notes": "; ".join(notes),
    }


def run_task(task, X, ids, gc, extra, n_perm=1000):
    Xa, y, groups, cov, cov_names, meta, n_missing = align(task, X, ids, gc, extra)
    raw_dim = int(Xa.shape[1])
    if raw_dim > 128:
        from sklearn.decomposition import TruncatedSVD
        k = min(128, raw_dim - 1, max(2, Xa.shape[0] - 1))
        Xa_model = TruncatedSVD(n_components=k, random_state=0).fit_transform(Xa)
        feat_names = [f"svd_{i}" for i in range(Xa_model.shape[1])]
        feature_preprocessing = f"unsupervised_truncated_svd_{Xa_model.shape[1]}_from_{raw_dim}"
    else:
        Xa_model = Xa
        feat_names = [f"sae_{i}" for i in range(Xa_model.shape[1])]
        feature_preprocessing = "none"
    outdir = os.path.join(RESDIR, task)
    res = run_report(
        Xa_model, y, groups=groups, covariates=cov, outdir=outdir,
        title=f"aim2_{task}", task="classification", seed=0,
        feature_names=feat_names, n_perm=n_perm, n_splits=5, n_boot=1000,
    )
    v = verdict(res)
    # per-held-out-chrom class balance check (adversarial care)
    import collections
    bal = collections.Counter()
    for r in meta:
        if r["split"] == "test":
            bal[(r["chrom"], r["label"])] += 1
    v["covariates"] = cov_names
    v["n_samples"] = int(len(y))
    v["n_features"] = int(Xa_model.shape[1])
    v["n_raw_features"] = raw_dim
    v["feature_preprocessing"] = feature_preprocessing
    v["n_groups"] = int(len(set(groups.tolist())))
    v["n_missing_features"] = int(n_missing)
    v["gc_available"] = True
    v["test_chrom_balance"] = {f"{c}:{l}": n for (c, l), n in sorted(bal.items())}
    return res, v


def write_results_doc(summary):
    lines = ["# Aim 2 results — Evo2-SAE feature content of popgen-flagged regions\n"]
    lines.append("Does the Evo2 layer-26 SAE feature content of an 8 kb window "
                 "centered on a popgen-flagged region separate positives from "
                 "matched controls, **beyond a length/GC/repeat/mappability/"
                 "gene-density confound baseline** and "
                 "**beyond a label-shuffle null**, under a by-chromosome held-out "
                 "split?\n")
    for task in TASKS:
        v = summary.get(task)
        if v is None:
            continue
        title = "A. Selective sweeps vs controls" if task == "sweeps" \
            else "B. Archaic introgression vs controls"
        lines.append(f"## {title}\n")
        lo, hi = v["auroc_ci95"]
        lines.append(f"- held-out AUROC (logreg, GroupKFold by chrom): "
                     f"**{v['auroc']:.3f}** [{lo:.3f}, {hi:.3f}]")
        lines.append(f"- permutation (within-chromosome label shuffle) p = "
                     f"**{v['perm_p']:.4g}**")
        lines.append(f"- covariates: {v['covariates']} "
                     f"(GC available: {v['gc_available']})")
        if v["notes"]:
            lines.append(f"- confound control: {v['notes']}")
        sc = v["survives_confound"]
        sc_s = "n/a (no covariates)" if sc is None else ("yes" if sc else "NO")
        lines.append(f"- beats chance: {'yes' if v['beats_chance'] else 'no'}; "
                     f"perm-significant: {'yes' if v['perm_significant'] else 'no'}; "
                     f"survives confound: {sc_s}")
        lines.append(f"- FDR<0.05 differential features: {v['n_sig_features_fdr05']}")
        lines.append(f"- n={v['n_samples']}, d={v['n_features']}, "
                     f"raw_d={v.get('n_raw_features', v['n_features'])}, "
                     f"preprocess={v.get('feature_preprocessing', 'none')}, "
                     f"groups(chrom)={v['n_groups']}, "
                     f"missing-feature rows={v['n_missing_features']}")
        lines.append(f"- held-out-chrom class balance: {v['test_chrom_balance']}")
        lines.append(f"\n**Verdict: {'REAL, non-confounded signal' if v['real_signal'] else 'NOT established beyond confound/chance'}**\n")
    lines.append("\n## How to read this\n")
    lines.append("A feature set carries genuine, non-confounded signal only when "
                 "all hold: (1) held-out AUROC CI above 0.5, (2) permutation "
                 "p<0.05, (3) features beat covariate-only AND residualized "
                 "features stay above chance. If covariate-only is already high "
                 "and residualized collapses, the apparent signal was "
                 "length/GC/repeat/mappability/gene-density confound, not "
                 "function.\n")
    with open(os.path.join(ROOT, "docs/RESULTS_AIM2.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main_real(n_perm=1000):
    os.makedirs(RESDIR, exist_ok=True)
    summary = {}
    X, ids, gc = load_features(DATADIR)
    extra = load_extra_covariates(DATADIR)
    for task in TASKS:
        print(f"=== {task} ===")
        res, v = run_task(task, X, ids, gc, extra, n_perm=n_perm)
        summary[task] = v
        print(json.dumps(v, indent=2))
    with open(os.path.join(RESDIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    write_results_doc(summary)
    print(f"\nWrote {RESDIR}/summary.json and docs/RESULTS_AIM2.md")


# --------------------------------------------------------------------------- #
# synthetic self-test (planted signal) — runs before real features exist
# --------------------------------------------------------------------------- #
def selftest():
    """Plant signal into a few SAE dims that is *correlated with label but not
    with the covariates*, plus a confound-only case, and assert the driver's
    verdict logic behaves. Uses by-chromosome groups like the real run."""
    rng = np.random.default_rng(0)
    n_per_chrom = 30
    chroms = [f"chr{i}" for i in range(1, 11)]
    n = len(chroms) * n_per_chrom
    groups = np.repeat(chroms, n_per_chrom)
    y = np.tile([0, 1], n // 2)[:n]
    rng.shuffle(y)
    d = 64
    log_len = rng.normal(3.5, 0.3, n)
    gc = rng.uniform(0.35, 0.55, n)
    cov = np.c_[log_len, gc]

    # CASE A: real planted signal in 4 dims, independent of covariates
    Xsig = rng.normal(0, 1, (n, d))
    Xsig[:, :4] += (y[:, None] * 1.6)
    resA = run_report(Xsig, y, groups=groups, covariates=cov,
                      outdir=os.path.join(RESDIR, "_selftest_signal"),
                      title="selftest_signal", task="classification",
                      seed=0, n_perm=200, n_splits=5, n_boot=200)
    vA = verdict(resA)
    print("SELFTEST signal:", json.dumps({k: vA[k] for k in
          ["auroc", "perm_p", "beats_chance", "perm_significant",
           "survives_confound", "real_signal", "n_sig_features_fdr05"]}))

    # CASE B: pure noise -> should NOT be real
    Xn = rng.normal(0, 1, (n, d))
    resB = run_report(Xn, y, groups=groups, covariates=cov,
                      outdir=os.path.join(RESDIR, "_selftest_noise"),
                      title="selftest_noise", task="classification",
                      seed=0, n_perm=200, n_splits=5, n_boot=200)
    vB = verdict(resB)
    print("SELFTEST noise: ", json.dumps({k: vB[k] for k in
          ["auroc", "perm_p", "beats_chance", "perm_significant",
           "real_signal", "n_sig_features_fdr05"]}))

    # CASE C: confounded -> label is driven by GC; features only echo GC.
    yc = (gc > np.median(gc)).astype(int)
    Xc = rng.normal(0, 1, (n, d))
    Xc[:, :4] += (gc[:, None] - gc.mean()) * 6.0  # features carry GC, not label-beyond-GC
    covc = np.c_[log_len, gc]
    resC = run_report(Xc, yc, groups=groups, covariates=covc,
                      outdir=os.path.join(RESDIR, "_selftest_confound"),
                      title="selftest_confound", task="classification",
                      seed=0, n_perm=200, n_splits=5, n_boot=200)
    vC = verdict(resC)
    print("SELFTEST confound:", json.dumps({k: vC[k] for k in
          ["auroc", "perm_p", "survives_confound", "real_signal", "notes"]}))

    assert vA["real_signal"] is True, "planted signal should be called REAL"
    assert vA["n_sig_features_fdr05"] >= 1, "planted dims should be FDR-significant"
    assert vB["real_signal"] is False, "pure noise must not be called real"
    assert vC["survives_confound"] is False, "confound case must fail residualized test"
    assert vC["real_signal"] is False, "confound case must not be called real"
    print("\nALL AIM2 SELF-TESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic planted-signal checks")
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    try:
        if args.selftest:
            selftest()
        else:
            main_real(n_perm=args.n_perm)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
