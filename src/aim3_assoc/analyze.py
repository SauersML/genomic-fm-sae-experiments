#!/usr/bin/env python3
"""
Aim-3 association analysis: do Evo2-SAE feature profiles of individual haplotypes
predict gene expression, held out by individual?

Per gene we ask the adversarial question carefully:
  1. Do SAE haplotype features predict expression on held-out individuals
     (Spearman/R2 with bootstrap CI + permutation null)?
  2. Do they add signal BEYOND ancestry (EUR/AFR)? -> ancestry-covariate model
     (features_residualized must stay above chance) + EUR-only sensitivity run.
  3. Do they add signal BEYOND a trivial variant-burden baseline (ALT-allele count
     in the window = essentially the eQTL genotype)? -> altcount-only baseline +
     features-residualized-on-altcount.
Then aggregate across genes: how many show held-out signal beyond chance, beyond
ancestry, and beyond variant burden, with BH-FDR over genes.

Feature combination per individual: **mean of the two haplotypes**. Justification:
expression is a diploid (additive) readout of both alleles, so the symmetric mean
is the natural per-individual summary and is permutation-invariant to phase labelling
(h1/h2 are arbitrary). Concatenation would double the dimension, break that symmetry,
and tie the result to an arbitrary phase order. (A concat variant is available via
--combine concat for sensitivity.)

Inputs (data/aim3_assoc/): features.npy + ids.txt (from the box), genes.tsv,
outcome.tsv, covariates.tsv, altcount.tsv, splits_pilot.json.
Outputs: results/aim3_assoc/<gene>/ (run_report artifacts) + summary.json +
report.md, and a top-level results/aim3_assoc/aggregate.json.

Run the self-test (synthetic features, no GPU needed) with:  python analyze.py --selftest
"""
import argparse, csv, json, os, sys, tempfile
import numpy as np

ROOT = "/Users/user/bio-interp-experiments"
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import analysis as A  # noqa: E402

DATA = os.path.join(ROOT, "data/aim3_assoc")
RES = os.path.join(ROOT, "results/aim3_assoc")


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def parse_id(rid):
    """'SYMBOL|ENSG|SAMPLE|hH' -> (gene='SYMBOL|ENSG', sample, hap_int)."""
    gene, sample, hap = rid.rsplit("|", 2)
    return gene, sample, int(hap.lstrip("h"))


def load_manifest_ids(data_dir):
    path = os.path.join(data_dir, "manifest.jsonl")
    ids = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            ids.append(str(rec["id"]))
    if not ids:
        raise ValueError(f"{path}: no manifest records")
    return ids


def load_ready_meta(data_dir):
    path = os.path.join(data_dir, "FEATURES_READY")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. Do not run the real analysis until the embedding job "
            "has written FEATURES_READY, features.npy, and ids.txt."
        )
    text = open(path).read().strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: expected JSON metadata or an empty file: {e}") from e


def load_features(data_dir=DATA, require_ready=True, check_manifest=True):
    if require_ready:
        meta = load_ready_meta(data_dir)
    else:
        meta = {}
    fpath = os.path.join(data_dir, "features.npy")
    ipath = os.path.join(data_dir, "ids.txt")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"{fpath} is missing")
    if not os.path.exists(ipath):
        raise FileNotFoundError(f"{ipath} is missing")
    feats = np.load(fpath)
    ids = [l.strip() for l in open(ipath) if l.strip()]
    if feats.ndim != 2:
        raise ValueError(f"{fpath}: expected a 2D matrix, got shape {feats.shape}")
    if feats.shape[0] != len(ids):
        raise ValueError(f"features rows {feats.shape[0]} != ids {len(ids)}")
    if not np.isfinite(feats).all():
        raise ValueError(f"{fpath}: contains NaN or infinite values")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{ipath}: duplicate row ids present")
    if meta:
        if "n" in meta and int(meta["n"]) != feats.shape[0]:
            raise ValueError(f"FEATURES_READY n={meta['n']} != features rows {feats.shape[0]}")
        if "dim" in meta and int(meta["dim"]) != feats.shape[1]:
            raise ValueError(f"FEATURES_READY dim={meta['dim']} != features dim {feats.shape[1]}")
    if check_manifest:
        manifest_ids = load_manifest_ids(data_dir)
        missing = sorted(set(manifest_ids) - set(ids))
        extra = sorted(set(ids) - set(manifest_ids))
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {len(missing)} manifest ids, first={missing[:3]}")
            if extra:
                details.append(f"extra {len(extra)} ids not in manifest, first={extra[:3]}")
            raise ValueError(f"{ipath} does not match manifest.jsonl: {'; '.join(details)}")
        if len(ids) != len(manifest_ids):
            raise ValueError(f"{ipath}: {len(ids)} ids but manifest has {len(manifest_ids)} records")
    return feats, ids


def build_per_individual(feats, ids, combine, require_complete=True):
    """Collapse the two haplotype vectors per (gene, sample) into one row.

    Returns dict gene -> (samples list, X array [n_samples, d_combined])."""
    by_gs = {}  # (gene,sample) -> {hap: row_index}
    for i, rid in enumerate(ids):
        g, s, h = parse_id(rid)
        if h not in (1, 2):
            raise ValueError(f"{rid}: hap must be h1 or h2")
        if h in by_gs.setdefault((g, s), {}):
            raise ValueError(f"{rid}: duplicate haplotype row")
        by_gs[(g, s)][h] = i
    genes = sorted({g for (g, _s) in by_gs})
    out = {}
    for g in genes:
        samples, rows = [], []
        incomplete = []
        for (gg, s), hd in sorted(by_gs.items()):
            if gg != g:
                continue
            if 1 not in hd or 2 not in hd:
                incomplete.append(s)
                continue  # need both haplotypes
            v1, v2 = feats[hd[1]], feats[hd[2]]
            if combine == "mean":
                rows.append((v1 + v2) / 2.0)
            elif combine == "concat":
                rows.append(np.concatenate([v1, v2]))
            else:
                raise ValueError(combine)
            samples.append(s)
        if incomplete and require_complete:
            raise ValueError(f"{g}: missing h1/h2 pair for {len(incomplete)} samples, first={incomplete[:3]}")
        if not rows:
            raise ValueError(f"{g}: no complete per-individual haplotype pairs")
        out[g] = (samples, np.asarray(rows))
    return out


# --------------------------------------------------------------------------- #
# per-gene analysis
# --------------------------------------------------------------------------- #
def aligned_tables(gene, samples, data_dir=DATA):
    """Return y(expression), afr_dummy, altcount(mean_alt), split-label aligned to `samples`."""
    out = {r["sample"]: r for r in load_tsv(os.path.join(data_dir, "outcome.tsv"))
           if r["locus_id"] == gene}
    ac = {r["sample"]: r for r in load_tsv(os.path.join(data_dir, "altcount.tsv"))
          if r["locus_id"] == gene}
    missing_out = [s for s in samples if s not in out]
    missing_ac = [s for s in samples if s not in ac]
    if missing_out or missing_ac:
        details = []
        if missing_out:
            details.append(f"outcome missing {len(missing_out)} samples, first={missing_out[:3]}")
        if missing_ac:
            details.append(f"altcount missing {len(missing_ac)} samples, first={missing_ac[:3]}")
        raise ValueError(f"{gene}: companion table mismatch: {'; '.join(details)}")
    y = np.array([float(out[s]["expression"]) for s in samples])
    afr = np.array([float(out[s]["afr_dummy"]) for s in samples])
    alt = np.array([float(ac[s]["mean_alt"]) for s in samples])
    split = np.array([out[s]["split"] for s in samples])
    return y, afr, alt, split


def heldout_test_eval(X, y, train_mask, test_mask, covariates=None, seed=0):
    """Train Ridge on train rows, predict the held-out TEST individuals.
    Returns Spearman + R2 on the explicit test set, plus a group-bootstrap CI
    (resampling test individuals) for Spearman, and a permutation p (shuffle y in
    train, refit, recompute test Spearman)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    def fit_predict(Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    if covariates is not None:
        Ctr, Cte = covariates[train_mask], covariates[test_mask]
        Xtr = np.c_[Xtr, Ctr]
        Xte = np.c_[Xte, Cte]
    yp = fit_predict(Xtr, ytr, Xte)
    rho = float(stats.spearmanr(yp, yte)[0])
    from sklearn.metrics import r2_score
    r2 = float(r2_score(yte, yp))
    # bootstrap CI over test individuals
    rng = np.random.default_rng(seed)
    boots = []
    n = len(yte)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        if len(np.unique(yte[idx])) < 3:
            continue
        boots.append(stats.spearmanr(yp[idx], yte[idx])[0])
    ci = (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))) if boots else (np.nan, np.nan)
    # permutation null: shuffle train labels, refit, recompute test rho
    null = []
    for k in range(200):
        yp_ = fit_predict(Xtr, rng.permutation(ytr), Xte)
        null.append(stats.spearmanr(yp_, yte)[0])
    null = np.array(null)
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1))
    return {"spearman": rho, "spearman_ci95": ci, "r2": r2,
            "perm_p_twosided": p, "n_test": int(n),
            "null_mean": float(np.nanmean(null)), "null_std": float(np.nanstd(null))}


def analyze_gene(gene, samples, X, combine, eur_only=False, n_perm=500, seed=0,
                 data_dir=DATA, res_dir=RES):
    y, afr, alt, split = aligned_tables(gene, samples, data_dir)
    samples = np.array(samples)

    if eur_only:
        keep = afr == 0
        samples, X, y, afr, alt, split = samples[keep], X[keep], y[keep], afr[keep], alt[keep], split[keep]

    raw_dim = int(X.shape[1])
    if raw_dim > 128:
        from sklearn.decomposition import TruncatedSVD
        k = min(128, raw_dim - 1, max(2, X.shape[0] - 1))
        X_model = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
        feature_preprocessing = f"unsupervised_truncated_svd_{X_model.shape[1]}_from_{raw_dim}"
    else:
        X_model = X
        feature_preprocessing = "none"

    groups = samples  # each individual = its own group (one row per gene); leakage-safe CV
    cov_anc = afr.reshape(-1, 1)
    cov_alt = alt.reshape(-1, 1)

    safe = gene.replace("|", "_") + ("__EURonly" if eur_only else "")
    outdir = os.path.join(res_dir, safe)

    # 1) features vs ancestry covariate (CV; harness handles residualization + null)
    rep_anc = A.run_report(
        X_model, y, groups=groups, covariates=cov_anc, outdir=outdir,
        title=f"{gene}{' (EUR-only)' if eur_only else ''} | SAE-features vs ancestry",
        task="regression", seed=seed, n_perm=n_perm,
        n_splits=min(5, len(np.unique(groups))), n_boot=500,
    )
    # 2) features vs variant-burden (altcount) covariate -- the trivial-eQTL bar
    rep_alt = A.evaluate_separation(
        X_model, y, groups=groups, covariates=cov_alt, task="regression",
        seed=seed, n_splits=min(5, len(np.unique(groups))), n_boot=500,
    )
    # 3) altcount baseline ALONE predicting expression (is the simple burden enough?)
    rep_burden = A.evaluate_separation(
        cov_alt, y, groups=groups, covariates=None, task="regression",
        seed=seed, n_splits=min(5, len(np.unique(groups))), n_boot=500,
    )

    res = {
        "gene": gene, "eur_only": eur_only, "combine": combine,
        "n_individuals": int(len(samples)),
        "n_features": int(X_model.shape[1]),
        "n_raw_features": raw_dim,
        "feature_preprocessing": feature_preprocessing,
        "cv": {
            "features": rep_anc["separation"]["features"],
            "covariates_only_ancestry": rep_anc["separation"].get("covariates_only"),
            "features_residualized_ancestry": rep_anc["separation"].get("features_residualized"),
            "features_residualized_altcount": rep_alt.get("features_residualized"),
            "altcount_only": rep_burden["features"],
            "permutation_p": rep_anc["permutation_test"]["p_value"],
            "permutation_observed": rep_anc["permutation_test"]["observed"],
            "n_significant_features_fdr05": rep_anc["n_significant_features_fdr05"],
        },
    }

    # 4) explicit held-out TEST evaluation (train on train+val, predict test)
    if not eur_only:  # explicit test set only meaningful with full ancestry mix
        test_mask = split == "test"
        train_mask = ~test_mask
        if test_mask.sum() >= 10:
            res["heldout_test"] = {
                "features": heldout_test_eval(X_model, y, train_mask, test_mask, seed=seed),
                "features_plus_ancestry": heldout_test_eval(
                    X_model, y, train_mask, test_mask, covariates=cov_anc, seed=seed),
                "ancestry_only": heldout_test_eval(
                    cov_anc, y, train_mask, test_mask, seed=seed),
                "altcount_only": heldout_test_eval(
                    cov_alt, y, train_mask, test_mask, seed=seed),
            }
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    return res


# --------------------------------------------------------------------------- #
# aggregation across genes (BH-FDR)
# --------------------------------------------------------------------------- #
def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def aggregate(per_gene):
    genes = [r["gene"] for r in per_gene]
    cv_p = [r["cv"]["permutation_p"] for r in per_gene]
    cv_fdr = bh_fdr(cv_p)
    rows = []
    for r, q in zip(per_gene, cv_fdr):
        cv = r["cv"]
        feat_sp = _metric(cv["features"], "spearman")
        feat_sp_ci = _metric_ci(cv["features"], "spearman")
        resid_anc = _metric(cv.get("features_residualized_ancestry"), "spearman")
        resid_anc_ci = _metric_ci(cv.get("features_residualized_ancestry"), "spearman")
        resid_alt = _metric(cv.get("features_residualized_altcount"), "spearman")
        anc_only = _metric(cv.get("covariates_only_ancestry"), "spearman")
        alt_only = _metric(cv.get("altcount_only"), "spearman")
        ho = r.get("heldout_test", {})
        rows.append({
            "gene": r["gene"], "n": r["n_individuals"],
            "n_features": r.get("n_features"),
            "n_raw_features": r.get("n_raw_features"),
            "feature_preprocessing": r.get("feature_preprocessing"),
            "cv_features_spearman": feat_sp, "cv_features_spearman_ci95": feat_sp_ci,
            "cv_perm_p": cv["permutation_p"], "cv_perm_q_fdr": float(q),
            "cv_resid_ancestry_spearman": resid_anc, "cv_resid_ancestry_ci95": resid_anc_ci,
            "cv_resid_altcount_spearman": resid_alt,
            "cv_ancestry_only_spearman": anc_only,
            "cv_altcount_only_spearman": alt_only,
            "heldout_test_features_spearman": ho.get("features", {}).get("spearman"),
            "heldout_test_features_ci95": ho.get("features", {}).get("spearman_ci95"),
            "heldout_test_perm_p": ho.get("features", {}).get("perm_p_twosided"),
            "heldout_test_ancestry_only_spearman": ho.get("ancestry_only", {}).get("spearman"),
            "heldout_test_altcount_only_spearman": ho.get("altcount_only", {}).get("spearman"),
        })

    def beats(row):
        ci = row["cv_features_spearman_ci95"]
        rci = row["cv_resid_ancestry_ci95"]
        return {
            "cv_signal": (row["cv_perm_q_fdr"] < 0.05) and bool(ci) and ci[0] > 0,
            "beyond_ancestry": bool(rci) and rci[0] > 0,
            "beyond_altcount": (row["cv_resid_altcount_spearman"] or -1) > 0,
        }
    flags = [beats(r) for r in rows]
    summary = {
        "n_genes": len(rows),
        "n_cv_signal": sum(f["cv_signal"] for f in flags),
        "n_beyond_ancestry": sum(f["beyond_ancestry"] for f in flags),
        "n_beyond_altcount": sum(f["beyond_altcount"] for f in flags),
        "per_gene": [dict(r, **{"flags": f}) for r, f in zip(rows, flags)],
    }
    return summary


def _metric(block, key):
    if not block:
        return None
    # block is {model: {metric: val, metric_ci95: (lo,hi)}}; take best (ridge) then max
    vals = [m.get(key) for m in block.values() if isinstance(m, dict) and m.get(key) is not None]
    return float(max(vals)) if vals else None


def _metric_ci(block, key):
    if not block:
        return None
    best, bestci = None, None
    for m in block.values():
        if isinstance(m, dict) and m.get(key) is not None:
            if best is None or m[key] > best:
                best, bestci = m[key], m.get(key + "_ci95")
    return tuple(bestci) if bestci else None


# --------------------------------------------------------------------------- #
def write_report(summary, combine, eur_summary=None, res_dir=RES):
    os.makedirs(res_dir, exist_ok=True)
    L = []
    A_ = L.append
    A_("# Aim 3 — do Evo2-SAE haplotype feature profiles predict gene expression?\n")
    A_(f"Per-individual feature = **{combine}** of the two haplotype SAE vectors. "
       "Held out by individual (CV groups = individual; plus an explicit held-out test set).\n")
    A_(f"- genes analysed: **{summary['n_genes']}**")
    prep = sorted({r.get("feature_preprocessing", "none") for r in summary["per_gene"]})
    raw_dims = sorted({r.get("n_raw_features", r.get("n_features")) for r in summary["per_gene"]})
    A_(f"- feature preprocessing: **{', '.join(prep)}**; raw feature dims: **{raw_dims}**")
    A_(f"- genes with CV held-out signal (FDR<0.05 perm AND Spearman CI>0): "
       f"**{summary['n_cv_signal']}/{summary['n_genes']}**")
    A_(f"- ... that survive **ancestry** residualization (resid Spearman CI>0): "
       f"**{summary['n_beyond_ancestry']}/{summary['n_genes']}**")
    A_(f"- ... that add signal beyond the **ALT-allele-burden** baseline (resid Spearman>0): "
       f"**{summary['n_beyond_altcount']}/{summary['n_genes']}**\n")
    A_("## Per-gene (CV)\n")
    A_("| gene | n | feat Spearman [CI] | perm q | resid(ancestry) | ancestry-only | "
       "resid(altcount) | altcount-only | heldout-test feat [CI] |")
    A_("|---|---|---|---|---|---|---|---|---|")
    for r in summary["per_gene"]:
        ci = r["cv_features_spearman_ci95"]
        cistr = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "-"
        ho = r["heldout_test_features_spearman"]
        hoci = r["heldout_test_features_ci95"]
        hostr = f"{ho:.2f} [{hoci[0]:.2f},{hoci[1]:.2f}]" if (ho is not None and hoci) else (f"{ho:.2f}" if ho is not None else "-")
        A_("| {g} | {n} | {f} {c} | {q:.3g} | {ra} | {ao} | {rl} | {al} | {h} |".format(
            g=r["gene"], n=r["n"],
            f=("%.2f" % r["cv_features_spearman"]) if r["cv_features_spearman"] is not None else "-",
            c=cistr, q=r["cv_perm_q_fdr"],
            ra=("%.2f" % r["cv_resid_ancestry_spearman"]) if r["cv_resid_ancestry_spearman"] is not None else "-",
            ao=("%.2f" % r["cv_ancestry_only_spearman"]) if r["cv_ancestry_only_spearman"] is not None else "-",
            rl=("%.2f" % r["cv_resid_altcount_spearman"]) if r["cv_resid_altcount_spearman"] is not None else "-",
            al=("%.2f" % r["cv_altcount_only_spearman"]) if r["cv_altcount_only_spearman"] is not None else "-",
            h=hostr))
    if eur_summary:
        A_("\n## EUR-only sensitivity (ancestry confound removed by design)\n")
        A_(f"- genes with CV held-out signal (FDR<0.05 & CI>0): "
           f"**{eur_summary['n_cv_signal']}/{eur_summary['n_genes']}**\n")
        A_("| gene | n | feat Spearman [CI] | perm q | resid(altcount) |")
        A_("|---|---|---|---|---|")
        for r in eur_summary["per_gene"]:
            ci = r["cv_features_spearman_ci95"]
            cistr = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "-"
            A_("| {g} | {n} | {f} {c} | {q:.3g} | {rl} |".format(
                g=r["gene"], n=r["n"],
                f=("%.2f" % r["cv_features_spearman"]) if r["cv_features_spearman"] is not None else "-",
                c=cistr, q=r["cv_perm_q_fdr"],
                rl=("%.2f" % r["cv_resid_altcount_spearman"]) if r["cv_resid_altcount_spearman"] is not None else "-"))
    A_("\n## Interpretation rules\n")
    A_("- A gene shows **real, non-confounded** SAE signal only if: CV perm q<0.05, "
       "feature Spearman CI>0, the ancestry-residualized Spearman CI stays >0, "
       "**and** it holds in the EUR-only run. ")
    A_("- If features predict only when ancestry varies (collapse EUR-only / on residualization), "
       "that's the ancestry confound, not haplotype-specific regulation.")
    A_("- The ALT-allele-burden baseline is the trivial-eQTL bar: SAE features are interesting "
       "insofar as they match/exceed it and add beyond it.")
    open(os.path.join(res_dir, "report.md"), "w").write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# self-test on synthetic features (no GPU / no features.npy needed)
# --------------------------------------------------------------------------- #
def selftest():
    print("[selftest] building synthetic features.npy + ids.txt in a temporary directory ...")
    rng = np.random.default_rng(0)
    genes = load_tsv(os.path.join(DATA, "genes.tsv"))[:3]
    samples = [l.strip() for l in open(os.path.join(DATA, "samples_pilot.txt")) if l.strip()]
    outc = {(r["locus_id"], r["sample"]): r for r in load_tsv(os.path.join(DATA, "outcome.tsv"))}
    d = 64
    ids, rows = [], []
    # z-score each gene's expression so the injected effect size is comparable
    ymean, ystd = {}, {}
    for g in genes:
        gid = g["locus_id"]
        ys = np.array([float(outc[(gid, s)]["expression"]) for s in samples])
        ymean[gid], ystd[gid] = ys.mean(), ys.std() + 1e-9
    for g in genes:
        gid = g["locus_id"]
        for s in samples:
            yz = (float(outc[(gid, s)]["expression"]) - ymean[gid]) / ystd[gid]
            afr = float(outc[(gid, s)]["afr_dummy"])
            for h in (1, 2):
                base = rng.normal(0, 1, d)
                # gene 0: features genuinely track expression (real signal) ->
                #         must be DETECTED (CV spearman CI>0, perm sig, survives ancestry+altcount).
                # gene 1: feature[0] tracks ancestry ONLY (pure confound) ->
                #         must be CAUGHT (collapses on ancestry-residualization / EUR-only).
                # gene 2: pure noise -> must stay null.
                if gid == genes[0]["locus_id"]:
                    base[0] += 2.0 * yz
                    base[1] += 1.5 * yz
                elif gid == genes[1]["locus_id"]:
                    base[0] += 3.0 * afr
                ids.append(f"{gid}|{s}|h{h}")
                rows.append(base)
    with tempfile.TemporaryDirectory(prefix="aim3_assoc_selftest_") as tmp:
        data_dir = os.path.join(tmp, "data")
        res_dir = os.path.join(tmp, "results")
        os.makedirs(data_dir, exist_ok=True)
        for name in ("outcome.tsv", "altcount.tsv", "genes.tsv", "samples_pilot.txt"):
            src = os.path.join(DATA, name)
            dst = os.path.join(data_dir, name)
            open(dst, "w").write(open(src).read())
        with open(os.path.join(data_dir, "manifest.jsonl"), "w") as fh:
            for rid in ids:
                g, s, h = parse_id(rid)
                fh.write(json.dumps({"id": rid, "gene": g, "sample": s, "hap": h}) + "\n")
        np.save(os.path.join(data_dir, "features.npy"), np.asarray(rows))
        open(os.path.join(data_dir, "ids.txt"), "w").write("\n".join(ids) + "\n")
        open(os.path.join(data_dir, "FEATURES_READY"), "w").write(
            json.dumps({"n": len(ids), "dim": d, "selftest": True}) + "\n"
        )
        print(f"[selftest] wrote {len(ids)} synthetic rows ({len(genes)} genes). Running analysis ...")
        summary = run_all(combine="mean", n_perm=300, do_eur=True,
                          data_dir=data_dir, res_dir=res_dir,
                          require_ready=True, check_manifest=True)
        eur_summary = aggregate_eur_from_disk(genes, res_dir)
    by = {r["gene"]: r for r in summary["per_gene"]}
    g_sig, g_conf, g_noise = (g["locus_id"] for g in genes)

    sig = by[g_sig]
    print(f"[selftest] SIGNAL gene  {g_sig}: spearman={sig['cv_features_spearman']:.2f} "
          f"CI={sig['cv_features_spearman_ci95']} q={sig['cv_perm_q_fdr']:.3g} "
          f"resid_anc={sig['cv_resid_ancestry_spearman']:.2f} resid_alt={sig['cv_resid_altcount_spearman']:.2f}")
    assert sig["cv_features_spearman"] > 0.3, "signal gene not detected (spearman)"
    assert sig["cv_features_spearman_ci95"][0] > 0, "signal CI not above 0"
    assert sig["cv_perm_q_fdr"] < 0.05, "signal perm q not significant"
    assert sig["cv_resid_ancestry_spearman"] > 0.2, "signal lost after ancestry residualization"
    assert sig["flags"]["cv_signal"] and sig["flags"]["beyond_ancestry"], "signal flags wrong"

    conf = by[g_conf]
    print(f"[selftest] CONFOUND gene {g_conf}: spearman={conf['cv_features_spearman']:.2f} "
          f"ancestry_only={conf['cv_ancestry_only_spearman']:.2f} "
          f"resid_anc={conf['cv_resid_ancestry_spearman']:.2f}")
    # confound: ancestry-only explains it; residualizing ancestry should collapse the feature signal
    assert conf["cv_resid_ancestry_spearman"] < 0.2, "ancestry confound NOT caught (resid stayed high)"

    noise = by[g_noise]
    print(f"[selftest] NOISE gene   {g_noise}: spearman={noise['cv_features_spearman']:.2f} "
          f"q={noise['cv_perm_q_fdr']:.3g}")
    assert not noise["flags"]["cv_signal"], "noise gene falsely flagged as signal"

    # EUR-only: the real-signal gene should survive; the confound gene should NOT
    eur_by = {r["gene"]: r for r in eur_summary["per_gene"]}
    assert eur_by[g_sig]["cv_features_spearman"] > 0.25, "signal gene lost in EUR-only"
    print("[selftest] EUR-only: signal gene spearman="
          f"{eur_by[g_sig]['cv_features_spearman']:.2f}, "
          f"confound gene spearman={eur_by[g_conf]['cv_features_spearman']:.2f}")
    print("[selftest] ALL ASSERTIONS PASSED")


def aggregate_eur_from_disk(genes, res_dir=RES):
    eur = []
    for g in genes:
        p = os.path.join(res_dir, g["locus_id"].replace("|", "_") + "__EURonly", "summary.json")
        if os.path.exists(p):
            eur.append(json.load(open(p)))
    return aggregate(eur)


def run_all(combine="mean", n_perm=500, do_eur=True, data_dir=DATA, res_dir=RES,
            require_ready=True, check_manifest=True):
    feats, ids = load_features(data_dir, require_ready=require_ready, check_manifest=check_manifest)
    print(f"[load] features {feats.shape}, ids {len(ids)}")
    per_gene_feats = build_per_individual(feats, ids, combine)
    per_gene = []
    eur = []
    for gene, (samples, X) in per_gene_feats.items():
        print(f"[gene] {gene}: {len(samples)} individuals, X={X.shape}")
        per_gene.append(analyze_gene(gene, samples, X, combine, eur_only=False,
                                     n_perm=n_perm, data_dir=data_dir, res_dir=res_dir))
        if do_eur:
            eur.append(analyze_gene(gene, samples, X, combine, eur_only=True,
                                    n_perm=n_perm, data_dir=data_dir, res_dir=res_dir))
    summary = aggregate(per_gene)
    eur_summary = aggregate(eur) if do_eur and eur else None
    os.makedirs(res_dir, exist_ok=True)
    json.dump({"all": summary, "eur_only": eur_summary, "combine": combine},
              open(os.path.join(res_dir, "aggregate.json"), "w"), indent=2)
    write_report(summary, combine, eur_summary, res_dir)
    print(f"[done] {summary['n_cv_signal']}/{summary['n_genes']} genes CV signal; "
          f"{summary['n_beyond_ancestry']} beyond ancestry; "
          f"{summary['n_beyond_altcount']} beyond altcount.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", choices=["mean", "concat"], default="mean")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--no-eur", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    try:
        if args.selftest:
            selftest()
        else:
            run_all(combine=args.combine, n_perm=args.n_perm, do_eur=not args.no_eur)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
