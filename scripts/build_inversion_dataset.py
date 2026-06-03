#!/usr/bin/env python3
"""Build an HPRC inversion-specific table from the release2 wave VCF.

The Aim-1 SV table intentionally kept sequence-resolved events with
|len(ALT)-len(REF)| >= 50, which captures indels. HPRC inversion calls in the
wave VCF are usually balanced sequence substitutions marked by the INFO flag
`INV`; they therefore need their own extraction path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/"
    "release2/minigraph-cactus/hprc-v2.0-mc-grch38.wave.vcf.gz"
)
GENCODE = ROOT / "data/annotations/gencode_v44_features.pkl"
CCRE = ROOT / "data/annotations/encode_ccre_features.pkl"
OUTDIR = ROOT / "data/inversions"
FLANK = 3072
MAX_ALLELE = 1024


def parse_info(info: str) -> dict[str, object]:
    out: dict[str, object] = {"inv_flag": False}
    for item in info.split(";"):
        if item == "INV":
            out["inv_flag"] = True
            continue
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        if key in {"AC", "AN", "NS"}:
            if key == "AC":
                out["ac_values"] = val
            else:
                try:
                    out[key.lower()] = int(val)
                except ValueError:
                    out[key.lower()] = np.nan
        elif key == "AF":
            out["af_values"] = val
        elif key == "LEN":
            out["info_len_values"] = val
        elif key == "TYPE":
            out["type_info"] = val
        elif key == "ORIGIN":
            out["origin"] = val
        elif key in {"LV", "PS", "CONFLICT"}:
            out[key.lower()] = val
    return out


def allele_item(raw: object, idx: int, cast, default=np.nan):
    if raw is None or pd.isna(raw):
        return default
    vals = str(raw).split(",")
    if not vals:
        return default
    val = vals[min(idx, len(vals) - 1)]
    try:
        return cast(val)
    except (TypeError, ValueError):
        return default


def overlaps(arr: np.ndarray | None, start: int, end: int) -> bool:
    if arr is None or len(arr) == 0:
        return False
    starts = arr[:, 0]
    idx = np.searchsorted(starts, end, side="right") - 1
    return idx >= 0 and arr[idx, 1] > start and arr[idx, 0] < end


def load_annotations() -> tuple[dict, dict]:
    with open(GENCODE, "rb") as fh:
        gencode = pickle.load(fh)
    with open(CCRE, "rb") as fh:
        ccre = pickle.load(fh)
    return gencode, ccre


def label_interval(gencode: dict, ccre: dict, chrom: str, start: int, end: int) -> dict[str, object]:
    end = max(end, start + 1)
    cds = overlaps(gencode["cds"].get(chrom), start, end)
    splice = overlaps(gencode["splice"].get(chrom), start, end)
    utr = overlaps(gencode["utr"].get(chrom), start, end)
    exon = overlaps(gencode["exon"].get(chrom), start, end)
    gene_any = overlaps(gencode["gene_any"].get(chrom), start, end)
    gene_coding = overlaps(gencode["gene_coding"].get(chrom), start, end)
    ccre_any = overlaps(ccre["any"].get(chrom), start, end)
    if cds:
        consequence = "cds"
    elif splice:
        consequence = "splice"
    elif utr:
        consequence = "utr"
    elif exon:
        consequence = "exon_noncod"
    elif gene_any:
        consequence = "intronic"
    elif ccre_any:
        consequence = "regulatory"
    else:
        consequence = "intergenic"
    coarse = {
        "cds": "coding",
        "splice": "coding",
        "utr": "noncoding_genic",
        "exon_noncod": "noncoding_genic",
        "intronic": "noncoding_genic",
        "regulatory": "regulatory",
        "intergenic": "intergenic",
    }[consequence]
    return {
        "ov_cds": cds,
        "ov_splice": splice,
        "ov_utr": utr,
        "ov_exon": exon,
        "ov_gene_any": gene_any,
        "ov_gene_coding": gene_coding,
        "ov_ccre": ccre_any,
        "consequence": consequence,
        "consequence_coarse": coarse,
        "is_coding_disrupting": int(consequence in {"cds", "splice"}),
    }


@dataclass
class BuildResult:
    full: pd.DataFrame
    pilot: pd.DataFrame


def stream_inv_records(limit: int | None = None) -> list[dict[str, object]]:
    cmd = ["bcftools", "view", "-i", "INV=1", "-H", URL]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    rows: list[dict[str, object]] = []
    for line in proc.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 8:
            continue
        chrom, pos_s, vcf_id, ref, alt, _qual, _flt, info = fields[:8]
        pos = int(pos_s)
        parsed = parse_info(info)
        if not parsed.get("inv_flag", False):
            continue
        ref_len = len(ref)
        start0 = pos - 1
        end0 = start0 + max(ref_len, 1)
        for allele_idx, alt1 in enumerate(alt.split(",")):
            alt_len = len(alt1)
            info_len = allele_item(parsed.get("info_len_values"), allele_idx, lambda x: abs(int(x)))
            inv_len = int(info_len) if pd.notna(info_len) else max(ref_len, alt_len)
            ac = allele_item(parsed.get("ac_values"), allele_idx, int)
            af = allele_item(parsed.get("af_values"), allele_idx, float)
            typ = allele_item(parsed.get("type_info"), allele_idx, str, default="")
            h = hashlib.sha1(
                f"{chrom}:{pos}:{allele_idx}:{ref}:{alt1}:INV".encode()
            ).hexdigest()[:12]
            row: dict[str, object] = {
                "sv_id": f"hprcv2_{chrom}_{pos}_INV_a{allele_idx + 1}_{h}",
                "chrom": chrom,
                "pos": pos,
                "vcf_id": vcf_id,
                "allele_index": allele_idx + 1,
                "ref": ref,
                "alt": alt1,
                "ref_len": ref_len,
                "alt_len": alt_len,
                "len_delta": alt_len - ref_len,
                "inv_len": inv_len,
                "svtype": "INV",
                "af": af,
                "ac": ac,
                "type_allele": typ,
                "ref_start0": start0,
                "ref_end0": end0,
                "window_start0": start0 - FLANK,
                "window_end0": end0 + FLANK,
                "flank_bp": FLANK,
                "max_allele": MAX_ALLELE,
            }
            row.update({k: v for k, v in parsed.items() if k not in {"af_values", "ac_values"}})
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break
    if limit is None:
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(f"bcftools failed with {rc}: {err}")
    else:
        proc.terminate()
    return rows


def choose_pilot(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = ["cds", "splice", "utr", "exon_noncod", "intronic", "regulatory", "intergenic"]
    parts = []
    per_class = max(1, n // len(classes))
    for cls in classes:
        sub = df[df["consequence"] == cls]
        if len(sub) == 0:
            continue
        take = min(per_class, len(sub))
        parts.append(sub.sample(take, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(parts) if parts else df.head(0)
    if len(out) < min(n, len(df)):
        rest = df.drop(out.index, errors="ignore")
        take = min(n - len(out), len(rest))
        if take > 0:
            out = pd.concat([out, rest.sample(take, random_state=seed + 1)])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def build(limit: int | None, pilot_n: int, seed: int) -> BuildResult:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gencode, ccre = load_annotations()
    rows = stream_inv_records(limit=limit)
    labeled = []
    for row in rows:
        labels = label_interval(
            gencode,
            ccre,
            str(row["chrom"]),
            int(row["ref_start0"]),
            int(row["ref_end0"]),
        )
        row.update(labels)
        labeled.append(row)
    df = pd.DataFrame(labeled)
    if len(df) == 0:
        raise RuntimeError("No INV records found")
    df["af"] = pd.to_numeric(df.get("af"), errors="coerce")
    df["ac"] = pd.to_numeric(df.get("ac"), errors="coerce")
    df["an"] = pd.to_numeric(df.get("an"), errors="coerce")
    df["ns"] = pd.to_numeric(df.get("ns"), errors="coerce")
    df["log_inv_len"] = np.log10(pd.to_numeric(df["inv_len"]).clip(lower=1))
    df["is_balanced"] = df["len_delta"].abs() <= 10
    df.to_parquet(OUTDIR / "hprc_inversions.parquet", index=False)
    public_cols = [c for c in df.columns if c not in {"ref", "alt"}]
    df[public_cols].to_csv(OUTDIR / "hprc_inversions.summary.tsv", sep="\t", index=False)
    pilot = choose_pilot(df, pilot_n, seed)
    pilot.to_parquet(OUTDIR / "hprc_inversions_pilot.parquet", index=False)
    pilot[public_cols].to_csv(OUTDIR / "hprc_inversions_pilot.summary.tsv", sep="\t", index=False)
    with open(OUTDIR / "summary.json", "w") as fh:
        json.dump({
            "n_inversions": int(len(df)),
            "n_pilot": int(len(pilot)),
            "chrom_counts": df["chrom"].value_counts().to_dict(),
            "type_allele_counts": df["type_allele"].fillna("NA").value_counts().head(20).to_dict(),
            "records_with_multiple_inversion_alleles": int((df.groupby(["chrom", "pos", "vcf_id"]).size() > 1).sum()),
            "consequence_counts": df["consequence"].value_counts().to_dict(),
            "balanced_fraction_abs_delta_le_10": float(df["is_balanced"].mean()),
            "inv_len_quantiles": {str(k): float(v) for k, v in df["inv_len"].quantile([0.1, 0.5, 0.9, 0.99]).items()},
            "af_quantiles": {str(k): float(v) for k, v in df["af"].dropna().quantile([0.1, 0.5, 0.9, 0.99]).items()},
        }, fh, indent=2)
    return BuildResult(full=df, pilot=pilot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pilot-n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    res = build(args.limit, args.pilot_n, args.seed)
    print(f"inversions: {len(res.full)}")
    print(res.full["consequence"].value_counts().to_string())
    print("\ntype_allele:")
    print(res.full["type_allele"].fillna("NA").value_counts().head(15).to_string())
    print("\nlength quantiles:")
    print(res.full["inv_len"].quantile([.1, .5, .9, .99]).to_string())
    print(f"\npilot: {len(res.pilot)}")


if __name__ == "__main__":
    main()
