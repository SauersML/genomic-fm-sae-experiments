#!/usr/bin/env python3
"""Render local plots and docs for inversion SAE results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "inversions"
PLOTS = ROOT / "plots"
DOC = ROOT / "docs" / "RESULTS_INVERSIONS.md"


def bootstrap_ci(values, seed=0, n=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5]).tolist()


def load_json(path):
    return json.loads(path.read_text())


def savefig(fig, name):
    for d in (PLOTS, OUT):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=220)
    plt.close(fig)


def plot_profile(summary):
    l1 = np.load(OUT / "synthetic_profile_l1.npy")
    l2 = np.load(OUT / "synthetic_profile_l2.npy")
    inv_start, inv_end = 2000, 6000
    x = np.arange(len(l1))
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(x, l1, color="#264653", lw=1.0, label="mean per-token L1")
    ax2 = ax.twinx()
    ax2.plot(x, l2, color="#C44536", lw=0.9, alpha=0.9, label="mean per-token L2")
    ax.axvspan(inv_start, inv_end, color="#E9C46A", alpha=0.16, label="inverted 4 kb interior")
    for pos, label in [(inv_start, "bp1"), (inv_end, "bp2")]:
        ax.axvline(pos, color="black", lw=0.9, ls="--")
        ax.text(pos + 30, ax.get_ylim()[1] * 0.92, label, fontsize=8)
    ax.set_title("Evo2 layer-26 SAE delta for synthetic 4 kb inversions")
    ax.set_xlabel("position in 8 kb hg38 window")
    ax.set_ylabel("L1 delta")
    ax2.set_ylabel("L2 delta")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
    fig.tight_layout()
    savefig(fig, "inversion_delta_profile.png")


def plot_strand():
    summary = load_json(OUT / "summary.json")
    scatter = np.load(OUT / "strand_scatter_points.npy")
    strand = pd.read_csv(OUT / "strand_symmetry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    hb = axes[0].hexbin(scatter[:, 0], scatter[:, 1], gridsize=55, mincnt=1, cmap="viridis")
    lim = float(np.percentile(scatter, 99.5))
    axes[0].plot([0, lim], [0, lim], color="white", lw=0.8)
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel("ref interior pooled SAE")
    axes[0].set_ylabel("alt revcomp interior pooled SAE")
    axes[0].set_title("Matched forward vs reverse-complement content")
    fig.colorbar(hb, ax=axes[0], label="feature bins")
    axes[1].hist(strand["pearson"], bins=28, color="#2A9D8F", alpha=0.75, label="all features")
    axes[1].hist(strand["active_pearson"], bins=28, color="#E76F51", alpha=0.65, label="active union")
    axes[1].axvline(summary["strand_symmetry"]["pearson_mean"], color="black", lw=1.0)
    axes[1].set_xlabel("per-window Pearson r")
    axes[1].set_ylabel("windows")
    axes[1].set_title("Interior strand symmetry is near identity")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "strand_symmetry_scatter.png")


def plot_size():
    size = pd.read_csv(OUT / "size_dependence.csv")
    rows = []
    for s, g in size.groupby("size"):
        ci = bootstrap_ci(g["pooled_l2"], seed=int(s))
        rows.append((s, g["pooled_l2"].mean(), ci[0], ci[1]))
    rows.sort()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    ax.errorbar(xs, means, yerr=[means - lo, hi - means], marker="o", color="#264653", capsize=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [f"{x//1000} kb" for x in xs])
    ax.set_xlabel("inverted segment size")
    ax.set_ylabel("pooled delta L2")
    ax.set_title("Pooled SAE delta grows with inversion size")
    fig.tight_layout()
    savefig(fig, "inversion_size_dependence.png")


def plot_heatmap():
    heat = np.load(OUT / "breakpoint_feature_heatmap.npy")
    top = pd.read_csv(OUT / "top_breakpoint_features.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(heat, aspect="auto", cmap="magma", interpolation="nearest")
    for pos in (2000 / 8000 * heat.shape[1], 6000 / 8000 * heat.shape[1]):
        ax.axvline(pos, color="white", lw=0.9, ls="--")
    ax.set_yticks(np.arange(len(top)), top["feature"].astype(str), fontsize=7)
    ax.set_xlabel("position bins across 8 kb window")
    ax.set_ylabel("SAE feature id")
    ax.set_title("Top breakpoint-ranked feature traces across the inversion assay")
    fig.colorbar(im, ax=ax, label="mean |delta|")
    fig.tight_layout()
    savefig(fig, "breakpoint_feature_heatmap.png")


def plot_indel():
    size = pd.read_csv(OUT / "size_dependence.csv")
    aim = np.load(OUT / "aim1_indel_norms_matched_size.npz")
    data = [
        size.loc[size["size"] == 1000, "pooled_l2"].to_numpy(),
        size.loc[size["size"] == 2000, "pooled_l2"].to_numpy(),
        size.loc[size["size"] == 4000, "pooled_l2"].to_numpy(),
        aim["DEL_l2"],
        aim["INS_l2"],
    ]
    labels = ["INV 1 kb", "INV 2 kb", "INV 4 kb", "Aim1 DEL\n0.5-8 kb", "Aim1 INS\n0.5-8 kb"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.72)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel("pooled delta L2")
    ax.set_title("Inversion visibility relative to Aim1 deletions and insertions")
    fig.tight_layout()
    savefig(fig, "inversion_vs_indel_delta_distribution.png")


def plot_real():
    real_path = OUT / "real_inversions_matched_controls.csv"
    if not real_path.exists():
        return
    real = pd.read_csv(real_path)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].scatter(real["control_bp_l2"], real["real_bp_l2"], s=25, alpha=0.82, color="#386641")
    lim = float(max(real["control_bp_l2"].max(), real["real_bp_l2"].max()))
    axes[0].plot([0, lim], [0, lim], color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("matched control breakpoint L2")
    axes[0].set_ylabel("real gnomAD INV breakpoint L2")
    axes[0].set_title("Real inversion breakpoint deltas")
    diff = real["real_bp_l2"] - real["control_bp_l2"]
    axes[1].hist(diff, bins=18, color="#6A994E", alpha=0.85)
    axes[1].axvline(0, color="black", lw=0.9)
    axes[1].set_xlabel("real - matched control L2")
    axes[1].set_ylabel("pairs")
    axes[1].set_title("Paired GC/repeat-matched null")
    fig.tight_layout()
    savefig(fig, "real_inversions_matched_controls.png")


def update_summary_and_docs():
    summary = load_json(OUT / "summary.json")
    real_summary_path = OUT / "real_inversions_summary.json"
    if real_summary_path.exists():
        real = load_json(real_summary_path)
        summary["real_inversions"] = {
            "source_status": {
                "attempted": True,
                "source": real["source"],
                "n_clean": real["n_clean"],
                "n_tested": real["n_tested"],
                "geometry_note": "follow-up used BED span=end-start for exact 8 kb windows; original SVLEN-field geometry produced 8001 bp windows for many records",
            },
            **{k: v for k, v in real.items() if k != "source"},
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    bp = summary["delta_profile"]
    strand = summary["strand_symmetry"]
    size = summary["size_dependence"]
    real_block = summary.get("real_inversions", {})
    top_features = ", ".join(map(str, bp["top_features"][:20]))
    doc = f"""# Results: Inversions

This run used Evo2-7B layer 26 and the Goodfire `Evo-2-Layer-26-Mixed` SAE on the Azure A100 box. Dense per-token SAE activations were streamed in 256-token chunks; saved arrays are compact summaries, not fabricated pooled stand-ins.

## Controlled Synthetic Inversions

- Windows: 300 clean hg38 autosomal 8 kb windows, balanced as 150 genic and 150 intergenic.
- Mean GC: {summary['sample_metadata']['mean_gc']:.3f}; mean RepeatMasker overlap: {summary['sample_metadata']['mean_repeat_frac']:.3f}.
- Design: centered inversions of 1, 2, and 4 kb inside the 8 kb window.

The SAE is strongly strand-aware for content. For 4 kb interiors, pooled ref-forward vs alt-revcomp feature vectors have Pearson r = {strand['pearson_mean']:.4f} (95% CI {strand['pearson_ci95'][0]:.4f}, {strand['pearson_ci95'][1]:.4f}); active-union Pearson r = {strand['active_pearson_mean']:.4f} (95% CI {strand['active_pearson_ci95'][0]:.4f}, {strand['active_pearson_ci95'][1]:.4f}). Matched-token cosine is lower at {strand['matched_token_cosine_mean']:.4f}, so exact token-level feature identities are not invariant, but the aggregate content code is nearly reverse-complement symmetric.

Breakpoint localization is mixed, not clean. Mean L1 delta in +/-64-token breakpoint bands is {bp['mean_l1_breakpoint']:.2f}, but the central interior away from breakpoints is higher at {bp['mean_l1_interior_mid']:.2f}; distant flanks are much lower at {bp['mean_l1_flanks']:.2f}. Mechanistically, Evo2/SAE sees the whole reversed interior, not only the two junctions, despite strong aggregate strand symmetry.

Pooled delta L2 grows monotonically with size:

- 1 kb: {size['1000']['pooled_l2_mean']:.4f} (95% CI {size['1000']['pooled_l2_ci95'][0]:.4f}, {size['1000']['pooled_l2_ci95'][1]:.4f})
- 2 kb: {size['2000']['pooled_l2_mean']:.4f} (95% CI {size['2000']['pooled_l2_ci95'][0]:.4f}, {size['2000']['pooled_l2_ci95'][1]:.4f})
- 4 kb: {size['4000']['pooled_l2_mean']:.4f} (95% CI {size['4000']['pooled_l2_ci95'][0]:.4f}, {size['4000']['pooled_l2_ci95'][1]:.4f})

Top breakpoint-ranked SAE features: {top_features}.

## DEL/INS Comparison

Aim1 matched-size pooled delta means:

- DEL, 0.5-8 kb: n={summary['aim1_indel_comparison']['DEL']['n_0p5_to_8kb']}, mean L2 {summary['aim1_indel_comparison']['DEL']['pooled_l2_mean']:.4f}
- INS, 0.5-8 kb: n={summary['aim1_indel_comparison']['INS']['n_0p5_to_8kb']}, mean L2 {summary['aim1_indel_comparison']['INS']['pooled_l2_mean']:.4f}

Synthetic 4 kb inversions ({size['4000']['pooled_l2_mean']:.4f}) are more visible than the Aim1 deletion subset and comparable to the Aim1 insertion subset. The 1 kb inversion mean is below both.

## Real gnomAD v4.1 Inversions

Source: gnomAD v4.1 non-neuro controls SV sites BED, GRCh38 (`{real_block.get('source_status', {}).get('source', 'NA')}`).

The initial real-arm geometry using the `SVLEN` field yielded zero exact 8 kb clean windows because many BED spans differ by one base from `SVLEN`. A follow-up used `end-start` span and tested {real_block.get('n_tested', 0)} clean 0.5-4 kb inversion sites against GC/repeat-matched synthetic-window controls.

- Real breakpoint L2 mean: {real_block.get('real_bp_l2_mean', float('nan')):.4f}
- Matched control breakpoint L2 mean: {real_block.get('control_bp_l2_mean', float('nan')):.4f}
- Paired real-control difference: {real_block.get('paired_diff_mean', float('nan')):.4f} (95% CI {real_block.get('paired_diff_ci95', [float('nan'), float('nan')])[0]:.4f}, {real_block.get('paired_diff_ci95', [float('nan'), float('nan')])[1]:.4f}); sign-flip p={real_block.get('paired_signflip_p', float('nan')):.4f}

There is no evidence here that real gnomAD inversion breakpoints separate from matched composition controls.

## Artifacts

Results are in `results/inversions/`; figures are mirrored in `plots/`:

- `inversion_delta_profile.png`
- `strand_symmetry_scatter.png`
- `inversion_size_dependence.png`
- `breakpoint_feature_heatmap.png`
- `inversion_vs_indel_delta_distribution.png`
- `real_inversions_matched_controls.png`
"""
    DOC.write_text(doc)
    (OUT / "report.md").write_text(doc)


def main():
    PLOTS.mkdir(exist_ok=True)
    update_summary_and_docs()
    summary = load_json(OUT / "summary.json")
    plot_profile(summary)
    plot_strand()
    plot_size()
    plot_heatmap()
    plot_indel()
    plot_real()


if __name__ == "__main__":
    main()
