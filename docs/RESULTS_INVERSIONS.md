# Results: Inversions

This run used Evo2-7B layer 26 and the Goodfire `Evo-2-Layer-26-Mixed` SAE on the Azure A100 box. Dense per-token SAE activations were streamed in 256-token chunks; saved arrays are compact summaries, not fabricated pooled stand-ins.

## Controlled Synthetic Inversions

- Windows: 300 clean hg38 autosomal 8 kb windows, balanced as 150 genic and 150 intergenic.
- Mean GC: 0.404; mean RepeatMasker overlap: 0.498.
- Design: centered inversions of 1, 2, and 4 kb inside the 8 kb window.

The SAE is strongly strand-aware for content. For 4 kb interiors, pooled ref-forward vs alt-revcomp feature vectors have Pearson r = 0.9957 (95% CI 0.9951, 0.9964); active-union Pearson r = 0.9957 (95% CI 0.9949, 0.9963). Matched-token cosine is lower at 0.4947, so exact token-level feature identities are not invariant, but the aggregate content code is nearly reverse-complement symmetric.

Breakpoint localization is mixed, not clean. Mean L1 delta in +/-64-token breakpoint bands is 35.53, but the central interior away from breakpoints is higher at 50.88; distant flanks are much lower at 4.43. Mechanistically, Evo2/SAE sees the whole reversed interior, not only the two junctions, despite strong aggregate strand symmetry.

Pooled delta L2 grows monotonically with size:

- 1 kb: 0.0856 (95% CI 0.0809, 0.0908)
- 2 kb: 0.1332 (95% CI 0.1246, 0.1425)
- 4 kb: 0.2091 (95% CI 0.1950, 0.2240)

Top breakpoint-ranked SAE features: 29844, 17323, 26395, 15809, 13657, 5814, 32710, 24144, 6002, 5388, 100, 8044, 26443, 16151, 26750, 18982, 32713, 6483, 872, 1667.

## DEL/INS Comparison

Aim1 matched-size pooled delta means:

- DEL, 0.5-8 kb: n=102, mean L2 0.1130
- INS, 0.5-8 kb: n=40, mean L2 0.1974

Synthetic 4 kb inversions (0.2091) are more visible than the Aim1 deletion subset and comparable to the Aim1 insertion subset. The 1 kb inversion mean is below both.

## Real gnomAD v4.1 Inversions

Source: gnomAD v4.1 non-neuro controls SV sites BED, GRCh38 (`https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/genome_sv/gnomad.v4.1.sv.non_neuro_controls.sites.bed.gz`).

The initial real-arm geometry using the `SVLEN` field yielded zero exact 8 kb clean windows because many BED spans differ by one base from `SVLEN`. A follow-up used `end-start` span and tested 50 clean 0.5-4 kb inversion sites against GC/repeat-matched synthetic-window controls.

- Real breakpoint L2 mean: 4.0356
- Matched control breakpoint L2 mean: 4.1552
- Paired real-control difference: -0.1196 (95% CI -0.4343, 0.1887); sign-flip p=0.4665

There is no evidence here that real gnomAD inversion breakpoints separate from matched composition controls.

## Artifacts

Results are in `results/inversions/`; figures are mirrored in `plots/`:

- `inversion_delta_profile.png`
- `strand_symmetry_scatter.png`
- `inversion_size_dependence.png`
- `breakpoint_feature_heatmap.png`
- `inversion_vs_indel_delta_distribution.png`
- `real_inversions_matched_controls.png`
