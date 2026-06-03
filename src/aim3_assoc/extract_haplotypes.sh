#!/usr/bin/env bash
# Extract per-locus phased genotypes for the 449 overlap individuals from the
# 1000G 30x phased panel (GRCh38, NYGC/EBI), one bgzipped VCF per locus.
#
# These per-locus phased VCFs + a GRCh38 reference FASTA are what the GPU/Evo2
# agent will use to reconstruct each individual-haplotype's cis-window sequence
# (see docs/DATA_ASSOC.md "Haplotype reconstruction").
#
# Public access, no credentials (HTTP + remote tabix). CPU/network only.
# Resumable: skips loci whose output already exists & indexes OK.
set -uo pipefail

ASSOC="/Users/user/bio-interp-experiments/data/assoc"
OUT="$ASSOC/haplotypes"
SAMPLES="$ASSOC/samples.txt"
LOCI="$ASSOC/loci.bed"
PANEL_BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
PANEL_TMPL="1kGP_high_coverage_Illumina.CHR.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

mkdir -p "$OUT"
manifest="$ASSOC/haplotypes_manifest.tsv"
echo -e "locus_id\tchr\tstart\tend\tn_records\tvcf" > "$manifest"

n=0
while IFS=$'\t' read -r chrom start end locus_id; do
  n=$((n+1))
  region="${chrom}:$((start+1))-${end}"
  url="$PANEL_BASE/${PANEL_TMPL/CHR/$chrom}"
  safe=$(echo "$locus_id" | tr '|/' '__')
  out_vcf="$OUT/${safe}.vcf.gz"

  if [[ -s "$out_vcf" && -s "${out_vcf}.csi" ]]; then
    rec=$(bcftools index -n "$out_vcf" 2>/dev/null || echo NA)
    echo "[$n] skip (exists) $locus_id  records=$rec"
    echo -e "${locus_id}\t${chrom}\t${start}\t${end}\t${rec}\t${out_vcf}" >> "$manifest"
    continue
  fi

  echo "[$n] $locus_id  $region"
  # Keep biallelic SNV+INDEL, drop spanning-deletion '*' alleles and SVs (Evo2
  # reconstruction here targets SNV/indel substitution into the ref window;
  # SVs left for Aim-1). Keep phased GT only.
  if timeout 600 bcftools view -r "$region" -S "$SAMPLES" --force-samples "$url" 2>/dev/null \
       | bcftools view -m2 -M2 -e 'ALT="*" || INFO/SVTYPE!="."' -Oz -o "$out_vcf" 2>/dev/null; then
    bcftools index -f "$out_vcf" 2>/dev/null
    rec=$(bcftools index -n "$out_vcf" 2>/dev/null || echo NA)
    echo "    -> records=$rec"
    echo -e "${locus_id}\t${chrom}\t${start}\t${end}\t${rec}\t${out_vcf}" >> "$manifest"
  else
    echo "    !! FAILED $locus_id ($region)"
    echo -e "${locus_id}\t${chrom}\t${start}\t${end}\tFAILED\t${out_vcf}" >> "$manifest"
  fi
done < "$LOCI"

echo "[done] manifest: $manifest"
