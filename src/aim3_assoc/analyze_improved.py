"""Aim 3 (improved baselines): does Evo2-SAE haplotype features predict expression
BEYOND a proper cis-genotype eQTL model + genotype PCs (not just ancestry + variant-count)?
Held out by individual. Run: .venv/bin/python src/aim3_assoc/analyze_improved.py
"""
import csv, json, gzip, os, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import make_pipeline
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
INK="#1f2630"
plt.rcParams.update({"savefig.dpi":230,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#aab0b7","axes.spines.top":False,"axes.spines.right":False,
 "axes.titleweight":"bold"})

A="data/aim3_assoc"
F=np.load(f"{A}/features.npy"); ids=[l.strip() for l in open(f"{A}/ids.txt")]
genes={r["locus_id"]:r for r in csv.DictReader(open(f"{A}/genes.tsv"),delimiter="\t")}
out=list(csv.DictReader(open(f"{A}/outcome.tsv"),delimiter="\t"))
# per-individual mean SAE vector per locus
feat={}  # (locus,sample)->vec
fi={x:i for i,x in enumerate(ids)}
for loc in genes:
    g=genes[loc]
    pass
locsamp=set((r["locus_id"],r["sample"]) for r in out)
def saevec(loc,samp):
    a=f"{loc}|{samp}|h1"; b=f"{loc}|{samp}|h2"
    if a in fi and b in fi: return (F[fi[a]]+F[fi[b]])/2
    return None

def parse_vcf_dosage(path, lo, hi, want_samples):
    """return (samples list, pos array, dosage matrix [nsamp x nsnp]) for POS in [lo,hi]."""
    sm=None; POS=[]; D=[]
    with gzip.open(path,"rt") as fh:
        for line in fh:
            if line.startswith("##"): continue
            if line.startswith("#CHROM"):
                sm=line.rstrip("\n").split("\t")[9:]; idx=[i for i,s in enumerate(sm) if s in want_samples]; sub=[sm[i] for i in idx]; continue
            f=line.split("\t"); p=int(f[1])
            if p<lo: continue
            if p>hi: break
            gts=f[9:]; row=[]
            for i in idx:
                g=gts[i].split(":")[0].replace("|","/")
                try: a,b=g.split("/"); row.append((0 if a=="." else int(a))+(0 if b=="." else int(b)))
                except: row.append(np.nan)
            POS.append(p); D.append(row)
    if not D: return sub,np.array([]),np.zeros((len(sub),0))
    return sub,np.array(POS),np.nan_to_num(np.array(D).T)  # samples x snps

# build per-gene tables
samples_all=sorted(set(r["sample"] for r in out))
geno3={}; genoPCsrc=[]; pcsamp=None
expr={}; split={}; pop={}
for r in out:
    expr[(r["locus_id"],r["sample"])]=float(r["expression"]); split[r["sample"]]=r["split"]; pop[r["sample"]]=r["afr_dummy"]
alt={}
for r in csv.DictReader(open(f"{A}/altcount.tsv"),delimiter="\t"): alt[(r["locus_id"],r["sample"])]=float(r["mean_alt"])

results=[]
for loc,g in genes.items():
    tss=int(g["tss"]); vcf=g["vcf"]
    samps=[s for s in samples_all if (loc,s) in expr and saevec(loc,s) is not None]
    sub,pos,Dwide=parse_vcf_dosage(vcf, tss-50000, tss+50000, set(samps))
    if Dwide.shape[1]==0: continue
    order={s:i for i,s in enumerate(sub)}
    keep=[s for s in samps if s in order]; ix=[order[s] for s in keep]
    Dwide=Dwide[ix]                      # keep x snps (±50kb, for PCs)
    m3=(pos>=tss-3000)&(pos<=tss+3000)
    D3=Dwide[:,m3]                        # ±3kb, fair head-to-head vs SAE window
    y=np.array([expr[(loc,s)] for s in keep]); sp=np.array([split[s] for s in keep])
    X=np.stack([saevec(loc,s) for s in keep]); al=np.array([alt[(loc,s)] for s in keep]).reshape(-1,1)
    tr=sp=="train"; te=sp=="test"
    if te.sum()<8 or tr.sum()<8: continue
    Xsae=TruncatedSVD(min(50,len(keep)-1),random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(X))
    PCg=PCA(min(5,D3.shape[1],len(keep)-1),random_state=0).fit_transform(StandardScaler().fit_transform(Dwide)) if Dwide.shape[1]>1 else np.zeros((len(keep),1))
    def ho(Xin):
        if Xin.shape[1]==0: return np.nan
        m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,5,13))).fit(Xin[tr],y[tr])
        return spearmanr(y[te],m.predict(Xin[te]))[0]
    row=dict(gene=g["symbol"],n_test=int(te.sum()),n_snp3=int(D3.shape[1]),
             sae=ho(Xsae), geno3=ho(D3) if D3.shape[1] else np.nan, alt=ho(al), gpc=ho(PCg),
             sae_resid_pc=ho(np.c_[Xsae - np.c_[np.ones(len(keep)),PCg]@np.linalg.lstsq(np.c_[np.ones(len(keep)),PCg],Xsae,rcond=None)[0]]))
    results.append(row); print(f"{g['symbol']:9s} test n={row['n_test']:3d} snps3={row['n_snp3']:4d} | SAE {row['sae']:+.2f}  cis-geno(3kb) {row['geno3']:+.2f}  ALT {row['alt']:+.2f}  genoPC {row['gpc']:+.2f}")

os.makedirs("results/aim3_assoc",exist_ok=True)
json.dump(results,open("results/aim3_assoc/improved_baselines.json","w"),indent=2)

# ---- plot: per-gene held-out Spearman for the 4 models ----
import numpy as np
res=[r for r in results]; res.sort(key=lambda r:-(r["sae"] if r["sae"]==r["sae"] else -9))
genes_=[r["gene"] for r in res]; x=np.arange(len(genes_)); w=0.2
series=[("SAE features","#2a9d8f","sae"),("cis-genotype (±3kb)","#3b6ea5","geno3"),
        ("ALT-burden","#e9c46a","alt"),("genotype PCs","#b8b0a2","gpc")]
fig,ax=plt.subplots(figsize=(max(8,1.1*len(genes_)+3),5.6))
for i,(lab,c,k) in enumerate(series):
    ax.bar(x+(i-1.5)*w,[r[k] for r in res],w,label=lab,color=c,edgecolor=INK,linewidth=.5)
ax.axhline(0,color=INK,lw=1); ax.set_xticks(x); ax.set_xticklabels(genes_,fontsize=11)
ax.set_ylabel("held-out Spearman (predict expression, test individuals)")
ax.set_title("Aim 3 — does Evo2-SAE beat a real cis-genotype eQTL model?",loc="left",fontsize=14)
ax.legend(frameon=False,ncol=2,fontsize=10,loc="upper right")
fig.tight_layout(); fig.savefig("plots/aim3_baselines.png",bbox_inches="tight",facecolor="white")
print("wrote plots/aim3_baselines.png")
PY=None
