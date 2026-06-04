"""Per-feature SAE association diagnostics: improved volcano + QQ + p-value histogram (vs uniform null).
All significance is Benjamini-Hochberg FDR across all 32,768 features.
Run: .venv/bin/python src/plots/perfeat_diagnostics.py
"""
import numpy as np, csv, os
from scipy.stats import rankdata, norm, chi2
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

INK="#1b2330"; MUTED="#6b7480"; NULLC="#9aa3ad"
C={"sweeps":"#3b6ea5","introgression":"#c44e6a"}
plt.rcParams.update({"savefig.dpi":220,"font.size":13,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":INK,"axes.linewidth":1.0,"axes.spines.top":False,
 "axes.spines.right":False,"axes.titleweight":"bold","axes.titlesize":15,"legend.frameon":False,"text.color":INK,
 "xtick.color":INK,"ytick.color":INK,"axes.labelcolor":INK})
os.makedirs("plots/perfeat",exist_ok=True)
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}

def perfeat(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid])
    X=feats[ix]; y=np.array([int(T[i]["y"]) for i in rid]); n1=int(y.sum()); n0=len(y)-n1
    nz=(X!=0).sum(0); R=np.apply_along_axis(rankdata,0,X)
    U1=R[y==1].sum(0)-n1*(n1+1)/2; auc=U1/(n1*n0)
    # tie-corrected variance: SAE features are sparse, so the zeros form one big tie group
    n=n1+n0; z0=(n-nz).astype(float); tie_sum=z0**3-z0   # nonzero activations are ~unique floats
    var=(n1*n0/12.0)*((n+1)-tie_sum/(n*(n-1))); var=np.clip(var,1e-9,None)
    z=(U1-n1*n0/2)/np.sqrt(var); p=2*norm.sf(np.abs(z))
    tested=nz>=10; p[~tested]=np.nan
    pv=p[tested]; rej,q,_,_=multipletests(pv,method="fdr_bh")   # BH-FDR on tested features
    full_q=np.full_like(p,np.nan); full_rej=np.zeros(len(p),bool)
    full_q[tested]=q; full_rej[tested]=rej
    lam=np.median(chi2.isf(np.clip(pv,1e-300,1),1))/chi2.isf(0.5,1)
    return dict(task=task,auc=auc,p=p,q=full_q,rej=full_rej,tested=tested,n_sig=int(rej.sum()),
                n_test=int(tested.sum()),lam=lam,pv=pv)

D={t:perfeat(t) for t in ("sweeps","introgression")}
for t,d in D.items():
    print(f"{t}: tested {d['n_test']}/32768 | BH-FDR<0.05: {d['n_sig']} | genomic inflation lambda={d['lam']:.2f}")

# -------- improved volcanoes (transparency) --------
for t,d in D.items():
    fig,ax=plt.subplots(figsize=(8,6))
    auc,p,rej=d["auc"],d["p"],d["rej"]; ok=d["tested"]
    lp=-np.log10(np.clip(p,1e-300,1))
    ax.scatter(auc[ok&~rej],lp[ok&~rej],s=7,color=NULLC,alpha=0.18,edgecolor="none",rasterized=True)
    ax.scatter(auc[rej],lp[rej],s=30,color=C[t],alpha=0.55,edgecolor=INK,linewidth=0.3,zorder=5,
               label=f"BH-FDR<0.05  (n={d['n_sig']})")
    if rej.sum(): ax.axhline(-np.log10(p[rej].max()),color=C[t],lw=1,ls=":",alpha=.8)
    ax.axvline(0.5,color=INK,lw=1,ls=(0,(4,3)),alpha=.7)
    ax.set_xlabel("per-feature AUC   (0.5 = no association)"); ax.set_ylabel("-log10 p   (Mann-Whitney U)")
    ax.set_title(f"{t.capitalize()} — single SAE-feature association",loc="left")
    ax.legend(loc="upper center"); fig.savefig(f"plots/perfeat/{t}_volcano.png",bbox_inches="tight"); plt.close(fig)

# -------- QQ plot (both tasks) --------
fig,ax=plt.subplots(figsize=(7.6,7.2))
for t,d in D.items():
    pv=np.sort(d["pv"]); m=len(pv)
    exp=-np.log10((np.arange(1,m+1)-0.5)/m); obs=-np.log10(np.clip(pv,1e-300,1))
    ax.scatter(exp,obs,s=9,color=C[t],alpha=0.45,edgecolor="none",rasterized=True,
               label=f"{t}  (λ={d['lam']:.2f}, {d['n_sig']} sig)")
lim=max(ax.get_xlim()[1],ax.get_ylim()[1])
ax.plot([0,lim],[0,lim],color=INK,ls=(0,(4,3)),lw=1.3,label="null (y=x)")
ax.set_xlim(0,lim); ax.set_ylim(0,None)
ax.set_xlabel("expected  -log10 p   (uniform null)"); ax.set_ylabel("observed  -log10 p")
ax.set_title("QQ plot — single-feature association vs null",loc="left")
ax.legend(loc="upper left"); fig.savefig("plots/perfeat/qq_pvalues.png",bbox_inches="tight"); plt.close(fig)

# -------- p-value histogram / density vs uniform null --------
fig,ax=plt.subplots(figsize=(8.4,5.8))
bins=np.linspace(0,1,41)
for t,d in D.items():
    ax.hist(d["pv"],bins=bins,density=True,color=C[t],alpha=0.45,label=f"{t}",edgecolor="white",linewidth=0.4)
ax.axhline(1.0,color=INK,lw=1.6,ls=(0,(5,3)),label="uniform null (density = 1)")
ax.set_xlabel("p-value (Mann-Whitney U, per feature)"); ax.set_ylabel("density")
ax.set_title("p-value distribution vs the uniform null",loc="left")
ax.legend(loc="upper right"); ax.set_xlim(0,1)
fig.text(0.5,-0.02,"A flat profile = no signal (null). A spike near 0 = real enrichment of associated features.",
         ha="center",fontsize=10.5,color=MUTED)
fig.savefig("plots/perfeat/pvalue_hist.png",bbox_inches="tight"); plt.close(fig)
print("wrote: volcano x2, qq_pvalues.png, pvalue_hist.png")
