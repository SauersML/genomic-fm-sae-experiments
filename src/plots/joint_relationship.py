"""Selection vs introgression in SAE space + its mechanism (repeat content).
Run: .venv/bin/python src/plots/joint_relationship.py
"""
import numpy as np, csv, os
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
INK="#1b2330"; MUTED="#6b7480"
plt.rcParams.update({"savefig.dpi":220,"font.size":13,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":INK,"axes.spines.top":False,"axes.spines.right":False,
 "axes.titleweight":"bold","axes.titlesize":14.5,"legend.frameon":False})
os.makedirs("plots/joint",exist_ok=True)
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}
COV={r["id"]:r for r in csv.DictReader(open("data/aim2_popgen/covariates_extra.tsv"),delimiter="\t")}

def perfeat(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid])
    X=feats[ix]; y=np.array([int(T[i]["y"]) for i in rid]); n1=int(y.sum()); n0=len(y)-n1
    nz=(X!=0).sum(0); R=np.apply_along_axis(rankdata,0,X)
    U1=R[y==1].sum(0)-n1*(n1+1)/2; auc=U1/(n1*n0)
    n=n1+n0; z0=(n-nz).astype(float); tie=z0**3-z0
    var=np.clip((n1*n0/12.0)*((n+1)-tie/(n*(n-1))),1e-9,None)
    p=2*norm.sf(np.abs((U1-n1*n0/2)/np.sqrt(var))); tested=nz>=10; p[~tested]=np.nan
    rej=np.zeros(len(p),bool); rj,_,_,_=multipletests(p[tested],method="fdr_bh"); rej[tested]=rj
    return auc-0.5, rej, tested
es,rs,ts=perfeat("sweeps"); ei,ri,ti=perfeat("introgression"); both=ts&ti

REP=np.array([float(COV.get(i,{}).get("repeat_frac","nan") or "nan") for i in ids])
m=~np.isnan(REP)
Xz=(feats[m]-feats[m].mean(0))/(feats[m].std(0)+1e-9); rz=(REP[m]-REP[m].mean())/(REP[m].std()+1e-9)
rep_corr=(Xz*rz[:,None]).mean(0)
r=np.corrcoef(es[both],ei[both])[0,1]; diff=es-ei
rr=np.corrcoef(rep_corr[both],diff[both])[0,1]

fig,axes=plt.subplots(1,2,figsize=(14,6.5))
ax=axes[0]
vmax=0.3; sc=ax.scatter(es[both],ei[both],c=np.clip(rep_corr[both],-vmax,vmax),cmap="coolwarm",
                        s=11,alpha=.55,edgecolor="none",rasterized=True,vmin=-vmax,vmax=vmax)
xx=np.linspace(es[both].min(),es[both].max(),50)
ax.plot(xx,np.polyval(np.polyfit(es[both],ei[both],1),xx),color=INK,lw=2.2)
ax.axhline(0,color=MUTED,lw=.7); ax.axvline(0,color=MUTED,lw=.7)
ax.set_xlabel("selection effect   (per-feature AUC − 0.5)"); ax.set_ylabel("introgression effect   (per-feature AUC − 0.5)")
ax.set_title(f"Anticorrelated across 32,768 SAE features    r = {r:.2f}",loc="left")
cb=fig.colorbar(sc,ax=ax,fraction=0.046,pad=0.03); cb.set_label("feature ↔ repeat content (corr)",fontsize=11)
ax=axes[1]
ax.scatter(rep_corr[both],diff[both],s=10,color="#6a4c93",alpha=.30,edgecolor="none",rasterized=True)
xx=np.linspace(rep_corr[both].min(),rep_corr[both].max(),50)
ax.plot(xx,np.polyval(np.polyfit(rep_corr[both],diff[both],1),xx),color=INK,lw=2.2)
ax.axhline(0,color=MUTED,lw=.7); ax.axvline(0,color=MUTED,lw=.7)
ax.set_xlabel("feature ↔ repeat content (corr)"); ax.set_ylabel("selection − introgression effect")
ax.set_title(f"Repeat-tracking features drive the split    r = {rr:.2f}",loc="left")
fig.suptitle("Selection vs introgression sit at opposite ends of the SAE's repeat-content axis",
             x=0.06,ha="left",fontsize=16.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("plots/joint/relationship.png"); plt.close(fig)
print(f"r(sel,intro)={r:.3f} | r(repeat-tracking, sel-intro)={rr:.3f}"); print("wrote plots/joint/relationship.png")
