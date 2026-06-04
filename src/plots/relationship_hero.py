"""Headline: selection vs introgression per-feature effects, gamfit spline + 95% CI.
FDR-significant features (selection or introgression) get a thin black outline.
Run: .venv/bin/python src/plots/relationship_hero.py
"""
import numpy as np, csv, os, pandas as pd, gamfit, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK="#222a35"
plt.rcParams.update({"savefig.dpi":240,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#c8cdd3","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False})
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}
COV={r["id"]:r for r in csv.DictReader(open("data/aim2_popgen/covariates_extra.tsv"),delimiter="\t")}
def stat(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid]); X=feats[ix]
    y=np.array([int(T[i]["y"]) for i in rid]); n1=y.sum(); n0=len(y)-n1
    nz=(X!=0).sum(0); R=np.apply_along_axis(rankdata,0,X); U1=R[y==1].sum(0)-n1*(n1+1)/2
    eff=U1/(n1*n0)-0.5; n=n1+n0; z0=(n-nz).astype(float); tie=z0**3-z0
    var=np.clip((n1*n0/12.0)*((n+1)-tie/(n*(n-1))),1e-9,None)
    p=2*norm.sf(np.abs((U1-n1*n0/2)/np.sqrt(var))); tested=nz>=10
    rej=np.zeros(len(p),bool); rj,_,_,_=multipletests(p[tested],method="fdr_bh"); rej[tested]=rj
    return eff,rej,tested
es,rs,ts=stat("sweeps"); ei,ri,ti=stat("introgression"); both=ts&ti
REP=np.array([float(COV.get(i,{}).get("repeat_frac","nan") or "nan") for i in ids]); mm=~np.isnan(REP)
Xz=(feats[mm]-feats[mm].mean(0))/(feats[mm].std(0)+1e-9); rz=(REP[mm]-REP[mm].mean())/(REP[mm].std()+1e-9)
rep=(Xz*rz[:,None]).mean(0)
x,y=es[both],ei[both]; c=np.clip(rep[both],-0.28,0.28); sig=(rs|ri)[both]
print(f"features plotted={both.sum()} | FDR-significant (sel or intro)={int(sig.sum())}")

mdl=gamfit.fit(pd.DataFrame({"es":x,"ei":y}),"ei ~ s(es)")
gx=np.linspace(np.percentile(x,0.5),np.percentile(x,99.5),250)
pr=pd.DataFrame(mdl.predict(pd.DataFrame({"es":gx}),interval=0.95))

fig,ax=plt.subplots(figsize=(8.8,8))
kw=dict(cmap="RdBu_r",vmin=-0.28,vmax=0.28,s=11,alpha=1.0,rasterized=True)
pn=np.random.RandomState(0).permutation(int((~sig).sum()))
sc=ax.scatter(x[~sig][pn],y[~sig][pn],c=c[~sig][pn],edgecolor="none",zorder=3,**kw)             # non-sig
ax.scatter(x[sig],y[sig],c=c[sig],edgecolor="black",linewidth=0.35,zorder=4,**kw)               # FDR-sig outlined
ax.axhline(0,color="#e4e8ec",lw=1,zorder=0); ax.axvline(0,color="#e4e8ec",lw=1,zorder=0)
ax.fill_between(gx,pr["mean_lower"],pr["mean_upper"],color="#cfd5dc",zorder=1,linewidth=0)
ax.plot(gx,pr["mean"],color=INK,lw=2.8,zorder=6)
ax.set_xlabel("selection effect",fontsize=15); ax.set_ylabel("introgression effect",fontsize=15)
ax.tick_params(labelsize=11,color="#c8cdd3")
cb=fig.colorbar(sc,ax=ax,fraction=0.045,pad=0.02,ticks=[-0.28,0.28])
cb.ax.set_yticklabels(["repeat-poor","repeat-rich"],fontsize=11); cb.outline.set_visible(False); cb.ax.tick_params(length=0)
fig.tight_layout(); fig.savefig("plots/joint/relationship.png",bbox_inches="tight",facecolor="white")
print("wrote plots/joint/relationship.png")
