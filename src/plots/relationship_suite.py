"""Joint selection-vs-introgression relationship, many ways. Minimal text, well-designed.
Run: .venv/bin/python src/plots/relationship_suite.py
"""
import numpy as np, csv, os, pandas as pd, gamfit, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata, norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK="#222a35"
plt.rcParams.update({"savefig.dpi":240,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#c8cdd3","axes.linewidth":1.0,
 "axes.spines.top":False,"axes.spines.right":False})
OUT="plots/joint/ways"; os.makedirs(OUT,exist_ok=True)
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}
COV={r["id"]:r for r in csv.DictReader(open("data/aim2_popgen/covariates_extra.tsv"),delimiter="\t")}

def stats(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid]); X=feats[ix]
    y=np.array([int(T[i]["y"]) for i in rid]); n1=y.sum(); n0=len(y)-n1
    nz=(X!=0).sum(0); R=np.apply_along_axis(rankdata,0,X); U1=R[y==1].sum(0)-n1*(n1+1)/2
    eff=U1/(n1*n0)-0.5; n=n1+n0; z0=(n-nz).astype(float); tie=z0**3-z0
    var=np.clip((n1*n0/12.0)*((n+1)-tie/(n*(n-1))),1e-9,None); z=(U1-n1*n0/2)/np.sqrt(var)
    p=2*norm.sf(np.abs(z)); return eff,z,p,(nz>=10)
es,zs,ps,ts=stats("sweeps"); ei,zi,pi,ti=stats("introgression"); b=ts&ti
REP=np.array([float(COV.get(i,{}).get("repeat_frac","nan") or "nan") for i in ids]); mm=~np.isnan(REP)
Xz=(feats[mm]-feats[mm].mean(0))/(feats[mm].std(0)+1e-9); rz=(REP[mm]-REP[mm].mean())/(REP[mm].std()+1e-9)
rep=np.clip((Xz*rz[:,None]).mean(0),-0.28,0.28)
ss_s=np.sign(es)*-np.log10(np.clip(ps,1e-300,1)); ss_i=np.sign(ei)*-np.log10(np.clip(pi,1e-300,1))

def spline(ax,x,y):
    try:
        m=gamfit.fit(pd.DataFrame({"x":x,"y":y}),"y ~ s(x)")
        gx=np.linspace(np.percentile(x,1),np.percentile(x,99),200)
        pr=pd.DataFrame(m.predict(pd.DataFrame({"x":gx}),interval=0.95))
        ax.fill_between(gx,pr["mean_lower"],pr["mean_upper"],color=INK,alpha=0.13,linewidth=0,zorder=4)
        ax.plot(gx,pr["mean"],color=INK,lw=2.4,zorder=5)
    except Exception as e: print("spline miss",e)

def panel(xv,yv,xl,yl,fname,color=True,hexbin=False,line=True):
    fig,ax=plt.subplots(figsize=(7.6,7.2))
    x,y=xv[b],yv[b]; ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    if hexbin:
        hb=ax.hexbin(x,y,gridsize=55,cmap="magma",bins="log",mincnt=1)
    else:
        if color:
            ax.scatter(x,y,c=rep[b][ok],cmap="RdBu_r",vmin=-0.28,vmax=0.28,s=11,alpha=.5,edgecolor="none",rasterized=True)
        else:
            ax.scatter(x,y,color="#3b6ea5",s=11,alpha=.5,edgecolor="none",rasterized=True)
    ax.axhline(0,color="#e7eaee",lw=1,zorder=0); ax.axvline(0,color="#e7eaee",lw=1,zorder=0)
    if line and not hexbin: spline(ax,x,y)
    ax.set_xlabel(xl,fontsize=15); ax.set_ylabel(yl,fontsize=15); ax.tick_params(labelsize=11,color="#c8cdd3")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}",bbox_inches="tight",facecolor="white"); plt.close(fig); print("wrote",fname)

panel(es,ei,"selection effect","introgression effect","1_effect.png")
panel(zs,zi,"selection z  (effect / SE)","introgression z","2_precision_weighted.png")
panel(-np.log10(np.clip(ps,1e-300,1)),-np.log10(np.clip(pi,1e-300,1)),
      "selection  −log₁₀p","introgression  −log₁₀p","3_significance.png",line=False)
panel(ss_s,ss_i,"selection  signed −log₁₀p","introgression  signed −log₁₀p","4_signed_significance.png")
rk_s=np.full(es.shape,np.nan); rk_s[b]=rankdata(es[b]); rk_i=np.full(ei.shape,np.nan); rk_i[b]=rankdata(ei[b])
panel(rk_s,rk_i,"selection effect (rank)","introgression effect (rank)","5_rank_rank.png",color=False,line=False)
panel(es,ei,"selection effect","introgression effect","6_density.png",hexbin=True,line=False)
print("done")
