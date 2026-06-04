"""Compare gamfit spline/basis types on ei ~ s(es). Run: .venv/bin/python src/plots/spline_compare.py"""
import numpy as np, csv, os, pandas as pd, gamfit, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK="#222a35"; ACC="#b5364a"
plt.rcParams.update({"savefig.dpi":230,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#c8cdd3","axes.spines.top":False,"axes.spines.right":False,
 "axes.titleweight":"bold","axes.titlesize":13})
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}
def eff(t):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{t}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid]); X=feats[ix]
    y=np.array([int(T[i]["y"]) for i in rid]); n1=y.sum(); n0=len(y)-n1
    R=np.apply_along_axis(rankdata,0,X); U1=R[y==1].sum(0)-n1*(n1+1)/2
    return U1/(n1*n0)-0.5,(X!=0).sum(0)>=10
es,ts=eff("sweeps"); ei,ti=eff("introgression"); b=ts&ti
x,y=es[b],ei[b]; df=pd.DataFrame({"es":x,"ei":y})
gx=np.linspace(np.percentile(x,0.5),np.percentile(x,99.5),250); gdf=pd.DataFrame({"es":gx})

def fit_first(cands):
    for formula,smooths in cands:
        try:
            m=gamfit.fit(df,formula,smooths=smooths)
            pr=pd.DataFrame(m.predict(gdf,interval=0.95))
            edf=float(m.summary().smooth_terms_frame()["edf"].iloc[0])
            return pr["mean"].to_numpy(),pr["mean_lower"].to_numpy(),pr["mean_upper"].to_numpy(),edf,formula
        except Exception as e:
            print("  miss:",formula,smooths,"->",repr(e)[:90])
    return None

specs={
 "Default (auto)":[("ei ~ s(es)",None)],
 "B-spline (k=15)":[("ei ~ s(es, type=bspline, centers=15)",None),("ei ~ s(es)",{"es":gamfit.BSpline()})],
 "Thin-plate (k=15)":[("ei ~ s(es, type=tps, centers=15)",None),("ei ~ s(es, type=thinplate, centers=15)",None)],
 "Duchon (k=15)":[("ei ~ s(es, type=duchon, centers=15)",None),("ei ~ s(es)",{"es":gamfit.Duchon()})],
 "Matérn ν=1.5":[("ei ~ s(es, type=matern, centers=15)",None),("ei ~ s(es)",{"es":gamfit.Matern(nu=1.5)})],
}
res={}
for name,c in specs.items():
    print("fitting",name); r=fit_first(c)
    if r: res[name]=r; print(f"   ok  edf={r[3]:.1f}  via {r[4]}")

n=len(res); ncol=min(3,n); nrow=int(np.ceil(n/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(5.0*ncol,4.5*nrow),squeeze=False)
for ax,(name,(mean,lo,hi,edf,f)) in zip(axes.ravel(),res.items()):
    ax.scatter(x,y,s=5,color="#c9ccd1",alpha=.20,edgecolor="none",rasterized=True)
    ax.fill_between(gx,lo,hi,color=ACC,alpha=0.22,linewidth=0)
    ax.plot(gx,mean,color=ACC,lw=2.4)
    ax.axhline(0,color="#e4e8ec",lw=1,zorder=0); ax.axvline(0,color="#e4e8ec",lw=1,zorder=0)
    ax.set_title(f"{name}   ·   EDF {edf:.1f}",loc="left")
    ax.set_xlabel("selection effect"); ax.set_ylabel("introgression effect"); ax.set_ylim(-0.05,0.09)
for ax in axes.ravel()[n:]: ax.axis("off")
fig.suptitle("ei ~ s(es): spline-type comparison (95% CI)",x=0.04,ha="left",fontsize=16,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.97]); os.makedirs("plots/joint",exist_ok=True)
fig.savefig("plots/joint/spline_compare.png",bbox_inches="tight")
print("\nEDF summary:",{k:round(v[3],1) for k,v in res.items()}); print("wrote plots/joint/spline_compare.png")
