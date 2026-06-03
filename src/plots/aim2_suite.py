"""Experiment-2 figure suite: clean, minimal-text PNGs (sweeps + introgression).
Run: .venv/bin/python src/plots/aim2_suite.py
"""
import json, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial import cKDTree
from numpy.linalg import lstsq
import warnings; warnings.filterwarnings("ignore")

INK="#1b2330"; MUTED="#6b7480"
C_FEAT="#2a9d8f"; C_COV="#9b5de5"; C_RES="#577590"; C_MATCH="#e07a3f"
C_SWEEP="#3b6ea5"; C_CTRL="#b8b0a2"; C_INT="#c44e6a"
plt.rcParams.update({
    "figure.dpi":130,"savefig.dpi":220,"font.size":13,"font.family":"DejaVu Sans",
    "axes.edgecolor":INK,"axes.linewidth":1.0,"axes.titlesize":16,"axes.titleweight":"bold",
    "axes.labelsize":13.5,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white",
    "legend.frameon":False,"legend.fontsize":11.5,
})
OUT="plots/aim2"; os.makedirs(OUT,exist_ok=True)

feats=np.load("data/aim2_popgen/features.npy"); ids_all=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
gc_all=np.load("data/aim2_popgen/gc.npy")
COV={r["id"]:r for r in csv.DictReader(open("data/aim2_popgen/covariates_extra.tsv"),delimiter="\t")}
idpos={x:i for i,x in enumerate(ids_all)}
def cov(i,c):
    try: return float(COV.get(i,{}).get(c))
    except: return np.nan

def load(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]
    ix=np.array([idpos[i] for i in rid])
    X=feats[ix]; gcv=gc_all[ix]
    y=np.array([int(T[i]["y"]) for i in rid]); chrom=np.array([T[i]["chrom"] for i in rid])
    RF=np.array([cov(i,"repeat_frac") for i in rid]); GD=np.array([cov(i,"gene_density") for i in rid])
    MP=np.array([cov(i,"mappability") for i in rid])
    Cmat=np.nan_to_num(np.c_[RF,MP,GD,gcv])
    Xs=TruncatedSVD(128,random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(X))
    return dict(X=X,Xs=Xs,y=y,chrom=chrom,RF=RF,GD=GD,gc=gcv,C=Cmat)

def oof(Xin,yin,g):
    gk=GroupKFold(min(5,len(set(g)))); s=np.zeros(len(yin))
    for tr,te in gk.split(Xin,yin,g):
        sc=StandardScaler().fit(Xin[tr])
        m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(Xin[tr]),yin[tr])
        s[te]=m.decision_function(sc.transform(Xin[te]))
    return s

def matched(d):
    y,C=d["y"],d["C"]; pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    Cz=StandardScaler().fit_transform(C); tree=cKDTree(Cz[neg]); used=set(); mneg=[]
    for pi in pos:
        _,idx=tree.query(Cz[pi],k=min(12,len(neg)))
        for j in np.atleast_1d(idx):
            if neg[j] not in used: used.add(neg[j]); mneg.append(neg[j]); break
    return pos,np.array(mneg)

SW=load("sweeps"); IG=load("introgression")
for d in (SW,IG):
    d["s_feat"]=oof(d["Xs"],d["y"],d["chrom"]); d["auc_feat"]=roc_auc_score(d["y"],d["s_feat"])
    d["s_cov"]=oof(StandardScaler().fit_transform(d["C"]),d["y"],d["chrom"]); d["auc_cov"]=roc_auc_score(d["y"],d["s_cov"])
    pos,mneg=matched(d); d["pos"],d["mneg"]=pos,mneg; sel=np.concatenate([pos,mneg])
    d["sel"]=sel; d["s_match"]=oof(d["Xs"][sel],d["y"][sel],d["chrom"][sel]); d["auc_match"]=roc_auc_score(d["y"][sel],d["s_match"])

def ciget(task,blk):
    J=json.load(open(f"results/aim2_popgen/{task}/results.json"))["separation"]
    v=J[blk]["l2_logreg"]; return v["auroc"],tuple(v["auroc_ci95"])

# ---------- FIG 1: AUROC forest ----------
fig,ax=plt.subplots(figsize=(10,6.4))
rows=[]
for task,tag,col0 in [("sweeps","Sweeps",C_SWEEP),("introgression","Introgression",C_INT)]:
    f=ciget(task,"features"); c=ciget(task,"covariates_only"); r=ciget(task,"features_residualized")
    m=(SW if task=="sweeps" else IG)["auc_match"]
    rows += [(f"{tag} · SAE features",f[0],f[1],C_FEAT),
             (f"{tag} · covariate-only",c[0],c[1],C_COV),
             (f"{tag} · residualized",r[0],r[1],C_RES),
             (f"{tag} · composition-matched",m,None,C_MATCH)]
ypos=np.arange(len(rows))[::-1]
for yp,(lab,a,cii,col) in zip(ypos,rows):
    if cii: ax.plot(cii,[yp,yp],color=col,lw=4,alpha=.40,solid_capstyle="round")
    ax.scatter([a],[yp],s=120,color=col,zorder=5,edgecolor="white",linewidth=1.4)
    ax.text(a,yp+0.30,f"{a:.3f}",ha="center",va="bottom",fontsize=11,fontweight="bold")
ax.axvline(0.5,color=INK,lw=1.2,ls=(0,(4,3)))
ax.text(0.502,ypos.min()-0.7,"chance",rotation=90,va="bottom",fontsize=10,color=MUTED)
ax.axhline(3.5,color=GRID if (GRID:="#e6e9ee") else "#eee",lw=10,alpha=.0)
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows],fontsize=12)
ax.set_xlim(0.45,0.70); ax.set_xlabel("held-out AUROC  (by-chromosome CV, 95% CI)")
ax.set_title("Do Evo2-SAE features separate flagged regions from controls?",loc="left",pad=12)
fig.savefig(f"{OUT}/fig1_auroc.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# ---------- FIG 2: composition confound ----------
fig,axes=plt.subplots(1,2,figsize=(12,5.6))
for ax,(d,name,poscol) in zip(axes,[(SW,"Selective sweeps",C_SWEEP),(IG,"Introgression",C_INT)]):
    pos,mneg=d["pos"],d["mneg"]; neg=np.where(d["y"]==0)[0]
    grp=[("positives",pos,poscol),("random ctrl",neg,C_CTRL),("matched ctrl",mneg,C_MATCH)]
    covs=[("gene\ndensity",d["GD"]),("repeat\nfraction",d["RF"])]; w=0.25
    for gi,(gl,idx,col) in enumerate(grp):
        xs=np.arange(len(covs))+(gi-1)*w
        mu=[np.nanmean(cv[idx]) for _,cv in covs]
        se=[np.nanstd(cv[idx],ddof=1)/np.sqrt(np.sum(~np.isnan(cv[idx]))) for _,cv in covs]
        ax.bar(xs,mu,width=w,color=col,edgecolor=INK,linewidth=.7,label=gl,
               yerr=se,error_kw=dict(ecolor=INK,elinewidth=1,capsize=3))
    ax.set_xticks(range(len(covs))); ax.set_xticklabels([c[0] for c in covs])
    ax.set_ylim(0,0.85); ax.set_title(name,loc="left",fontsize=14)
    if ax is axes[0]: ax.set_ylabel("mean covariate value"); ax.legend(loc="upper right")
fig.suptitle("Positives vs controls — genomic-context confounds",x=0.09,ha="left",fontsize=16,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/fig2_composition.png",facecolor="white"); plt.close(fig)

# ---------- FIG 3: SVD = composition ----------
fig,axes=plt.subplots(1,2,figsize=(12.5,5.4),gridspec_kw=dict(width_ratios=[1,1.15]))
d=SW
svd=TruncatedSVD(6,random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(d["X"]))
Cz1=np.c_[np.ones(len(d["y"])),StandardScaler().fit_transform(d["C"])]
r2=[]
for k in range(6):
    b,_,_,_=lstsq(Cz1,svd[:,k],rcond=None); pr=Cz1@b
    r2.append(max(0,1-np.sum((svd[:,k]-pr)**2)/np.sum((svd[:,k]-svd[:,k].mean())**2)))
ax=axes[0]
ax.bar(range(6),r2,color=[C_FEAT if i<2 else "#a8c7bb" for i in range(6)],edgecolor=INK,linewidth=.7,width=.7)
for k,v in enumerate(r2): ax.text(k,v+0.015,f"{v:.2f}",ha="center",fontsize=11)
ax.set_xticks(range(6)); ax.set_xticklabels([f"SVD{i+1}" for i in range(6)])
ax.set_ylim(0,max(r2)*1.25); ax.set_ylabel("R²  (variance explained\nby composition)")
ax.set_title("Top SAE directions are composition",loc="left",fontsize=14)
ax=axes[1]
b,_,_,_=lstsq(Cz1,svd[:,0],rcond=None); pred1=Cz1@b
for lab,mask,col in [("control",d["y"]==0,C_CTRL),("sweep",d["y"]==1,C_SWEEP)]:
    ax.scatter(pred1[mask],svd[mask,0],s=18,color=col,alpha=.6,edgecolor="none",label=lab)
lims=[min(pred1.min(),svd[:,0].min()),max(pred1.max(),svd[:,0].max())]
ax.plot(lims,lims,color=INK,ls=(0,(4,3)),lw=1.1)
ax.set_xlabel("SVD1 predicted from composition\n(GC + repeat + gene density + mappability)")
ax.set_ylabel("actual SAE component 1 (SVD1)")
ax.set_title(f"Composition predicts SAE axis 1   (R² = {r2[0]:.2f})",loc="left",fontsize=14); ax.legend(loc="upper left")
fig.suptitle("Why it separates: the leading SAE axes encode sequence composition",x=0.07,ha="left",fontsize=16,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/fig3_svd_composition.png",facecolor="white"); plt.close(fig)

# ---------- FIG 4 & 5: ROC per task ----------
def roc_fig(d,name,fname,with_match):
    fig,ax=plt.subplots(figsize=(7.2,7))
    series=[(d["s_feat"],d["y"],f"SAE features  (AUROC {d['auc_feat']:.3f})",C_FEAT,2.6),
            (d["s_cov"],d["y"],f"covariate-only  (AUROC {d['auc_cov']:.3f})",C_COV,1.9)]
    if with_match: series.append((d["s_match"],d["y"][d["sel"]],f"SAE features, matched ctrl  (AUROC {d['auc_match']:.3f})",C_MATCH,2.6))
    for s_,yy,lab,col,lw in series:
        fpr,tpr,_=roc_curve(yy,s_); ax.plot(fpr,tpr,color=col,lw=lw,label=lab)
    ax.plot([0,1],[0,1],color=INK,ls=(0,(4,3)),lw=1.1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.01,1.01); ax.set_ylim(-0.01,1.01)
    ax.set_title(name,loc="left",pad=10); ax.legend(loc="lower right")
    fig.savefig(fname,bbox_inches="tight",facecolor="white"); plt.close(fig)
roc_fig(SW,"Selective sweeps — ROC","%s/fig4_roc_sweeps.png"%OUT,True)
roc_fig(IG,"Archaic introgression — ROC (a clean null)","%s/fig5_roc_introgression.png"%OUT,False)

print("wrote:",sorted(os.listdir(OUT)))
print(f"sweeps: feat {SW['auc_feat']:.3f} match {SW['auc_match']:.3f} cov {SW['auc_cov']:.3f}")
print(f"introg: feat {IG['auc_feat']:.3f} match {IG['auc_match']:.3f} cov {IG['auc_cov']:.3f}")
print("sweeps SVD R2:",[round(v,2) for v in r2])
