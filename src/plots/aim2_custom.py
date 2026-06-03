"""Custom Experiment-2 figure: the Evo2-SAE 'selection signal' is genomic-context confounding.
Hand-designed, 4 panels. Run: .venv/bin/python src/plots/aim2_custom.py
"""
import json, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial import cKDTree
import warnings; warnings.filterwarnings("ignore")

# ---------- style ----------
INK="#1b2330"; MUTED="#6b7480"; GRID="#e6e9ee"
C_SWEEP="#3b6ea5"; C_CTRL="#b8b0a2"; C_MATCH="#e07a3f"; C_FEAT="#2a9d8f"; C_COV="#9b5de5"
plt.rcParams.update({
    "figure.dpi":130,"savefig.dpi":220,"font.size":11,"font.family":"DejaVu Sans",
    "axes.edgecolor":INK,"axes.linewidth":0.9,"axes.titlesize":12.5,"axes.titleweight":"bold",
    "axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white",
})
def despine(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=3, color=MUTED)

base="data/aim2_popgen"
feats=np.load(f"{base}/features.npy"); ids=[l.strip() for l in open(f"{base}/ids.txt")]
T={r["id"]:r for r in csv.DictReader(open(f"{base}/table_sweeps.tsv"),delimiter="\t")}
COV={r["id"]:r for r in csv.DictReader(open(f"{base}/covariates_extra.tsv"),delimiter="\t")}
gc=np.load(f"{base}/gc.npy")
keep=[i for i,x in enumerate(ids) if x in T]
ids=[ids[i] for i in keep]; X=feats[keep]; gcv=gc[keep]
y=np.array([int(T[i]["y"]) for i in ids]); chrom=np.array([T[i]["chrom"] for i in ids])
def cov(i,c):
    try: return float(COV.get(i,{}).get(c))
    except: return np.nan
RF=np.array([cov(i,"repeat_frac") for i in ids]); GD=np.array([cov(i,"gene_density") for i in ids])
MP=np.array([cov(i,"mappability") for i in ids])
Cmat=np.nan_to_num(np.c_[RF,MP,GD,gcv])

Xs=TruncatedSVD(128,random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(X))
def oof(Xin,yin,groups):
    gk=GroupKFold(min(5,len(set(groups)))); s=np.zeros(len(yin))
    for tr,te in gk.split(Xin,yin,groups):
        sc=StandardScaler().fit(Xin[tr])
        m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(Xin[tr]),yin[tr])
        s[te]=m.decision_function(sc.transform(Xin[te]))
    return s
# unmatched
s_feat=oof(Xs,y,chrom); s_cov=oof(StandardScaler().fit_transform(Cmat),y,chrom)
auc_feat=roc_auc_score(y,s_feat); auc_cov=roc_auc_score(y,s_cov)
# matched controls (NN on standardized composition)
pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
Cz=StandardScaler().fit_transform(Cmat); tree=cKDTree(Cz[neg]); used=set(); mneg=[]
for pi in pos:
    _,idx=tree.query(Cz[pi],k=min(12,len(neg)))
    for j in np.atleast_1d(idx):
        if neg[j] not in used: used.add(neg[j]); mneg.append(neg[j]); break
sel=np.concatenate([pos,np.array(mneg)])
s_match=oof(Xs[sel],y[sel],chrom[sel]); auc_match=roc_auc_score(y[sel],s_match)

# stored CIs
J=json.load(open("results/aim2_popgen/sweeps/results.json"))["separation"]
def ci(blk): v=J[blk]["l2_logreg"]; return v["auroc"],v["auroc_ci95"]
sw_f=ci("features"); sw_c=ci("covariates_only"); sw_r=ci("features_residualized")
JI=json.load(open("results/aim2_popgen/introgression/results.json"))["separation"]
ig_f=(JI["features"]["l2_logreg"]["auroc"],JI["features"]["l2_logreg"]["auroc_ci95"])
ig_r=(JI["features_residualized"]["l2_logreg"]["auroc"],JI["features_residualized"]["l2_logreg"]["auroc_ci95"])

# ============ FIGURE ============
fig=plt.figure(figsize=(13.5,9.6))
gs=fig.add_gridspec(2,2,hspace=0.42,wspace=0.26,left=0.075,right=0.97,top=0.88,bottom=0.08)
fig.suptitle("Experiment 2 — the Evo2-SAE “selective-sweep signal” is genomic-context confounding",
             fontsize=16.5,fontweight="bold",x=0.075,ha="left",y=0.965)
fig.text(0.075,0.915,"Sweep regions sit in genic, repeat-rich DNA; the controls were random. Match on composition and the signal collapses toward chance.",
         fontsize=11.5,color=MUTED,ha="left")

# ---- Panel A: AUROC ladder ----
axA=fig.add_subplot(gs[0,0]); despine(axA)
rows=[("Sweeps · SAE features",sw_f[0],sw_f[1],C_FEAT),
      ("Sweeps · covariate-only",sw_c[0],sw_c[1],C_COV),
      ("Sweeps · residualized",sw_r[0],sw_r[1],"#577590"),
      ("Sweeps · composition-MATCHED",auc_match,None,C_MATCH),
      ("Introgression · SAE features",ig_f[0],ig_f[1],"#8d99ae"),
      ("Introgression · residualized",ig_r[0],ig_r[1],"#adb5bd")]
ypos=np.arange(len(rows))[::-1]
for yp,(lab,a,cii,col) in zip(ypos,rows):
    if cii is not None: axA.plot(cii,[yp,yp],color=col,lw=3,alpha=.45,solid_capstyle="round")
    axA.scatter([a],[yp],s=95,color=col,zorder=5,edgecolor="white",linewidth=1.2)
    axA.text(a, yp+0.28, f"{a:.3f}", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
axA.axvspan(auc_match, sw_f[0], color=C_MATCH, alpha=0.07, zorder=0)
axA.axvline(0.5,color=INK,lw=1.1,ls=(0,(4,3))); axA.text(0.501,-0.05,"chance",rotation=90,va="bottom",ha="left",fontsize=8.5,color=MUTED)
axA.set_yticks(ypos); axA.set_yticklabels([r[0] for r in rows],fontsize=10.3)
axA.set_xlim(0.45,0.72); axA.set_xlabel("held-out AUROC (by-chromosome CV, 95% CI)")
axA.set_title("A · The signal evaporates under matched controls",loc="left")
axA.annotate(f"matched controls\n−{(sw_f[0]-auc_match)*100:.1f} AUROC points",
             xy=((auc_match+sw_f[0])/2, ypos[3]-0.55), ha="center", va="top",
             fontsize=9.2, color=C_MATCH, fontweight="bold")

# ---- Panel B: confound imbalance (grouped means +/- SEM) ----
axB=fig.add_subplot(gs[0,1]); despine(axB)
grp_idx=[("sweeps",pos,C_SWEEP),("random ctrls",neg,C_CTRL),("matched ctrls",np.array(mneg),C_MATCH)]
covs=[("gene density",GD),("repeat fraction",RF)]
w=0.25
def msem(v): v=v[~np.isnan(v)]; return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
for gi,(glab,idx,col) in enumerate(grp_idx):
    xs=np.arange(len(covs))+(gi-1)*w
    means=[msem(cv[idx])[0] for _,cv in covs]; sems=[msem(cv[idx])[1] for _,cv in covs]
    axB.bar(xs,means,width=w,color=col,edgecolor=INK,linewidth=.7,label=glab,
            yerr=sems,error_kw=dict(ecolor=INK,elinewidth=1,capsize=3))
    for x,m in zip(xs,means): axB.text(x,m+0.012,f"{m:.2f}",ha="center",fontsize=8.4,color=INK)
axB.set_xticks(range(len(covs))); axB.set_xticklabels([c[0] for c in covs],fontsize=10.5)
axB.set_ylabel("mean covariate value"); axB.set_ylim(0,0.85)
axB.legend(frameon=False,fontsize=9.2,loc="upper right",ncol=1)
axB.set_title("B · Positives are a biased genomic sample",loc="left")
axB.text(0.0,-0.2,"Sweeps are more genic and repeat-rich than random controls;\nnearest-neighbor matching closes the gap (orange ≈ blue).",transform=axB.transAxes,fontsize=9.3,color=MUTED)

# ---- Panel C: composition explains the discriminative axis ----
axC=fig.add_subplot(gs[1,0]); despine(axC)
svd=TruncatedSVD(6,random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(X))
# R^2 of each svd comp explained by composition covariates
from numpy.linalg import lstsq
Cz1=np.c_[np.ones(len(ids)),StandardScaler().fit_transform(Cmat)]
r2=[]
for k in range(6):
    beta,_,_,_=lstsq(Cz1,svd[:,k],rcond=None); pred=Cz1@beta
    ss=1-np.sum((svd[:,k]-pred)**2)/np.sum((svd[:,k]-svd[:,k].mean())**2); r2.append(max(0,ss))
bars=axC.bar(range(6),r2,color=[C_FEAT if i in (0,1) else "#a8c7bb" for i in range(6)],edgecolor=INK,linewidth=.7,width=.7)
for k,v in enumerate(r2): axC.text(k,v+0.012,f"{v:.2f}",ha="center",fontsize=9.2,color=INK)
axC.set_xticks(range(6)); axC.set_xticklabels([f"SVD {i+1}" for i in range(6)])
axC.set_ylabel("variance explained by\ncomposition (R²)"); axC.set_ylim(0,max(r2)*1.25+0.05)
axC.set_title("C · The top SAE directions ARE composition",loc="left")
axC.text(0.0,-0.2,"GC + repeat + gene-density + mappability explain most of the leading SAE components\n(the very directions that separate sweeps from controls).",transform=axC.transAxes,fontsize=9.3,color=MUTED)

# ---- Panel D: ROC collapse ----
axD=fig.add_subplot(gs[1,1]); despine(axD)
for s_,yy,lab,col,lw in [(s_feat,y,f"SAE features (random ctrls)  AUROC {auc_feat:.3f}",C_FEAT,2.4),
                          (s_cov,y,f"covariate-only  AUROC {auc_cov:.3f}",C_COV,1.8),
                          (s_match,y[sel],f"SAE features (matched ctrls)  AUROC {auc_match:.3f}",C_MATCH,2.4)]:
    fpr,tpr,_=roc_curve(yy,s_); axD.plot(fpr,tpr,color=col,lw=lw,label=lab)
axD.plot([0,1],[0,1],color=INK,ls=(0,(4,3)),lw=1)
axD.set_xlabel("false positive rate"); axD.set_ylabel("true positive rate")
axD.set_title("D · ROC: matched curve falls toward the diagonal",loc="left")
axD.legend(loc="lower right",frameon=False,fontsize=9.0)
axD.set_xlim(-0.01,1.01); axD.set_ylim(-0.01,1.01)

os.makedirs("plots",exist_ok=True)
out="plots/aim2_experiment2_designed.png"
fig.savefig(out,bbox_inches="tight",facecolor="white")
print("wrote",out)
print(f"unmatched AUROC {auc_feat:.3f} | matched {auc_match:.3f} | cov-only {auc_cov:.3f} | n_matched {len(sel)}")
print("SVD R2 by composition:",[round(v,3) for v in r2])
