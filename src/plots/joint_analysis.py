"""Joint analysis of selection + introgression in SAE feature space.
(1) per-feature effect correlation (shared vs specific), (2) 3-class held-out classifier + confusion,
(3) joint embedding. Run: .venv/bin/python src/plots/joint_analysis.py
"""
import numpy as np, csv, os
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

INK="#1b2330"; MUTED="#6b7480"
COL={"control":"#b8b0a2","selection":"#3b6ea5","introgression":"#c44e6a"}
plt.rcParams.update({"savefig.dpi":220,"font.size":13,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":INK,"axes.spines.top":False,"axes.spines.right":False,
 "axes.titleweight":"bold","axes.titlesize":15,"legend.frameon":False})
os.makedirs("plots/joint",exist_ok=True)
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}

def perfeat(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid])
    X=feats[ix]; y=np.array([int(T[i]["y"]) for i in rid]); n1=int(y.sum()); n0=len(y)-n1
    nz=(X!=0).sum(0); R=np.apply_along_axis(rankdata,0,X)
    U1=R[y==1].sum(0)-n1*(n1+1)/2; auc=U1/(n1*n0)
    n=n1+n0; z0=(n-nz).astype(float); tie=z0**3-z0
    var=np.clip((n1*n0/12.0)*((n+1)-tie/(n*(n-1))),1e-9,None)
    p=2*norm.sf(np.abs((U1-n1*n0/2)/np.sqrt(var))); tested=nz>=10; p[~tested]=np.nan
    q=np.full_like(p,np.nan); rej=np.zeros(len(p),bool)
    rj,qq,_,_=multipletests(p[tested],method="fdr_bh"); q[tested]=qq; rej[tested]=rj
    return auc,rej,tested
auc_s,rej_s,tt_s=perfeat("sweeps"); auc_i,rej_i,tt_i=perfeat("introgression")
both=tt_s&tt_i; es=auc_s-0.5; ei=auc_i-0.5
r=np.corrcoef(es[both],ei[both])[0,1]
sig_s=rej_s&both; sig_i=rej_i&both; sig_both=sig_s&sig_i
print(f"(1) per-feature effect corr (selection vs introgression) r={r:.3f}")
print(f"    FDR-sig: selection {sig_s.sum()}, introgression {sig_i.sum()}, BOTH {sig_both.sum()}")

# ---- FIG1: effect correlation ----
fig,ax=plt.subplots(figsize=(7.6,7.2))
ax.axhline(0,color=MUTED,lw=.8); ax.axvline(0,color=MUTED,lw=.8)
ax.scatter(es[both&~sig_s&~sig_i],ei[both&~sig_s&~sig_i],s=6,color="#c7ccd1",alpha=.20,edgecolor="none",rasterized=True)
ax.scatter(es[sig_s&~sig_i],ei[sig_s&~sig_i],s=16,color=COL["selection"],alpha=.55,edgecolor="none",label=f"selection-only ({(sig_s&~sig_i).sum()})")
ax.scatter(es[sig_i&~sig_s],ei[sig_i&~sig_s],s=16,color=COL["introgression"],alpha=.6,edgecolor="none",label=f"introgression-only ({(sig_i&~sig_s).sum()})")
ax.scatter(es[sig_both],ei[sig_both],s=34,color="#6a4c93",alpha=.85,edgecolor=INK,linewidth=.3,zorder=6,label=f"both ({sig_both.sum()})")
ax.set_xlabel("selection effect  (per-feature AUC − 0.5)"); ax.set_ylabel("introgression effect  (per-feature AUC − 0.5)")
ax.set_title(f"Do the same SAE features drive both?   r = {r:.2f}",loc="left")
ax.legend(loc="upper left"); fig.savefig("plots/joint/effect_corr.png",bbox_inches="tight"); plt.close(fig)

# ---- build 3-class joint dataset ----
def posids(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    return set(i for i in T if T[i]["y"]=="1" and i in idpos), {i:T[i]["chrom"] for i in T if i in idpos}, \
           set(i for i in T if T[i]["y"]=="0" and i in idpos)
ps,chS,cs=posids("sweeps"); pi,chI,ci=posids("introgression")
chrom_of={**chS,**chI}
overlap=ps&pi
lab={};
for i in ps: lab[i]="selection"
for i in pi: lab.setdefault(i,"introgression")
for i in (cs|ci):
    if i not in ps and i not in pi: lab[i]="control"
rid=[i for i in lab if i in idpos]
ix=np.array([idpos[i] for i in rid]); X=feats[ix]
ylab=np.array([lab[i] for i in rid]); grp=np.array([chrom_of.get(i,"chrNA") for i in rid])
classes=["control","selection","introgression"]
print(f"(2) 3-class joint: {dict(zip(*np.unique(ylab,return_counts=True)))} | sweep∩introg overlap regions={len(overlap)}")
Xs=TruncatedSVD(128,random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(X))
yint=np.array([classes.index(c) for c in ylab])
gk=GroupKFold(min(5,len(set(grp)))); P=np.zeros((len(yint),3))
for tr,te in gk.split(Xs,yint,grp):
    sc=StandardScaler().fit(Xs[tr])
    m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xs[tr]),yint[tr])
    P[te]=m.predict_proba(sc.transform(Xs[te]))
for k,c in enumerate(classes):
    yb=(yint==k).astype(int)
    print(f"    one-vs-rest AUROC {c:13s}= {roc_auc_score(yb,P[:,k]):.3f}")
pred=P.argmax(1); cm=confusion_matrix(yint,pred,normalize="true")

# ---- FIG2: confusion ----
fig,ax=plt.subplots(figsize=(6.6,5.8))
im=ax.imshow(cm,cmap="BuPu",vmin=0,vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",
                color="white" if cm[i,j]>0.55 else INK,fontsize=14,fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(classes,rotation=20); ax.set_yticks(range(3)); ax.set_yticklabels(classes)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("3-class joint classifier (held out by chromosome)",loc="left",fontsize=13.5)
fig.colorbar(im,fraction=0.046,pad=0.04,label="row-normalized rate")
fig.savefig("plots/joint/confusion.png",bbox_inches="tight"); plt.close(fig)

# ---- FIG3: joint embedding ----
try:
    import umap; emb=umap.UMAP(n_neighbors=25,min_dist=0.25,random_state=0).fit_transform(Xs); name="UMAP"
except Exception as e:
    from sklearn.decomposition import PCA; emb=PCA(2).fit_transform(Xs); name="PCA"
fig,ax=plt.subplots(figsize=(7.8,6.8))
for c in classes:
    m=ylab==c; ax.scatter(emb[m,0],emb[m,1],s=20,color=COL[c],alpha=.55,edgecolor="none",label=f"{c} ({m.sum()})")
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"{name} 1"); ax.set_ylabel(f"{name} 2")
ax.set_title(f"Joint SAE-feature embedding ({name})",loc="left"); ax.legend(loc="best")
fig.savefig("plots/joint/embedding.png",bbox_inches="tight"); plt.close(fig)
print("wrote: effect_corr.png, confusion.png, embedding.png")
