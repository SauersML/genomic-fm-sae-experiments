"""BCL11A saturation-MPRA: does the Evo2 ref->alt SAE delta predict measured variant effect?
Composition is constant within the element, so any signal here is FUNCTIONAL, not compositional.
Held out by POSITION. Run after data/satmut/BCL11A_job/features.npy is local.
"""
import csv, os, numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
INK="#1f2630"
plt.rcParams.update({"savefig.dpi":230,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#aab0b7","axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})

J="data/satmut/BCL11A_job"
F=np.load(f"{J}/features.npy"); ids=[l.strip() for l in open(f"{J}/ids.txt")]
fi={x:i for i,x in enumerate(ids)}
ref=F[fi["BCL11A_REF"]]
meta=list(csv.DictReader(open("data/satmut/BCL11A_meta.tsv"),delimiter="\t"))
meta=[m for m in meta if m["id"] in fi]
D=np.stack([F[fi[m["id"]]]-ref for m in meta])     # variant deltas
y=np.array([float(m["score"]) for m in meta]); pos=np.array([int(m["pos"]) for m in meta])
print(f"BCL11A: {len(meta)} variants, {len(set(pos))} positions, score range [{y.min():.2f},{y.max():.2f}]")

# 1) delta magnitude vs |effect| (no training)
dmag=np.abs(D).sum(1)
print(f"\n[1] ||delta|| vs |effect|  Spearman = {spearmanr(dmag,np.abs(y))[0]:+.3f}")
print(f"    ||delta|| vs  effect   Spearman = {spearmanr(dmag,y)[0]:+.3f}")

# 2) predict effect from delta, HELD OUT BY POSITION
Xs=TruncatedSVD(min(64,len(meta)-1),random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(D))
gk=GroupKFold(5); pred=np.zeros(len(y))
for tr,te in gk.split(Xs,y,pos):
    m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,11))).fit(Xs[tr],y[tr]); pred[te]=m.predict(Xs[te])
rho=spearmanr(y,pred)[0]; r2=r2_score(y,pred)
print(f"\n[2] predict effect from SAE delta (held out by position): Spearman={rho:+.3f}  R2={r2:+.3f}")
# permutation null (shuffle effect within nothing -> full shuffle, held-out spearman)
rng=np.random.RandomState(0); null=[]
for _ in range(200):
    ys=rng.permutation(y); pr=np.zeros(len(y))
    for tr,te in gk.split(Xs,ys,pos):
        m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,11))).fit(Xs[tr],ys[tr]); pr[te]=m.predict(Xs[te])
    null.append(spearmanr(ys,pr)[0])
p=(np.sum(np.array(null)>=rho)+1)/(len(null)+1); print(f"    permutation p = {p:.3f}")

# 3) per-position tracks
P=sorted(set(pos)); mt=np.array([np.abs(y[pos==pp]).mean() for pp in P]); md=np.array([dmag[pos==pp].mean() for pp in P])
print(f"\n[3] per-position  mean|effect| vs mean||delta||  Spearman = {spearmanr(md,mt)[0]:+.3f}")

os.makedirs("results/satmut",exist_ok=True); os.makedirs("plots",exist_ok=True)
import json; json.dump({"n":len(meta),"npos":len(P),"dmag_vs_abseffect_spearman":spearmanr(dmag,np.abs(y))[0],
    "heldout_spearman":rho,"heldout_r2":r2,"perm_p":p,"perpos_spearman":spearmanr(md,mt)[0]},
    open("results/satmut/bcl11a_summary.json","w"),indent=2)

fig,ax=plt.subplots(1,2,figsize=(13.5,5.6))
fig.suptitle("BCL11A enhancer saturation MPRA — Evo2 SAE ref/alt delta vs measured effect",x=0.02,ha="left",fontsize=14,fontweight="bold")
ax[0].scatter(pred,y,s=10,c="#2a9d8f",alpha=.5,edgecolor="none")
ax[0].set_xlabel("predicted effect (SAE delta, held out by position)"); ax[0].set_ylabel("measured MPRA effect")
ax[0].set_title(f"Held-out prediction:  ρ = {rho:+.2f}  (p = {p:.3f})",loc="left",fontsize=12.5,pad=8)
ax2=ax[1]; ax2b=ax2.twinx()
ax2.plot(P,mt,color="#1f2630",lw=1.3,label="|measured effect|")
ax2b.plot(P,md/md.max(),color="#e76f51",lw=1.1,alpha=.8,label="||SAE delta|| (scaled)")
ax2.set_xlabel("position in BCL11A enhancer (bp)"); ax2.set_ylabel("mean |measured effect|"); ax2b.set_ylabel("mean ||SAE delta|| (scaled)")
ax2.set_title("Functional landscape along the element",loc="left",fontsize=12.5,pad=8)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("plots/bcl11a_satmut.png",bbox_inches="tight",facecolor="white")
print("wrote plots/bcl11a_satmut.png")
