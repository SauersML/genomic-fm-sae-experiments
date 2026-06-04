"""BCL11A — LOCAL (per-token) SAE delta at the variant vs measured effect.
The proper single-base readout (no mean-pool dilution). Held out by position.
"""
import csv, os, json, numpy as np
from scipy.stats import spearmanr
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
ids=[l.strip() for l in open(f"{J}/local_ids.txt")]
meta_all={m["id"]:m for m in csv.DictReader(open("data/satmut/BCL11A_meta.tsv"),delimiter="\t")}
meta=[meta_all[i] for i in ids]
y=np.array([float(m["score"]) for m in meta]); pos=np.array([int(m["pos"]) for m in meta])

def run(name, D):
    dmag=np.abs(D).sum(1)
    sp_mag=spearmanr(dmag,np.abs(y))[0]
    Xs=TruncatedSVD(min(64,len(meta)-1),random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(D))
    gk=GroupKFold(5); pred=np.zeros(len(y))
    for tr,te in gk.split(Xs,y,pos):
        m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,11))).fit(Xs[tr],y[tr]); pred[te]=m.predict(Xs[te])
    rho=spearmanr(y,pred)[0]; r2=r2_score(y,pred)
    rng=np.random.RandomState(0); null=[]
    for _ in range(200):
        ys=rng.permutation(y); pr=np.zeros(len(y))
        for tr,te in gk.split(Xs,ys,pos):
            mm=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,11))).fit(Xs[tr],ys[tr]); pr[te]=mm.predict(Xs[te])
        null.append(spearmanr(ys,pr)[0])
    p=(np.sum(np.array(null)>=rho)+1)/(len(null)+1)
    print(f"[{name}] ||delta||~|effect| rho={sp_mag:+.3f} | held-out predict rho={rho:+.3f} R2={r2:+.3f} p={p:.3f}")
    return dict(name=name,mag=sp_mag,rho=rho,r2=r2,p=p,pred=pred,dmag=dmag)

R={}
for nm,fn in [("single-token","local_delta.npy"),("local ±16bp","win_delta.npy")]:
    pth=f"{J}/{fn}"
    if os.path.exists(pth): R[nm]=run(nm,np.load(pth))
best=max(R.values(),key=lambda r:r["rho"])
print("BEST:",best["name"],"rho",round(best["rho"],3))
os.makedirs("results/satmut",exist_ok=True)
json.dump({k:{kk:vv for kk,vv in v.items() if kk not in ("pred","dmag")} for k,v in R.items()},
          open("results/satmut/bcl11a_local_summary.json","w"),indent=2)

# plot: best held-out scatter + per-position landscape (effect vs local-delta mag)
P=sorted(set(pos)); mt=np.array([np.abs(y[pos==pp]).mean() for pp in P]); md=np.array([best["dmag"][pos==pp].mean() for pp in P])
land=spearmanr(md,mt)[0]
fig,ax=plt.subplots(1,2,figsize=(13.5,5.6))
fig.suptitle(f"BCL11A saturation MPRA — LOCAL per-token SAE delta ({best['name']})",x=0.02,ha="left",fontsize=14,fontweight="bold")
ax[0].scatter(best["pred"],y,s=10,c="#2a9d8f",alpha=.5,edgecolor="none")
ax[0].set_xlabel("predicted effect (held out by position)"); ax[0].set_ylabel("measured MPRA effect")
ax[0].set_title(f"Held-out prediction:  rho = {best['rho']:+.2f}  (p = {best['p']:.3f})",loc="left",fontsize=12.5,pad=8)
a2=ax[1]; a2b=a2.twinx()
a2.plot(P,mt,color="#1f2630",lw=1.3); a2b.plot(P,md/md.max(),color="#e76f51",lw=1.1,alpha=.85)
a2.set_xlabel("position in BCL11A enhancer (bp)"); a2.set_ylabel("mean |measured effect|"); a2b.set_ylabel("mean ||local SAE delta|| (scaled)")
a2.set_title(f"Functional landscape  (effect vs delta rho={land:+.2f})",loc="left",fontsize=12.5,pad=8)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("plots/bcl11a_satmut_local.png",bbox_inches="tight",facecolor="white")
print("wrote plots/bcl11a_satmut_local.png")
