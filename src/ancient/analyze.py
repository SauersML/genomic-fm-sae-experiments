"""Ancient-DNA selection: do Evo2-SAE region features predict the selection coefficient
beyond B-statistic / recombination / composition confounds? Run AFTER features.npy is local.
Run: .venv/bin/python src/ancient/analyze.py
"""
import csv, os, numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree
import warnings; warnings.filterwarnings("ignore")

base="data/ancient_selection"
feats=np.load(f"{base}/features.npy"); ids=[l.strip() for l in open(f"{base}/ids.txt")]
rows={r["rsid"]:r for r in csv.DictReader(open(f"{base}/snps_pilot.tsv"),delimiter="\t")}
keep=[i for i,x in enumerate(ids) if x in rows]; ids=[ids[i] for i in keep]; X=feats[keep]
def col(c):
    out=[]
    for i in ids:
        try: out.append(float(rows[i].get(c,"")))
        except: out.append(np.nan)
    return np.array(out)
s=col("selection_coefficient"); chrom=np.array([rows[i]["chrom"] for i in ids])
ybin=col("label_binary")  # NaN for the continuous (non-labeled) SNPs
COVN=["gc","repeat_frac","gene_density","recomb_rate_cm_per_mb","b_statistic","dist_nearest_tss"]
C=np.c_[[col(c) for c in COVN]].T; C=np.nan_to_num(C)
print(f"n={len(ids)}  selected(label=1)={int((ybin==1).sum())} control={int((ybin==0).sum())}  chroms={len(set(chrom))}")

Xs=TruncatedSVD(128,random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(X))
def oof_reg(Xin,y,g):
    gk=GroupKFold(min(5,len(set(g)))); pred=np.zeros(len(y))
    for tr,te in gk.split(Xin,y,g):
        sc=StandardScaler().fit(Xin[tr]); m=Ridge(alpha=10).fit(sc.transform(Xin[tr]),y[tr])
        pred[te]=m.predict(sc.transform(Xin[te]))
    return spearmanr(y,pred)[0]
def resid_oof(Xin,y,g,cov):
    # out-of-fold residualize y and X-rows? residualize y on cov, predict residual from features
    gk=GroupKFold(min(5,len(set(g)))); pred=np.zeros(len(y)); yr=np.zeros(len(y))
    for tr,te in gk.split(Xin,y,g):
        A=np.c_[np.ones(len(tr)),cov[tr]]; b=np.linalg.lstsq(A,y[tr],rcond=None)[0]
        yr[te]=y[te]-np.c_[np.ones(len(te)),cov[te]]@b
        sc=StandardScaler().fit(Xin[tr]); m=Ridge(alpha=10).fit(sc.transform(Xin[tr]),y[tr]-(np.c_[np.ones(len(tr)),cov[tr]]@b))
        pred[te]=m.predict(sc.transform(Xin[te]))
    return spearmanr(yr,pred)[0]

print("\n=== REGRESSION: predict selection coefficient (held out by chromosome) ===")
rho_feat=oof_reg(Xs,s,chrom); rho_cov=oof_reg(C,s,chrom); rho_res=resid_oof(Xs,s,chrom,StandardScaler().fit_transform(C))
print(f"  SAE features      held-out Spearman = {rho_feat:+.3f}")
print(f"  covariate-only    held-out Spearman = {rho_cov:+.3f}   (B-stat/recomb/GC/repeat/gene/TSS)")
print(f"  features residualized on covariates = {rho_res:+.3f}   (signal beyond confounds)")
# permutation null (shuffle s within chrom groups)
rng=np.random.RandomState(0); nullr=[]
for _ in range(200):
    sp=s.copy()
    for c in set(chrom):
        idx=np.where(chrom==c)[0]; sp[idx]=s[idx][rng.permutation(len(idx))]
    nullr.append(oof_reg(Xs,sp,chrom))
p=(np.sum(np.array(nullr)>=rho_feat)+1)/(len(nullr)+1)
print(f"  permutation p (features) = {p:.3f}")

print("\n=== CLASSIFICATION: strongly-selected vs control ===")
def oof_clf(Xin,y,g):
    gk=GroupKFold(min(5,len(set(g)))); sc_=np.zeros(len(y))
    for tr,te in gk.split(Xin,y,g):
        ss=StandardScaler().fit(Xin[tr]); m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(ss.transform(Xin[tr]),y[tr])
        sc_[te]=m.decision_function(ss.transform(Xin[te]))
    return roc_auc_score(y,sc_)
mfit=(ybin==1)|(ybin==0)
auc_f=oof_clf(Xs[mfit],ybin[mfit],chrom[mfit]); auc_c=oof_clf(C[mfit],ybin[mfit],chrom[mfit])
print(f"  SAE features   AUROC={auc_f:.3f}   covariate-only AUROC={auc_c:.3f}")
# matched controls on confounds
pos=np.where(ybin==1)[0]; neg=np.where(ybin==0)[0]
Cz=StandardScaler().fit_transform(C); tree=cKDTree(Cz[neg]); used=set(); mneg=[]
for pi in pos:
    _,idx=tree.query(Cz[pi],k=min(10,len(neg)))
    for j in np.atleast_1d(idx):
        if neg[j] not in used: used.add(neg[j]); mneg.append(neg[j]); break
sel=np.concatenate([pos,np.array(mneg)])
auc_m=oof_clf(Xs[sel],ybin[sel],chrom[sel])
print(f"  matched-control AUROC={auc_m:.3f}  (matched on B-stat/recomb/GC/repeat/gene/DAF-ish)")

os.makedirs("results/ancient_selection",exist_ok=True)
import json
json.dump({"n":len(ids),"reg_features_spearman":rho_feat,"reg_covariate_spearman":rho_cov,
           "reg_residualized_spearman":rho_res,"reg_perm_p":p,"clf_features_auroc":auc_f,
           "clf_covariate_auroc":auc_c,"clf_matched_auroc":auc_m},
          open("results/ancient_selection/summary.json","w"),indent=2)
verdict=("SAE features predict selection coeff BEYOND confounds" if (rho_res>0.05 and p<0.05)
         else "NOT beyond confounds (confounded by B-stat/recombination/composition)")
print("\nVERDICT:",verdict)
print("wrote results/ancient_selection/summary.json")
