"""Ancient-DNA Experiment 2 — VARIANT-level test. SAE ref->alt feature DELTA at each SNP
predicts the selection coefficient? Fixed solver (RidgeCV, standardized; robust Spearman).
Run AFTER data/ancient_delta/features.npy is local. .venv/bin/python src/ancient/analyze_delta.py
"""
import csv, os, json, numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")

base="data/ancient_delta"
F=np.load(f"{base}/features.npy"); ids=[l.strip() for l in open(f"{base}/ids.txt")]
pos={x:i for i,x in enumerate(ids)}
rows={r["rsid"]:r for r in csv.DictReader(open("data/ancient_selection/snps_pilot.tsv"),delimiter="\t")}
rsids=[r for r in rows if (r+"#ref") in pos and (r+"#alt") in pos]
delta=np.stack([F[pos[r+"#alt"]]-F[pos[r+"#ref"]] for r in rsids])   # n x 32768
def col(c):
    o=[]
    for r in rsids:
        try:o.append(float(rows[r].get(c,"")))
        except:o.append(np.nan)
    return np.array(o)
s=col("selection_coefficient"); absS=np.abs(s); chrom=np.array([rows[r]["chrom"] for r in rsids])
ybin=col("label_binary")
print(f"SNPs with ref/alt delta: {len(rsids)}  |  mean ||delta||1={np.abs(delta).sum(1).mean():.1f}, nnz/delta~{int((delta!=0).sum(1).mean())}")
Xs=TruncatedSVD(128,random_state=0).fit_transform(StandardScaler(with_std=False).fit_transform(delta))

def heldout(X,y,g,task="reg"):
    gk=GroupKFold(min(5,len(set(g)))); rho=[]; r2=[]; auc=[]
    for tr,te in gk.split(X,y,g):
        if task=="reg":
            m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,5,13))).fit(X[tr],y[tr])
            p=m.predict(X[te]); rho.append(spearmanr(y[te],p)[0]); r2.append(r2_score(y[te],p))
        else:
            from sklearn.linear_model import LogisticRegression
            if len(set(y[tr]))<2 or len(set(y[te]))<2: continue
            m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced",C=0.5)).fit(X[tr],y[tr])
            auc.append(roc_auc_score(y[te],m.decision_function(X[te])))
    if task=="reg": return np.nanmean(rho),np.nanmean(r2)
    return (np.mean(auc) if auc else np.nan), len(auc)

print("\n=== Experiment 2: ref/alt DELTA -> selection coefficient (held out by chromosome) ===")
rho_s,r2_s=heldout(Xs,s,chrom); print(f"  predict signed s : Spearman={rho_s:+.3f}  R2={r2_s:+.3f}")
rho_a,r2_a=heldout(Xs,absS,chrom); print(f"  predict |s|      : Spearman={rho_a:+.3f}  R2={r2_a:+.3f}")
# delta magnitude vs |s|
dmag=np.abs(delta).sum(1)
gk=GroupKFold(5); mrho=[]
for tr,te in gk.split(dmag.reshape(-1,1),absS,chrom): mrho.append(spearmanr(absS[te],dmag[te])[0])
print(f"  ||delta|| vs |s| (held-out Spearman): {np.nanmean(mrho):+.3f}")
# classification selected vs control (folds with both classes only)
m2=np.isin(ybin,[0,1])
auc,nf=heldout(Xs[m2],ybin[m2].astype(int),chrom[m2],task="clf")
print(f"  selected vs control AUROC (folds w/ both classes, n={nf}): {auc if auc==auc else float('nan'):.3f}")

os.makedirs("results/ancient_selection",exist_ok=True)
json.dump({"n":len(rsids),"delta_predict_s_spearman":rho_s,"delta_predict_s_r2":r2_s,
           "delta_predict_absS_spearman":rho_a,"deltamag_vs_absS_spearman":float(np.nanmean(mrho)),
           "delta_clf_auroc":(None if auc!=auc else auc)},
          open("results/ancient_selection/delta_summary.json","w"),indent=2)
verdict=("DELTA carries held-out signal for selection" if (rho_s>0.05 or rho_a>0.05) else
         "NULL: ref/alt delta does not predict selection on held-out chromosomes")
print("\nVERDICT:",verdict)
