"""Does repeat content MEDIATE the selection<->introgression anticorrelation? Which classes?
Partial-correlation mediation: feature repeat-class affinity as mediator. Bootstrap CIs.
Run: .venv/bin/python src/plots/repeat_mediation.py
"""
import gzip, json, csv, os, numpy as np
from collections import defaultdict
from scipy.stats import rankdata, norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK="#222a35"
plt.rcParams.update({"savefig.dpi":230,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#c8cdd3","axes.spines.top":False,"axes.spines.right":False,
 "axes.titleweight":"bold"})
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}
win={json.loads(l)["id"]:json.loads(l) for l in open("data/aim2_popgen/manifest_w8.jsonl")}

# per-feature effects (tie-corrected MWU) for each task
def eff(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid]); X=feats[ix]
    y=np.array([int(T[i]["y"]) for i in rid]); n1=y.sum(); n0=len(y)-n1
    R=np.apply_along_axis(rankdata,0,X); U1=R[y==1].sum(0)-n1*(n1+1)/2
    return U1/(n1*n0)-0.5,(X!=0).sum(0)>=10
es,ts=eff("sweeps"); ei,ti=eff("introgression"); both=ts&ti

# region class-coverage matrix M (regions aligned to feats row order)
MAJOR=["SINE","LINE","LTR","DNA","Simple_repeat","Low_complexity","Satellite"]
reg_idx=[i for i,x in enumerate(ids) if x in win]; reg_ids=[ids[i] for i in reg_idx]
chroms=set(win[x]["chrom"] for x in reg_ids)
iv=defaultdict(lambda: defaultdict(list))
with gzip.open("data/annotations/rmsk.txt.gz","rt") as fh:
    for line in fh:
        f=line.split("\t"); chrom=f[5]
        if chrom not in chroms: continue
        cls=f[11]; cls=cls if cls in MAJOR else "Other"
        iv[chrom][cls].append((int(f[6]),int(f[7])))
arr={}
for c in iv:
    for k in iv[c]:
        a=np.array(sorted(iv[c][k])); arr[(c,k)]=(a[:,0],a[:,1])
CLS=MAJOR+["Other"]
def cov(ch,ws,we,k):
    key=(ch,k)
    if key not in arr: return 0.0
    st,en=arr[key]; lo=np.searchsorted(en,ws,"right"); hi=np.searchsorted(st,we,"left")
    if hi<=lo: return 0.0
    return float(np.clip(np.minimum(en[lo:hi],we)-np.maximum(st[lo:hi],ws),0,None).sum())/(we-ws)
M=np.array([[cov(win[x]["chrom"],win[x]["start0"],win[x]["end0"],k) for k in CLS] for x in reg_ids])
Freg=feats[reg_idx]  # regions x 32768, aligned to M

# feature repeat-class affinity RA[feature, class] = corr over regions
Zf=(Freg-Freg.mean(0))/(Freg.std(0)+1e-9); Zm=(M-M.mean(0))/(M.std(0)+1e-9)
RA=(Zf.T@Zm)/len(reg_ids)          # 32768 x nclass
RAb=RA[both]; x=es[both]; y=ei[both]

def partial_r(x,y,Z):
    if Z is None or Z.shape[1]==0: return np.corrcoef(x,y)[0,1]
    A=np.c_[np.ones(len(x)),Z]
    bx=np.linalg.lstsq(A,x,rcond=None)[0]; by=np.linalg.lstsq(A,y,rcond=None)[0]
    rx=x-A@bx; ry=y-A@by; return np.corrcoef(rx,ry)[0,1]
raw=np.corrcoef(x,y)[0,1]
full=partial_r(x,y,RAb)
per={k:partial_r(x,y,RAb[:,[j]]) for j,k in enumerate(CLS)}
print(f"raw anticorrelation r = {raw:.3f}")
print(f"controlling ALL repeat-class affinities -> partial r = {full:.3f}   ({100*(abs(raw)-abs(full))/abs(raw):.0f}% of |r| removed)")
print("per-class attenuation (% of |r| removed by controlling that class's affinity):")
att={k:100*(abs(raw)-abs(per[k]))/abs(raw) for k in CLS}
for k in sorted(CLS,key=lambda z:-att[z]): print(f"  {k:15s} partial r={per[k]:+.3f}   removes {att[k]:5.1f}%")

# bootstrap over features
rng=np.random.RandomState(0); B=500; bt_full=[]; bt_att={k:[] for k in CLS}; bt_raw=[]
n=len(x)
for _ in range(B):
    s=rng.randint(0,n,n); xs,ys,Zs=x[s],y[s],RAb[s]
    r0=np.corrcoef(xs,ys)[0,1]; bt_raw.append(r0); bt_full.append(partial_r(xs,ys,Zs))
    for j,k in enumerate(CLS): bt_att[k].append(100*(abs(r0)-abs(partial_r(xs,ys,Zs[:,[j]])))/abs(r0))
ci=lambda a:(np.percentile(a,2.5),np.percentile(a,97.5))
print(f"\nbootstrap: raw r {raw:.3f} CI{tuple(round(v,3) for v in ci(bt_raw))} | full partial {full:.3f} CI{tuple(round(v,3) for v in ci(bt_full))}")

# plot: per-class % mediated with CI
order=sorted(CLS,key=lambda z:att[z])
fig,ax=plt.subplots(figsize=(9,6))
vals=[att[k] for k in order]; los=[att[k]-np.percentile(bt_att[k],2.5) for k in order]; his=[np.percentile(bt_att[k],97.5)-att[k] for k in order]
cols=["#3b6ea5" if (np.percentile(bt_att[k],2.5)>0) else "#c8cdd3" for k in order]
ax.barh(range(len(order)),vals,xerr=[los,his],color=cols,edgecolor=INK,linewidth=.6,error_kw=dict(ecolor=INK,elinewidth=1,capsize=3))
ax.axvline(0,color=INK,lw=1)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set_xlabel("% of selection↔introgression anticorrelation mediated\n(removed by controlling this repeat class's feature-affinity)")
fa=100*(abs(raw)-abs(full))/abs(raw)
ax.set_title(f"What repeat MEDIATES the relationship?   (all classes together: {fa:.0f}%)",loc="left",fontsize=13)
fig.tight_layout(); os.makedirs("plots/joint",exist_ok=True)
fig.savefig("plots/joint/repeat_mediation.png",bbox_inches="tight",facecolor="white")
print("wrote plots/joint/repeat_mediation.png")
