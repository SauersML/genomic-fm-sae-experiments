"""Repeat-family mediation of the selection<->introgression anticorrelation.
Calibrated REGION-level permutation null (shuffle region->repeat-content; preserves feature
correlation) + region bootstrap CI; Benjamini-Hochberg q-values across families. Minimal design.
Run: .venv/bin/python src/plots/repeat_family_mediation.py
"""
import gzip, json, csv, os, numpy as np
from collections import defaultdict
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
INK="#1f2630"; MED="#2a9d8f"; SUP="#e76f51"; NS="#c4cad0"
plt.rcParams.update({"savefig.dpi":240,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#aab0b7","axes.spines.top":False,"axes.spines.right":False,
 "axes.spines.left":False,"ytick.left":False})
feats=np.load("data/aim2_popgen/features.npy"); ids=[l.strip() for l in open("data/aim2_popgen/ids.txt")]
idpos={x:i for i,x in enumerate(ids)}
win={json.loads(l)["id"]:json.loads(l) for l in open("data/aim2_popgen/manifest_w8.jsonl")}
def eff(task):
    T={r["id"]:r for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t")}
    rid=[i for i in T if i in idpos]; ix=np.array([idpos[i] for i in rid]); X=feats[ix]
    y=np.array([int(T[i]["y"]) for i in rid]); n1=y.sum(); n0=len(y)-n1
    R=np.apply_along_axis(rankdata,0,X); U1=R[y==1].sum(0)-n1*(n1+1)/2
    return U1/(n1*n0)-0.5,(X!=0).sum(0)>=10
es,ts=eff("sweeps"); ei,ti=eff("introgression"); both=ts&ti
def cat(cls,fam):
    return {("SINE","Alu"):"Alu",("SINE","MIR"):"MIR",("LINE","L1"):"L1",("LINE","L2"):"L2"}.get((cls,fam),
        {"Simple_repeat":"STR","Low_complexity":"Low-complexity","Satellite":"Satellite","LTR":"LTR"}.get(cls))
CATS=["Alu","MIR","L1","L2","STR","Low-complexity","Satellite","LTR"]
reg_idx=[i for i,x in enumerate(ids) if x in win]; reg_ids=[ids[i] for i in reg_idx]; nreg=len(reg_ids)
chroms=set(win[x]["chrom"] for x in reg_ids); iv=defaultdict(lambda: defaultdict(list))
with gzip.open("data/annotations/rmsk.txt.gz","rt") as fh:
    for line in fh:
        f=line.split("\t"); ch=f[5]
        if ch not in chroms: continue
        k=cat(f[11],f[12])
        if k: iv[ch][k].append((int(f[6]),int(f[7])))
arr={}
for c in iv:
    for k in iv[c]:
        a=np.array(sorted(iv[c][k])); arr[(c,k)]=(a[:,0],a[:,1])
def cov(ch,ws,we,k):
    if (ch,k) not in arr: return 0.0
    st,en=arr[(ch,k)]; lo=np.searchsorted(en,ws,"right"); hi=np.searchsorted(st,we,"left")
    return 0.0 if hi<=lo else float(np.clip(np.minimum(en[lo:hi],we)-np.maximum(st[lo:hi],ws),0,None).sum())/(we-ws)
M=np.array([[cov(win[x]["chrom"],win[x]["start0"],win[x]["end0"],k) for k in CATS] for x in reg_ids])
chrom=np.array([win[x]["chrom"] for x in reg_ids])
Freg=feats[reg_idx]
Zf=(Freg-Freg.mean(0))/(Freg.std(0)+1e-9); Zfb=Zf[:,both]      # nreg x nfeat_both
Zm=(M-M.mean(0))/(M.std(0)+1e-9)
x=es[both]; y=ei[both]; raw=np.corrcoef(x,y)[0,1]
def att_from_RA(RAb):   # RAb: nfeat x ncat -> per-cat % attenuation (vectorized residualize over features)
    out=[]
    for j in range(RAb.shape[1]):
        A=np.c_[np.ones(len(x)),RAb[:,j]]
        rx=x-A@np.linalg.lstsq(A,x,rcond=None)[0]; ry=y-A@np.linalg.lstsq(A,y,rcond=None)[0]
        out.append(100*(abs(raw)-abs(np.corrcoef(rx,ry)[0,1]))/abs(raw))
    return np.array(out)
RAb=(Zfb.T@Zm)/nreg
obs=att_from_RA(RAb)

rng=np.random.RandomState(0)
NP=2000; nullm=np.empty((NP,len(CATS)))
for t in range(NP):                         # region-level permutation: shuffle region->repeat profile
    nullm[t]=att_from_RA((Zfb.T@Zm[rng.permutation(nreg)])/nreg)
zc=(obs-nullm.mean(0))/(nullm.std(0,ddof=1)+1e-12)
rawp=2*norm.sf(np.abs(zc))
qval=multipletests(rawp,method="fdr_bh")[1]
B=400; boot=np.empty((B,len(CATS)))          # region bootstrap CI
for t in range(B):
    s=rng.randint(0,nreg,nreg); Zfb_s=(Freg[s][:,both]-Freg[s][:,both].mean(0))/(Freg[s][:,both].std(0)+1e-9)
    Ms=M[s]; Zm_s=(Ms-Ms.mean(0))/(Ms.std(0)+1e-9); boot[t]=att_from_RA((Zfb_s.T@Zm_s)/nreg)
ci=np.percentile(boot,[2.5,97.5],axis=0)
o=np.argsort(-obs)
print("family  att%   95%CI            BH_q")
for j in o: print(f"{CATS[j]:14s} {obs[j]:+6.1f}  [{ci[0,j]:+.1f},{ci[1,j]:+.1f}]   q={qval[j]:.2e}")

def role(j): return "med" if (qval[j]<0.05 and obs[j]>0) else ("sup" if (qval[j]<0.05 and obs[j]<0) else "ns")
def qf(q): return "q<0.001" if q<1e-3 else (f"q={q:.3f}" if q<0.1 else f"q={q:.2f}")
order=sorted(range(len(CATS)),key=lambda j:obs[j])
fig,ax=plt.subplots(figsize=(8.8,5.4))
ax.axvline(0,color="#9aa0a7",lw=1,zorder=1)
colmap={"med":MED,"sup":SUP,"ns":NS}
lo=[obs[j]-ci[0,j] for j in order]; hi=[ci[1,j]-obs[j] for j in order]
ax.barh(range(len(order)),[obs[j] for j in order],xerr=[lo,hi],color=[colmap[role(j)] for j in order],
        edgecolor="white",linewidth=.8,height=.7,zorder=3,error_kw=dict(ecolor="#5b636c",elinewidth=1.1,capsize=3))
for i,j in enumerate(order):
    v=obs[j]; xt=(ci[1,j] if v>=0 else ci[0,j])+(2.0 if v>=0 else -2.0)
    ax.text(xt,i,f"{v:+.0f}%   {qf(qval[j])}",va="center",ha="left" if v>=0 else "right",fontsize=10.5,color=INK)
ax.set_yticks(range(len(order))); ax.set_yticklabels([CATS[j] for j in order],fontsize=12.5)
ax.set_xlim(min(obs)-22,max(obs)+24)
ax.set_xlabel("mediation: % of the −0.40 selection↔introgression anticorrelation removed",fontsize=11)
ax.set_title("Which repeat families drive it",loc="left",fontsize=15,pad=8,color=INK)
ax.legend(handles=[Patch(color=MED,label="mediator"),Patch(color=SUP,label="suppressor"),Patch(color=NS,label="n.s.")],
          loc="lower right",frameon=False,fontsize=10.5,handlelength=1.1)
fig.tight_layout(); os.makedirs("plots/joint",exist_ok=True)
fig.savefig("plots/joint/repeat_family_mediation.png",bbox_inches="tight",facecolor="white")
print("wrote plots/joint/repeat_family_mediation.png  (q = BH-corrected, region-level permutation null)")
