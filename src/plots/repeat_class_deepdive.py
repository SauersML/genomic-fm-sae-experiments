"""Which repeat CLASS drives the sweep/introgression SAE signal? Uses UCSC rmsk repClass.
Run: .venv/bin/python src/plots/repeat_class_deepdive.py
"""
import gzip, json, csv, os, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK="#222a35"
plt.rcParams.update({"savefig.dpi":230,"font.family":"DejaVu Sans","figure.facecolor":"white",
 "axes.facecolor":"white","axes.edgecolor":"#c8cdd3","axes.spines.top":False,"axes.spines.right":False,
 "axes.titleweight":"bold"})

# ---- region windows (8kb) + labels ----
win={}
for ln in open("data/aim2_popgen/manifest_w8.jsonl"):
    r=json.loads(ln); win[r["id"]]=(r["chrom"],int(r["start0"]),int(r["end0"]))
lab={}
for task in ("sweeps","introgression"):
    for r in csv.DictReader(open(f"data/aim2_popgen/table_{task}.tsv"),delimiter="\t"):
        if r["id"] in win: lab[r["id"]]=("sweep" if task=="sweeps" else "introg",int(r["y"]))
regions=[(i,)+win[i]+lab[i] for i in win if i in lab]   # id,chrom,s,e,task,y
chroms=set(c for _,c,_,_,_,_ in regions)
print("regions",len(regions),"chroms",len(chroms))

# ---- parse rmsk: chrom -> class -> sorted (start,end) ----
MAJOR=["SINE","LINE","LTR","DNA","Simple_repeat","Low_complexity","Satellite"]
iv=defaultdict(lambda: defaultdict(list))
with gzip.open("data/annotations/rmsk.txt.gz","rt") as fh:
    for line in fh:
        f=line.rstrip("\n").split("\t")
        chrom,s,e,cls=f[5],f[6],f[7],f[11]
        if chrom not in chroms: continue
        cls=cls if cls in MAJOR else ("Simple_repeat" if cls in ("Simple_repeat",) else cls)
        key=cls if cls in MAJOR else "Other"
        iv[chrom][key].append((int(s),int(e)))
arr={}
for c in iv:
    for k in iv[c]:
        a=np.array(sorted(iv[c][k])); arr[(c,k)]=(a[:,0],a[:,1])
CLASSES=MAJOR+["Other"]
def cov(chrom,ws,we,k):
    key=(chrom,k)
    if key not in arr: return 0.0
    st,en=arr[key]; lo=np.searchsorted(en,ws,side="right"); hi=np.searchsorted(st,we,side="left")
    if hi<=lo: return 0.0
    s=np.maximum(st[lo:hi],ws); e=np.minimum(en[lo:hi],we)
    return float(np.clip(e-s,0,None).sum())/(we-ws)

# ---- per-region class coverage ----
ids=[r[0] for r in regions]
M=np.array([[cov(c,s,e,k) for k in CLASSES] for _,c,s,e,_,_ in regions])
task=np.array([r[4] for r in regions]); y=np.array([r[5] for r in regions])
sw=task=="sweep"; ig=task=="introg"
def grp(mask,yy): return M[mask & (y==yy)].mean(0)
groups={"sweep":grp(sw,1),"sweep ctrl":grp(sw,0),"introg":grp(ig,1),"introg ctrl":grp(ig,0)}

print("\nmean coverage fraction by repeat class:")
print("class           sweep  s.ctrl  intro  i.ctrl   | sweep-vs-ctrl AUROC")
aucs={}
for j,k in enumerate(CLASSES):
    a=roc_auc_score(y[sw],M[sw,j]) if M[sw,j].std()>0 else 0.5
    aucs[k]=a
    print(f"{k:15s} {groups['sweep'][j]:.3f}  {groups['sweep ctrl'][j]:.3f}  {groups['introg'][j]:.3f}  {groups['introg ctrl'][j]:.3f}   | {a:.3f}")

# ---- plot ----
fig,axes=plt.subplots(1,2,figsize=(14,6))
x=np.arange(len(CLASSES)); w=0.2
cols={"sweep":"#3b6ea5","sweep ctrl":"#aebfd0","introg":"#c44e6a","introg ctrl":"#e0b6c0"}
for i,(g,col) in enumerate(cols.items()):
    axes[0].bar(x+(i-1.5)*w,groups[g],w,label=g,color=col,edgecolor=INK,linewidth=.5)
axes[0].set_xticks(x); axes[0].set_xticklabels([c.replace("_","\n") for c in CLASSES],fontsize=10)
axes[0].set_ylabel("mean fraction of 8 kb window"); axes[0].legend(frameon=False,fontsize=10)
axes[0].set_title("Repeat-class content by region type",loc="left")
# panel 2: which class separates sweeps from controls
order=sorted(CLASSES,key=lambda k:-abs(aucs[k]-0.5))
axes[1].barh([k for k in order][::-1],[aucs[k]-0.5 for k in order][::-1],
             color=["#3b6ea5" if aucs[k]>0.5 else "#c44e6a" for k in order][::-1],edgecolor=INK,linewidth=.5)
axes[1].axvline(0,color=INK,lw=1); axes[1].set_xlabel("sweep-vs-control AUROC − 0.5  (single repeat class)")
axes[1].set_title("Which repeat class flags a sweep",loc="left")
fig.suptitle("What kind of repeat drives it?",x=0.06,ha="left",fontsize=17,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); os.makedirs("plots/joint",exist_ok=True)
fig.savefig("plots/joint/repeat_classes.png",bbox_inches="tight",facecolor="white")
print("\nwrote plots/joint/repeat_classes.png")
