"""Runs ON THE BOX. Per-token SAE ref/alt delta for BCL11A saturation variants:
extract the SAE feature change AT the variant position (and a +/-16bp local pool),
instead of mean-pooling over the whole element. Saves local_delta.npy + win_delta.npy.
"""
import json, sys, os, numpy as np, torch
sys.path.insert(0, os.getcwd())   # repo root (run from there); keeps installed `evo2` importable
from src.evo2.extract import embed_dna, sae_features  # per-token

JOB="data/satmut/BCL11A_job"
recs=[json.loads(l) for l in open(f"{JOB}/manifest.jsonl")]
ref=[r for r in recs if r["id"]=="BCL11A_REF"][0]
alts=[r for r in recs if r["id"]!="BCL11A_REF"]
print(f"ref len {len(ref['seq'])}, {len(alts)} variants", flush=True)

@torch.no_grad()
def per_token_sae(seq):
    acts=embed_dna([seq])[0]                 # [L,4096]
    s=sae_features(acts)                      # [L,32768]
    return s.float().cpu().numpy()

ref_sae=per_token_sae(ref["seq"])            # [L,32768]
L=ref_sae.shape[0]; K=16
loc=np.empty((len(alts),ref_sae.shape[1]),dtype=np.float32)
win=np.empty_like(loc); ids=[]
for i,r in enumerate(alts):
    p=int(r["id"].split("_")[1])-1           # 0-based variant position
    s=per_token_sae(r["seq"])
    loc[i]=s[p]-ref_sae[p]
    a0=max(0,p-K); b0=min(L,p+K+1)
    win[i]=(s[a0:b0]-ref_sae[a0:b0]).mean(0)
    ids.append(r["id"])
    if i%200==0: print(f"  {i}/{len(alts)}", flush=True)
np.save(f"{JOB}/local_delta.npy",loc); np.save(f"{JOB}/win_delta.npy",win)
open(f"{JOB}/local_ids.txt","w").write("\n".join(ids))
print("saved local_delta",loc.shape,"win_delta",win.shape, flush=True)
