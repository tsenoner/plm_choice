import json
import os
import sys

import h5py

D="/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k"
ids=json.load(open(os.path.expanduser("~/c1_strat/ec_strat_freeze.json")))["ids"]
out=os.path.expanduser("~/c1_strat/slices"); os.makedirs(out, exist_ok=True)
arms=sorted(f[:-3] for f in os.listdir(D) if f.endswith(".h5"))
for a in arms:
    dst=f"{out}/{a}.h5"
    if os.path.exists(dst): continue
    # default driver, not core: one pass over 3,503 datasets is cheap, and the in-RAM
    # core driver would pull the whole 3.5 GB arm into the login node cgroup.
    with h5py.File(f"{D}/{a}.h5","r") as fin, h5py.File(dst+".part","w") as fo:
        miss=[p for p in ids if p not in fin]
        if miss: sys.exit(f"{a}: {len(miss)} cohort ids absent -- PRE-FILTER VIOLATED")
        for p in ids: fo.create_dataset(p, data=fin[p][:], dtype=fin[p].dtype)
    os.replace(dst+".part", dst)
    print(f"  {a:14s} {os.path.getsize(dst)/1e6:7.1f} MB", flush=True)
print("all arms sliced")
