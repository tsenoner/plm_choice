import glob
import os
import sys

import h5py

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.h5")))
print(f"{'file':<36} {'keys':>9} {'dim':>6} {'GB':>7}  cohort")
names = {542238: "FULL 542,238", 542237: "full-1 542,237",
         540881: "2000-cap 540,881", 526871: "1022-cap 526,871"}
tot = 0.0
for p in files:
    with h5py.File(p, "r") as f:
        n = f.id.get_num_objs()
        dim = f[next(iter(f.keys()))].shape[0]
    gb = os.path.getsize(p) / 2**30
    tot += gb
    print(
        f"{os.path.basename(p):<36} {n:>9d} {dim:>6d} {gb:>7.2f}  "
        f"{names.get(n, '?? UNEXPECTED')}"
    )
print("")
print(f"{len(files)} files, {tot:.1f} GB total")
