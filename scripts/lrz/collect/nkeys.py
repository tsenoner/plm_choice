import glob
import os
import sys

import h5py

names = {542238: "FULL 542,238", 542237: "full-1", 540881: "2000-cap 540,881", 526871: "1022-cap 526,871"}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.h5"))):
    with h5py.File(p, "r") as f:
        n = f.id.get_num_objs()          # group metadata only; no link iteration
    print(
        f"{os.path.basename(p):<34} {n:>9d}  {names.get(n, '?? UNEXPECTED'):<18} "
        f"{os.path.getsize(p) / 2**30:>7.2f} GB",
        flush=True,
    )
