import glob
import os
import sys

import h5py

names = {542238: "FULL 542,238", 542237: "full-1", 540881: "2000-cap 540,881", 526871: "1022-cap 526,871"}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.h5"))):
    with h5py.File(p, "r") as f:
        n = f.id.get_num_objs()          # group metadata only; no link iteration
    print("%-34s %9d  %-18s %7.2f GB" % (os.path.basename(p), n, names.get(n, "?? UNEXPECTED"),
                                         os.path.getsize(p)/2**30), flush=True)
