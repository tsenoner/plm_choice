import glob
import os
import sys

import h5py

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.h5")))
print("%-36s %9s %6s %7s  %s" % ("file", "keys", "dim", "GB", "cohort"))
names = {542238: "FULL 542,238", 542237: "full-1 542,237",
         540881: "2000-cap 540,881", 526871: "1022-cap 526,871"}
tot = 0.0
for p in files:
    with h5py.File(p, "r") as f:
        n = f.id.get_num_objs()
        dim = f[next(iter(f.keys()))].shape[0]
    gb = os.path.getsize(p) / 2**30
    tot += gb
    print("%-36s %9d %6d %7.2f  %s" % (os.path.basename(p), n, dim, gb, names.get(n, "?? UNEXPECTED")))
print("")
print("%d files, %.1f GB total" % (len(files), tot))
