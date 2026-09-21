"""Walk models/<dataset>/<model>/<target>/<arm>/evaluation_results/*_metrics.txt -> one CSV row each."""
import csv
import os
import re
import sys

ROOT = os.path.expanduser("~/plm_choice/models")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/c1_strat/probe_metrics.csv")
rows, keys = [], []
for dataset in sorted(os.listdir(ROOT)):
    for model in sorted(os.listdir(f"{ROOT}/{dataset}")):
        for target in sorted(os.listdir(f"{ROOT}/{dataset}/{model}")):
            for arm in sorted(os.listdir(f"{ROOT}/{dataset}/{model}/{target}")):
                d = f"{ROOT}/{dataset}/{model}/{target}/{arm}/evaluation_results"
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if not f.endswith("_metrics.txt"):
                        continue
                    r = {"dataset": dataset, "model_type": model, "target": target, "arm": arm, "file": f}
                    m = re.search(r"epoch=(\d+)-step=(\d+)-val_loss=([\d.]+)", f)
                    if m:
                        r["epoch"], r["step"], r["val_loss"] = m.groups()
                    for line in open(f"{d}/{f}"):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            r[k.strip()] = v.strip()
                    rows.append(r)
                    keys += [k for k in r if k not in keys]
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
