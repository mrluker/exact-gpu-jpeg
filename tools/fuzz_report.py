"""Aggregate tools/fuzz_suite.sh results into one JSON (counts, seeds and settings only; no inputs or file names).

usage: python tools/fuzz_report.py <work_dir> <out.json>
"""
import collections
import glob
import json
import os
import sys


def main():
    w, out = sys.argv[1], sys.argv[2]
    parts, total, reasons = [], collections.Counter(), collections.Counter()
    for f in sorted(glob.glob(os.path.join(w, "part_*", "summary.json")), key=lambda p: int(p.split("part_")[1].split(os.sep)[0].split("/")[0])):
        s = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        parts.append(dict(part=name, seed=s["seed"], iterations=s["iterations"], threads=s["threads"],
                          local_seeds=s["seeds_dir"], canary=s["canary"], counts=s["counts"],
                          worst_decode_s=s["worst_decode_s"], wall_s=s["wall_s"]))
        total.update(s["counts"])
        reasons.update(s["declined_reasons"])
    findings = sorted(os.path.basename(p) for p in glob.glob(os.path.join(w, "part_*", "findings", "*.jpg")))
    res = dict(parts=parts, total=dict(total), declined_reasons=dict(reasons.most_common()),
               findings=len(findings), finding_kinds=dict(collections.Counter(f.split("_")[0] for f in findings)))
    for name in ("encode_final.json", "encode.json"):          # the latest encoder fuzz run
        enc = os.path.join(w, name)
        if os.path.exists(enc):
            res["encoder"] = dict(json.load(open(enc)), file=name)
            break
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(dict(total=res["total"], findings=res["findings"], parts=len(parts),
                          encoder=res.get("encoder", {}).get("counts"))))


if __name__ == "__main__":
    main()
