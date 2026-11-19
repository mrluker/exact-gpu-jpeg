"""The exactness table (bench/results/exactness.json, behind docs/images and the README's exactness chart), from
counts only: runs tools/check_simd.py on the synthetic corpus and the crafted extreme files, tools/check_dir.py on
the folders or lists given, and tools/check_encode.py on a folder of photos, each in its own process, and writes
one row each. No file or folder name is written: rows carry the label given on the command line.

usage: python tools/exactness_table.py <out.json> <work dir>
           [--decode LABEL=PATH[,PATH...] ...]    folders of JPEGs, or @list.txt files (one path a line)
           [--encode LABEL=DIR]                    photos for the encoder check (vs Pillow, 6 settings)
           [--phones LABEL=FILE.json]              a bench/real_photos.py check result (camera models named)
           [--keep LABEL=FILE.json:WHEN]           a row measured earlier, taken from its result file (files,
                                                   gpu_byte_identical, declined), labelled with when / which head
Rows: [label, files, decoded on the GPU, identical, note]. A GPU decode that is not identical is a failure: the
script exits 1 (and so it does when nothing was checked).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    r = subprocess.run([sys.executable, "-W", "ignore"] + args, capture_output=True, text=True, cwd=os.path.join(HERE, ".."))
    if r.returncode != 0:
        raise SystemExit(f"{args[0]} failed ({r.returncode}): {r.stdout[-400:]} {r.stderr[-400:]}")
    return r.stdout


def main():
    out, work = sys.argv[1], sys.argv[2]
    a = sys.argv[3:]
    decode = [a[i + 1] for i, x in enumerate(a) if x == "--decode"]
    encode = [a[i + 1] for i, x in enumerate(a) if x == "--encode"]
    keep = [a[i + 1] for i, x in enumerate(a) if x == "--keep"]
    phones = [a[i + 1] for i, x in enumerate(a) if x == "--phones"]
    os.makedirs(work, exist_ok=True)
    rows, bad = [], 0
    run(["tools/check_simd.py", os.path.join(work, "simd")])
    c = json.load(open(os.path.join(work, "simd", "check_simd.json")))["counts"]
    n = c.get("corpus: references agree", 0) + c.get("corpus: references differ", 0)
    ok = c.get("corpus: GPU equals all three", 0)
    rows.append(["Synthetic corpus (tests/corpus.py)", str(n), str(ok), str(ok),
                 f"same in libjpeg-turbo's AVX2, SSE2 and C paths; {n - ok} declined (progressive, RGB, warned files)"])
    bad += sum(v for k, v in c.items() if k.endswith("GPU DIFFERS"))
    ne = c.get("extreme: references agree", 0) + c.get("extreme: references differ", 0)
    oke = c.get("extreme: GPU equals all three", 0)
    rows.append(["Crafted extreme coefficients", str(ne), str(oke), str(oke),
                 f"all {c.get('extreme: references differ', 0)} where libjpeg-turbo's own code paths disagree were declined"
                 + (f"; {c['extreme: GPU declined (references agree)']} more declined where they agree"
                    if c.get("extreme: GPU declined (references agree)") else "")])
    for spec in decode:
        label, paths = spec.split("=", 1)
        tot = dict(files=0, exact=0, cpu_fallback=0)
        for k, p in enumerate(paths.split(",")):
            j = os.path.join(work, f"dir_{len(rows)}_{k}.json")
            run(["tools/check_dir.py", "--json", j] + (["--list", p[1:]] if p.startswith("@") else [p]))
            cc = json.load(open(j))["counts"]
            for key in tot:
                tot[key] += cc.get(key, 0)
            bad += cc.get("different", 0) + cc.get("gpu_but_reference_failed", 0)
        rows.append([label, str(tot["files"]), str(tot["exact"]), str(tot["exact"]),
                     f"{tot['cpu_fallback']} declined (decoded by the CPU decoder)" if tot["cpu_fallback"] else ""])
    for spec in phones:
        label, f = spec.split("=", 1)
        r = json.load(open(f))
        t = r["total"]
        n, same = t["files"], t.get("GPU path byte-identical", 0)
        both = t.get("egj.decode byte-identical", 0)
        rows.append([label, str(n), str(same + t.get("GPU path DIFFERS", 0)), str(same),
                     f"{', '.join(r['groups'])}; the default call also identical on {both} of {n}"])
        bad += t.get("GPU path DIFFERS", 0) + t.get("egj.decode DIFFERS", 0)
    for spec in keep:
        label, rest = spec.split("=", 1)
        f, when = rest.rsplit(":", 1)
        t = json.load(open(f))["total"]
        rows.append([label, str(t["files"]), str(t["gpu_byte_identical"] + t.get("differ", 0)), str(t["gpu_byte_identical"]),
                     f"measured {when}"])
    for spec in encode:
        label, d = spec.split("=", 1)
        j = os.path.join(work, "encode.json")
        run(["tools/check_encode.py", d, "--json", j])
        r = json.load(open(j))["results"]
        nn, ii = sum(v["n"] for v in r.values()), sum(v["identical"] for v in r.values())
        rows.append([label, str(nn), str(nn), str(ii), "vs Pillow 13.0.0, with the photos' EXIF"])
        bad += nn - ii
    import exact_gpu_jpeg as egj
    res = dict(head=run(["-c", "import subprocess; print(subprocess.run(['git', '-c', 'safe.directory=*', 'log', "
                             "'--oneline', '-1'], capture_output=True, text=True).stdout.strip())"]).strip(),
               library=egj.__version__, rows=rows)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(1 if bad or not rows else 0)


if __name__ == "__main__":
    main()
