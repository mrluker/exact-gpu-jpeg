"""Reference decodes with simplejpeg (libjpeg-turbo) for a list of JPEG files, one process per SIMD code path.

usage: python tools/refdecode.py <list.txt> <out.json> [--colorspace RGB] [--fastdct]
--fastdct: libjpeg-turbo's fast integer IDCT (JDCT_IFAST) instead of the accurate one. Writes {path: sha256 of the decoded array (with its shape), or "error"}. Run it with JSIMD_FORCENONE=1 /
JSIMD_FORCESSE2=1 / JSIMD_FORCEAVX2=1 in the environment (see refs() for a helper that does all three).
"""
import hashlib
import json
import os
import subprocess
import sys


def digest(a):
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def refs(paths, workdir, colorspace="RGB", modes=("AVX2", "SSE2", "NONE"), fastdct=False):
    """{mode: {path: digest or 'error'}}, each mode decoded in its own process."""
    os.makedirs(workdir, exist_ok=True)
    lst = os.path.join(workdir, "refs_list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(paths))
    out = {}
    for m in modes:
        env = {k: v for k, v in os.environ.items() if not k.startswith("JSIMD_")}
        if m != "DEFAULT":                # DEFAULT: no override, what libjpeg-turbo picks on this CPU
            env["JSIMD_FORCE" + m] = "1"
        o = os.path.join(workdir, f"refs_{m}{'_fast' if fastdct else ''}.json")
        subprocess.run([sys.executable, __file__, lst, o, "--colorspace", colorspace] + (["--fastdct"] if fastdct else []),
                       env=env, check=True)
        out[m] = json.load(open(o))
    return out


def main():
    import simplejpeg
    lst, outp = sys.argv[1], sys.argv[2]
    cs = sys.argv[sys.argv.index("--colorspace") + 1] if "--colorspace" in sys.argv else "RGB"
    fast = "--fastdct" in sys.argv
    res = {}
    for p in open(lst).read().split("\n"):
        if not p:
            continue
        try:
            a = simplejpeg.decode_jpeg(open(p, "rb").read(), colorspace=cs, fastdct=fast, fastupsample=False,
                                       strict=False)
            res[p] = digest(a)
        except Exception:
            res[p] = "error"
    json.dump(res, open(outp, "w"))


if __name__ == "__main__":
    main()
