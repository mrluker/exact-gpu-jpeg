"""Phone and camera JPEGs on your own disks, found by EXIF make, decoded by this library and compared with
libjpeg-turbo. Counts and camera models only: no file name, path or other EXIF field is written to the result.

usage: python bench/real_photos.py find <list.json> <folder> [<folder> ...]
           headers only: every .jpg/.jpeg whose EXIF make is a phone or camera maker (or whose model names an
           iPhone) goes into list.json as [path, make, model]. The list is yours: keep it outside the repository.
       python bench/real_photos.py check <list.json> <out.json>
           every listed file through a Decoder of its own (the GPU path alone) and through the default
           egj.decode (with the CPU route and fallback), against libjpeg-turbo (simplejpeg, accurate IDCT, fancy
           upsampling); per camera model: files, sampling layouts and the counts identical / different / declined.
"""
import collections
import json
import os
import sys

MAKERS = ("apple", "samsung", "google", "huawei", "xiaomi", "oneplus", "lg", "motorola", "sony", "htc", "oppo", "vivo",
          "nokia")


def find(out, roots):
    from PIL import Image
    found, seen = [], 0
    for r in roots:
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if not d.startswith((".", "$")) and d not in ("node_modules", "AppData", ".venv")]
            for f in fn:
                if not f.lower().endswith((".jpg", ".jpeg")):
                    continue
                p = os.path.join(dp, f)
                seen += 1
                try:
                    with Image.open(p) as im:
                        ex = im.getexif()
                        make, model = (ex.get(271) or "").strip().lower(), (ex.get(272) or "").strip()
                except Exception:                       # noqa: BLE001 - not a readable JPEG: skipped
                    continue
                if make.startswith(MAKERS) or "iphone" in model.lower():
                    found.append((p, make, model))
    json.dump(found, open(out, "w"))
    print("JPEGs scanned", seen, "; listed", len(found), "; models:",
          dict(collections.Counter(m for _, _, m in found).most_common(15)))


def check(lst, out):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import numpy as np
    import simplejpeg
    import exact_gpu_jpeg as egj
    found = json.load(open(lst))
    egj.self_check()
    dec = egj.Decoder()
    groups = collections.defaultdict(collections.Counter)
    layouts = collections.defaultdict(collections.Counter)
    for p, make, model in found:
        g = ("iPhone " + model.replace("iPhone", "").strip()) if make == "apple" else \
            (make.split()[0].capitalize() + " " + model)
        data = open(p, "rb").read()
        c = groups[g]
        c["files"] += 1
        info = egj.inspect(data)
        if info.get("supported"):
            layouts[g][f"{info['subsampling']}{' + restart markers' if info['restart_interval'] else ''}"] += 1
        try:
            ref = simplejpeg.decode_jpeg(data, fastdct=False, fastupsample=False)
        except Exception:                               # noqa: BLE001
            ref = None
            c["libjpeg-turbo raised"] += 1
        img = dec.decode(data)
        if img is None:
            c["GPU path declined"] += 1
        elif ref is not None:
            c["GPU path byte-identical" if np.array_equal(img.get(), ref) else "GPU path DIFFERS"] += 1
        try:
            o = egj.decode(data)
            if ref is not None:
                c["egj.decode byte-identical" if np.array_equal(o.get(), ref) else "egj.decode DIFFERS"] += 1
        except Exception as e:                          # noqa: BLE001
            c[f"egj.decode raised {type(e).__name__}"] += 1
    tot = collections.Counter()
    for c in groups.values():
        tot.update(c)
    res = dict(what="Phone and camera JPEGs found locally by EXIF make, through a Decoder of its own (the GPU path "
                    "alone) and the default egj.decode, against libjpeg-turbo (simplejpeg, accurate IDCT, fancy "
                    "upsampling). Counts and camera models only; the files stay local.",
               library=egj.__version__, libjpeg_turbo=f"simplejpeg {simplejpeg.__version__}",
               groups={g: dict(counts=dict(c), layouts=dict(layouts[g])) for g, c in sorted(groups.items())},
               total=dict(tot))
    print(json.dumps(res, indent=1))
    json.dump(res, open(out, "w"), indent=1)
    return 0 if not any("DIFFERS" in k for k in tot) else 1


if __name__ == "__main__":
    if sys.argv[1] == "find":
        find(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(check(sys.argv[2], sys.argv[3]))
