"""The decode benchmark's size ladder from one image: cropped to 3:2 around its centre, resized (Lanczos) to 2, 6,
12 and 24 MP and encoded by Pillow (libjpeg-turbo) at quality 90 in 4:4:4, 4:2:2, 4:2:0 and 4:2:0 with a restart
marker every MCU row. The published ladder is made from big_building.ppm of the public set (imagecompression.info's
rgb8bit.zip, 7216x5412; bench/public_set.py downloads and checks it), so anyone can rebuild it. Writes ladder.json
next to the files: the source's name, size and SHA-256, the crop, the resampling and the settings (bench.py
--about puts it into its result). Keep the output out of git.

usage: python bench/make_ladder.py <image (JPEG, PPM, PNG ...)> <out dir>
   then: python bench/bench.py out.json 02mp_444=<out dir>/02mp_444.jpg ... --about <out dir>/ladder.json
"""
import hashlib
import json
import os
import sys

from PIL import Image

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
im0 = Image.open(src).convert("RGB")
W0, H0 = im0.size
if W0 * 2 > H0 * 3:                                  # wider than 3:2: crop the width
    w, h = H0 * 3 // 2, H0
else:                                                # taller: crop the height
    w, h = W0, W0 * 2 // 3
box = ((W0 - w) // 2, (H0 - h) // 2, (W0 - w) // 2 + w, (H0 - h) // 2 + h)
im0 = im0.crop(box)
with open(src, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
about = dict(source=os.path.basename(src), source_size=[W0, H0], source_sha256=sha, crop_3_2=list(box),
             resampling="Lanczos (Pillow)", encoder="Pillow (libjpeg-turbo), quality 90", files=[])
for mp, (w, h) in {2: (1728, 1152), 6: (3000, 2000), 12: (4242, 2828), 24: (6000, 4000)}.items():
    im = im0.resize((w, h), Image.LANCZOS)
    for tag, ss, kw in (("444", 0, {}), ("422", 1, {}), ("420", 2, {}), ("420rst", 2, {"restart_marker_rows": 1})):
        p = os.path.join(out, f"{mp:02d}mp_{tag}.jpg")
        im.save(p, "JPEG", quality=90, subsampling=ss, **kw)
        about["files"].append(dict(name=os.path.basename(p), size=[w, h], bytes=os.path.getsize(p)))
        print(p, w, h, os.path.getsize(p))
with open(os.path.join(out, "ladder.json"), "w") as f:
    json.dump(about, f, indent=1)
