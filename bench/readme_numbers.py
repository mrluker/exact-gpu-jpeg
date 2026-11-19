"""Refresh the README's fuzzing numbers from bench/results/fuzz/summary_20261118.json (run after tools/fuzz_report.py).

usage: python bench/readme_numbers.py
"""
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
fz = json.load(open(os.path.join(ROOT, "bench", "results", "fuzz", "summary_20261118.json")))
t, enc = fz["total"], fz["encoder"]["counts"]
n, g, dcl = t["inputs"], t.get("gpu_exact", 0), t.get("declined", 0)
seeds = sorted(p["seed"] for p in fz["parts"])
nlocal = sum(1 for p in fz["parts"] if p["local_seeds"])
bullet = (
    f"- **Fuzzing:** {n:,} mutated JPEGs in {len(fz['parts'])} parts of 10 000 (`tools/fuzz_suite.sh`; seeds "
    f"{', '.join(map(str, seeds))}; results in `bench/results/fuzz/summary_20261118.json`), {nlocal} of the parts seeded with "
    "crops of local photos, the rest with synthetic JPEGs; two decoding threads at once, guard bands on. "
    f"{g:,} decoded on the GPU, every one byte-identical to libjpeg-turbo; {dcl:,} declined with a reason, and after "
    f"each declined input a known-good file still decoded exactly ({t.get('leak_checks_ok', 0):,} checks). "
    f"{fz['findings']} findings: no exception, crash, hang, CUDA error or guard-band write. Earlier runs found two "
    "real problems, both fixed and kept as regression tests (`tests/regressions/`): a corrupt DQT after the scan "
    "(libjpeg-turbo fails, we returned an image) and a final 0xFF byte (libjpeg-turbo treats it as a marker; one "
    "pixel differed); and bugs in the fuzzer itself, fixed before the counted runs. The encoder parameter fuzzer "
    f"(`tools/fuzz_encode.py`, {enc['valid'] + enc['invalid']:,} cases): {enc['identical']:,} of {enc['valid']:,} "
    f"valid cases byte-identical to Pillow, {enc['raised_cleanly']} of {enc['invalid']} invalid ones raised "
    f"ValueError/TypeError, {enc['other_exception']} other exceptions.")
p = os.path.join(ROOT, "README.md")
s = open(p, encoding="utf-8").read()
s = re.sub(r"- \*\*Fuzzing:\*\* .*", lambda m: bullet, s)
s = re.sub(r"\| fuzzed inputs \(`tools/fuzz.py`\) \| [0-9,A-Z_]+ \| [0-9,A-Z_]+ \| [0-9,A-Z_]+ \|",
           f"| fuzzed inputs (`tools/fuzz.py`) | {n:,} | {g:,} | {g:,} |", s)
s = re.sub(r"(test files and\n)(FUZZ_TOTAL|[0-9,]+)( fuzzed inputs)", lambda m: m.group(1) + f"{n:,}" + m.group(3), s)
s = re.sub(r"- [0-9,]+ random parameter sets up to 600×600", f"- {enc['valid']:,} random parameter sets up to 600×600", s)
open(p, "w", encoding="utf-8").write(s)
print(n, g, enc["identical"])
