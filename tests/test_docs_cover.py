"""Nothing ships undocumented: every public name, keyword, environment variable and user-facing script of the
library appears in README.md or on a page under docs/. CPU-only (the sources are read, not imported), so it runs in CI.

Add a public function, method, keyword, EGJ_* variable or a script under tools/ or bench/ without writing it up,
and this test names what is missing.
"""
import ast
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "src", "exact_gpu_jpeg")
PUBLIC_CLASSES = ("Decoder", "DecodeJob", "Encoder", "EncodeJob", "CoefHandle", "RepeatCache")


def _read(*p):
    return open(os.path.join(*p), encoding="utf-8").read()


def _args(fn):
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs if x.arg not in ("self", "cls")]


def public_surface():
    """{name: where} for everything a user can name."""
    out = {}
    init = ast.parse(_read(SRC, "__init__.py"))
    exported = []
    for node in init.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            exported = [e.value for e in node.value.elts]
    assert exported
    for name in exported:
        out[name] = "__all__"
    for node in init.body:                                   # the module-level functions and their keywords
        if isinstance(node, ast.FunctionDef) and node.name in exported:
            for arg in _args(node):
                out[arg] = f"{node.name}()"
    for mod in sorted(os.listdir(SRC)):
        if not mod.endswith(".py"):
            continue
        text = _read(SRC, mod)
        for var in re.findall(r"EGJ_[A-Z0-9_]+", text):     # environment variables
            out[var] = f"environment ({mod})"
        for node in ast.parse(text).body:
            if isinstance(node, ast.ClassDef) and node.name in PUBLIC_CLASSES:
                for item in node.body:
                    if not isinstance(item, ast.FunctionDef):
                        continue
                    if item.name == "__init__":
                        for arg in _args(item):
                            if node.name in ("Decoder", "Encoder"):          # the jobs are made by the library
                                out[arg] = f"{node.name}()"
                        for sub in ast.walk(item):                           # public attributes set there
                            if (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store)
                                    and getattr(sub.value, "id", "") == "self" and not sub.attr.startswith("_")):
                                out[sub.attr] = f"{node.name}.{sub.attr}"
                    elif not item.name.startswith("_"):
                        out[item.name] = f"{node.name}.{item.name}()"
                        for arg in _args(item):
                            out[arg] = f"{node.name}.{item.name}()"
    for d in ("tools", "bench"):                             # scripts a user runs
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith((".py", ".sh")):
                out[f"{d}/{f}"] = "script"
    return out


def test_every_public_name_is_documented():
    docs = _read(ROOT, "README.md") + "".join("\n" + _read(ROOT, "docs", f) for f in sorted(os.listdir(os.path.join(ROOT, "docs")))
                                              if f.endswith(".md"))
    missing = []
    for name, where in sorted(public_surface().items()):
        pat = re.escape(name) if "/" in name else r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
        if not re.search(pat, docs):
            missing.append(f"{name} ({where})")
    assert not missing, "not in README.md or on a page under docs/: " + ", ".join(missing)


def test_the_surface_is_what_we_think():
    """The collector really sees the API (so the test above cannot pass by finding nothing)."""
    s = public_surface()
    for name in ("decode", "decode_batch", "encode", "Decoder", "Encoder", "upload_ahead", "decode_async",
                 "encode_async", "ready", "dtype", "layout", "scale", "bias", "fast_dct", "fast_upsample",
                 "max_pixels", "canary", "EGJ_CANARY", "EGJ_STREAM_READ", "result", "redone", "tools/fuzz.py",
                 "bench/bench.py"):
        assert name in s, name
    assert len(s) > 70
