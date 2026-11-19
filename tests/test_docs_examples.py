"""Every Python example in the README and in the docs pages runs as printed, page by page, top to bottom in one
namespace per page, with small stand-ins for the reader's own files and functions. Every relative link resolves, and
every script a page names exists.

A block fenced as ```python torch needs PyTorch and runs only where it is installed. EGJ_DOCS_ROOT points the test
at another copy of the pages (a draft)."""
import glob
import io
import os
import re

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import exact_gpu_jpeg as egj  # noqa: E402
from exact_gpu_jpeg import _route  # noqa: E402

ROOT = os.environ.get("EGJ_DOCS_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAGES = [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
BLOCK = re.compile(r"^```python( torch)?[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
# python blocks per page: a block added to a page must be counted here, so that none can be left unrun by a typo
COUNT = {"README.md": 2, "docs/batches.md": 9, "docs/decode.md": 12, "docs/encode.md": 10,
         "docs/exactness.md": 5, "docs/first-call.md": 8, "docs/getting-started.md": 4, "docs/how-it-works.md": 9,
         "docs/new-gpu.md": 6, "docs/pipelines.md": 5, "docs/reference.md": 10, "docs/repeat.md": 7,
         "docs/small-images.md": 9,
         "docs/troubleshooting.md": 9, "docs/use-cases.md": 7}


def _jpeg(rng, w, h, **kw):
    from PIL import Image
    a = (rng.random((h // 8 + 1, w // 8 + 1, 3)) * 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).resize((w, h), Image.BICUBIC).save(b, "JPEG", quality=90, **kw)
    return b.getvalue()


def _have_torch():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.parametrize("page", PAGES, ids=[os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) for p in PAGES])
def test_every_example_on_the_page_runs(page, tmp_path, monkeypatch):
    blocks = BLOCK.findall(open(page, encoding="utf-8").read())
    if COUNT is not None:
        key = ("docs/" if os.path.basename(os.path.dirname(page)) == "docs" else "") + os.path.basename(page)
        assert len(blocks) == COUNT.get(key, 0), (key, len(blocks))
    if not blocks:
        return
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_route, "settings_path", lambda: str(tmp_path / "settings" / "settings.json"))
    rng = np.random.default_rng(1)
    names = ["photo.jpg", "a.jpg", "b.jpg"] + [f"f{k}.jpg" for k in range(5)]
    for k, n in enumerate(names):
        (tmp_path / n).write_bytes(_jpeg(rng, 320 + 16 * k, 240))
    (tmp_path / "progressive.jpg").write_bytes(_jpeg(rng, 200, 120, progressive=True))
    jpeg_bytes = _jpeg(rng, 96, 64)
    ns = dict(jpeg_bytes=jpeg_bytes, data=jpeg_bytes, icc=None, exif=None, files=[f"f{k}.jpg" for k in range(5)],
              my_kernel=lambda x: x + 1, use=lambda *a, **k: None,
              my_resize=lambda img: (cp.ascontiguousarray(img), cp.ascontiguousarray(img[::2, ::2])))
    torch_ok = _have_torch()
    try:
        for k, (needs_torch, code) in enumerate(blocks):
            if needs_torch and not torch_ok:
                continue
            try:
                exec(compile(code, f"{os.path.basename(page)} block {k + 1}", "exec"), ns)
            except Exception as e:
                raise AssertionError(f"{os.path.basename(page)}, python block {k + 1} failed: {type(e).__name__}: {e}\n{code}") from e
    finally:
        egj.set_cpu_decoder(None)
        cp.cuda.get_current_stream().synchronize()


def test_links_and_named_scripts_exist():
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    for page in PAGES:
        text = open(page, encoding="utf-8").read()
        base = os.path.dirname(page)
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            here = os.path.normpath(os.path.join(base, target))
            there = os.path.normpath(os.path.join(repo if os.path.samefile(base, ROOT) else os.path.join(repo, "docs"), target))
            assert os.path.exists(here) or os.path.exists(there), (os.path.basename(page), target)
        for script in set(re.findall(r"\b((?:bench|tools)/\w+\.(?:py|sh))\b", text)):
            assert os.path.exists(os.path.join(repo, script)), (os.path.basename(page), script)
