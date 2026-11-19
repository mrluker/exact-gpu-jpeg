"""The source tree holds no user-profile path, no account or machine name of the computer running the tests and no
e-mail address (tools/privacy_scan.py). Needs no GPU."""
import importlib.util
import os

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOOL = os.path.join(ROOT, "tools", "privacy_scan.py")


@pytest.mark.skipif(not os.path.exists(TOOL), reason="tools/ is not part of this copy (an installed package)")
def test_no_personal_details_in_the_tree():
    spec = importlib.util.spec_from_file_location("privacy_scan", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    hits = m.scan(ROOT)
    assert not hits, "\n".join(f"{r}:{n}: {w}" for r, n, w in hits[:50])


@pytest.mark.skipif(not os.path.exists(TOOL), reason="tools/ is not part of this copy (an installed package)")
def test_the_scan_finds_what_it_should(tmp_path):
    spec = importlib.util.spec_from_file_location("privacy_scan", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    us, home, at = "Us" + "ers", "ho" + "me", "@"          # built at run time, so this file itself is not a hit
    (tmp_path / "a.json").write_text('{"p": "C:\\\\%s\\\\someone\\\\x", "q": "/%s/someone/y"}\n' % (us, home))
    (tmp_path / "b.md").write_text(f"mail someone{at}somewhere.org; author noreply{at}anthropic.com; a{at}example.com\n")
    (tmp_path / "c.txt").write_text("built on Machine-Name-Xyz\n")
    hits = m.scan(str(tmp_path), extra=["machine-name-xyz"])
    kinds = sorted(w for _, _, w in hits)
    assert kinds == ["account or machine name", "e-mail address", "user-profile path", "user-profile path"], hits
