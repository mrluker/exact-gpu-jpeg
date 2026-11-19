"""Scan a source tree for personal details before a release: user-profile paths, the account and machine names of
the computer the scan runs on, and e-mail addresses.

usage: python tools/privacy_scan.py [root] [--name TEXT ...]

Every file under root (default: this repository) is read as bytes, text and binary alike (result files, notes,
docs, chart sources, images: a PNG can carry a path in its metadata), except the .git folder, virtual
environments and caches. Hits:

* user-profile paths: C:\\Users\\..., C:/Users/..., /Users/..., /home/... (in any spelling of the separators,
  JSON-escaped included);
* this computer's account name and host name, read at run time (they are never written into this file);
  --name adds more texts to look for (another account name, a full name), case-insensitive;
* e-mail addresses, except the commit-trailer address noreply@anthropic.com, the example.com/.org domains and the
  contact addresses inside third-party licence texts (licenses/, THIRD_PARTY_NOTICES.md), kept word for word.

This file is not scanned (it holds the patterns). tests/test_privacy_scan.py runs it on every test run.

Prints each hit as path:line: what, the count at the end, and exits 1 if there is any.
"""
import getpass
import os
import re
import socket
import sys

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", ".mypy_cache", ".ruff_cache",
             "node_modules"}
SKIP_SUFFIX = (".egg-info",)
PROFILE = re.compile(rb"(?i)(?:[a-z]:(?:\\\\|\\|/)+users(?:\\\\|\\|/)+[^\\/\s\"']+|(?<![\w.])/users/[^/\s\"']+|"
                     rb"(?<![\w.])/home/[^/\s\"']+)")
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
EMAIL_OK = {b"noreply@anthropic.com"}
# third-party licence texts are kept word for word, contact addresses included (an organisation's, not a person's
# of this project): e-mail hits there are not counted; every other kind of hit is
LICENCE_TEXTS = ("licenses" + os.sep, "THIRD_PARTY_NOTICES.md")
SELF = os.path.abspath(__file__)          # this file's own examples of the patterns
EMAIL_OK_DOMAINS = (b"example.com", b"example.org", b"example.net")
# not addresses: package pins and decorators such as "torch@2.14", "foo@cuda.jit" do not match EMAIL (no dot-TLD
# after a name made of letters), but some do; these are dropped:
NOT_EMAIL = re.compile(rb"\.(?:png|jpg|jpeg|py|json|md|txt|cu|svg)$", re.I)


def names(extra):
    out = set()
    for v in (os.environ.get("USERNAME"), os.environ.get("USER")):
        if v:
            out.add(v)
    try:
        out.add(getpass.getuser())
    except Exception:
        pass
    for v in (socket.gethostname(), os.environ.get("COMPUTERNAME")):
        if v:
            out.add(v)
    out.update(extra)
    return sorted({n for n in out if len(n) >= 3}, key=str.lower)


def scan(root, extra=()):
    who = names(extra)
    pats = [(re.compile(re.escape(n.encode()), re.I), "account or machine name") for n in who]
    hits = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.endswith(SKIP_SUFFIX)]
        for f in files:
            p = os.path.join(d, f)
            if os.path.abspath(p) == SELF:
                continue
            try:
                b = open(p, "rb").read()
            except OSError:
                continue
            rel = os.path.relpath(p, root)
            for ln, line in enumerate(b.split(b"\n"), 1):
                for m in PROFILE.finditer(line):
                    hits.append((rel, ln, "user-profile path"))
                for rx, what in pats:
                    if rx.search(line):
                        hits.append((rel, ln, what))
                for m in EMAIL.finditer(line):
                    if rel.startswith(LICENCE_TEXTS):
                        continue
                    e = m.group(0)
                    if e.lower() in EMAIL_OK or e.lower().endswith(EMAIL_OK_DOMAINS) or NOT_EMAIL.search(e):
                        continue
                    hits.append((rel, ln, "e-mail address"))
    return hits


def main():
    args = sys.argv[1:]
    extra = []
    while "--name" in args:
        i = args.index("--name")
        extra.append(args[i + 1])
        del args[i:i + 2]
    root = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    hits = scan(root, extra)
    for rel, ln, what in hits:
        print(f"{rel}:{ln}: {what}")       # the location only: the text found is not printed again
    print(f"{len(hits)} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
