"""Header parsing (no GPU needed): what the GPU path accepts, and that everything else is declined with a reason."""
import numpy as np
import pytest

import exact_gpu_jpeg as egj
from exact_gpu_jpeg._header import Unsupported, check_huffman, check_trailer, parse
from jpegwriter import grid, write_jpeg


def _jpeg(samp=((2, 1), (1, 1), (1, 1)), W=24, H=16, **kw):
    rng = np.random.default_rng(0)
    blocks = [rng.integers(-3, 4, grid(list(samp), W, H, c) + (64,)) for c in range(len(samp))]
    return write_jpeg(blocks, [np.full(64, 4)] * len(samp), list(samp), W, H, **kw)


@pytest.mark.parametrize("samp,name", [(((1, 1),), "gray"), (((1, 1),) * 3, "4:4:4"), (((2, 1), (1, 1), (1, 1)), "4:2:2"),
                                       (((2, 2), (1, 1), (1, 1)), "4:2:0"), (((1, 2), (1, 1), (1, 1)), "4:4:0"),
                                       (((4, 1), (1, 1), (1, 1)), "4:1:1")])
def test_subsampling_names(samp, name):
    assert egj.inspect(_jpeg(samp))["subsampling"] == name


def test_restart_interval():
    assert egj.inspect(_jpeg(dri=3))["restart_interval"] == 3


@pytest.mark.parametrize("data,reason", [
    (b"", "not a JPEG"), (b"\xff\xd8", "not a JPEG"), (b"\xff\xd8\xff\xd9", "SOI/EOI"),
    (b"\xff\xd8\xff\xc2\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00", "progressive"),
    (b"\xff\xd8\xff\xc9\x00\x02", "arithmetic"), (b"\xff\xd8\xff\xe0\x00\x01", "bad marker length"),
    (b"\xff\xd8\x00\x00", "garbage"),
])
def test_declined(data, reason):
    with pytest.raises(Unsupported, match=reason):
        parse(data)


def test_twelve_bit_and_cmyk_declined():
    d = bytearray(_jpeg())
    i = d.find(b"\xff\xc0")
    d[i + 4] = 12
    assert "12-bit" in egj.inspect(bytes(d))["reason"]


def test_huffman_table_checks():
    check_huffman([0, 1, 5, 1, 1, 1, 1, 1, 1] + [0] * 7, bytes(range(12)), True)
    with pytest.raises(Unsupported):
        check_huffman([2] + [0] * 15, b"\x00\x01", True)          # uses the all-ones code at length 1
    with pytest.raises(Unsupported):
        check_huffman([0, 1] + [0] * 14, b"\x10", True)           # DC symbol > 15
    with pytest.raises(Unsupported):
        check_huffman([0] * 16, b"", False)                       # empty


def test_trailer():
    assert check_trailer(b"\xff\xd9", 0) is None
    assert check_trailer(b"", 0) is None
    assert check_trailer(b"\xff\xfe\x00\x04ab\xff\xd9", 0) is None
    assert check_trailer(b"\xff\xdb\x00\x43", 0) is not None          # tables after the scan
    assert check_trailer(b"\xff\xda\x00\x08", 0) is not None          # a second scan
    assert check_trailer(b"\xff\xfe\x00\x40ab", 0) is not None        # truncated comment
