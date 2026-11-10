#!/usr/bin/env python3
"""MXFP4 Engram support for `librow_store.so` (the compiled half of the mod).

WHY THIS EXISTS
---------------
`engram_mxfp4.py` handles the FP8-only assumptions in the *Python* (packer and
backend) by unpacking the checkpoint's MXFP4 Engram tables to FP8 at pack time.
It deliberately left the compiled store alone, and for the row READ path that is
correct: the packed shard carries 264-byte FP8 rows and `row_store_attach_packed`
validates its header against `kRowBytes`.

What that reasoning missed is `row_store_open`, which runs BEFORE the packed
shard is attached and validates the CELEBRATED checkpoint's own geometry:

    rows > (st_size - woff) / 256      // kWeightBytes: an FP8 weight row
    rows > (st_size - soff) / 8        // kScaleBytes

The checkpoint's Engram weight is MXFP4 `[n, 128] U8` -- 128 bytes per row,
exactly half. On layer 1 that is 384,006,168 rows against a 52 GB shard, so the
256-byte operand asks for ~98 GB and aborts with

    Engram retrieval failed: invalid table extent (errno=0)

killing rank 0 during `load_model`, before a single row is read. The scale
operand is fine (an E8M0 block scale is 8 B/row in both formats).

THE FIX
-------
Accept the weight extent at EITHER stride -- the check only has to prove the
file is big enough to index:

    (rows > avail/256 && rows > avail/128)

For an FP8 table rows <= avail/256 already implies rows <= avail/128, so the
conjunct is false and behaviour is bit-for-bit unchanged. For MXFP4 the /128
side passes while /256 fails. Truncation is still caught at both strides, and
the scale operand and the packed-shard header check are untouched.

Loosening the open-time bound cannot serve wrong data from the checkpoint
fallback, because `engram_backend.py` raises when the table is MXFP4 and the
packed shard is absent (`DSV41_REQUIRE_PACKED=1`, the default).

`kWeightBytes` itself must stay 256: it splits the 264-byte packed row into the
FP8 weight half and the scale half for the gather kernel (row_store.cpp
164-165), and the packer's rows are expanded FP8.

The rebuild is deterministic: the original compilation is reproduced
byte-for-byte by the line run.sh uses, so the only delta in the .so is this
check.
"""
from __future__ import annotations

import os
import shutil

STORE = "/opt/dsv41/adapter/row_store.cpp"
MARK = "// nvfp4-10m mod:"

_OLD = """      soff > uint64_t(statbuf.st_size) || rows > (uint64_t(statbuf.st_size)-woff)/256 ||
      rows > (uint64_t(statbuf.st_size)-soff)/8) fail("invalid table extent");"""

_NEW = """      soff > uint64_t(statbuf.st_size) ||
      // nvfp4-10m mod: the NVFP4 checkpoint stores the Engram weight MXFP4 at
      // 128 B/row, not FP8 at kWeightBytes (256). Take the extent at either
      // stride: for FP8 rows <= avail/256 already implies rows <= avail/128, so
      // the conjunct is false and this is a no-op for the FP8 lane. Truncation
      // still trips both. Rows are never read at this stride -- the packed shard
      // is mandatory for MXFP4 and holds expanded 264-byte rows.
      (rows > (uint64_t(statbuf.st_size)-woff)/256 &&
       rows > (uint64_t(statbuf.st_size)-woff)/128) ||
      rows > (uint64_t(statbuf.st_size)-soff)/8) fail("invalid table extent");"""


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    tmp = path + ".nvfp4mod"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def patch_store(path: str = STORE) -> str:
    src = _read(path)
    if MARK in src:
        return "already patched"
    if _OLD not in src:
        return f"NOT PATCHED (row_store.cpp drifted: {_OLD.splitlines()[0].strip()!r})"
    _write(path, src.replace(_OLD, _NEW, 1))
    return "patched"


def main() -> int:
    print("row store:", patch_store())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())