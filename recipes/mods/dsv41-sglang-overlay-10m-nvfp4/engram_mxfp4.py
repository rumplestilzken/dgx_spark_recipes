#!/usr/bin/env python3
"""MXFP4 Engram support for the knapcio DSV41 overlay (NVFP4 checkpoint lane).

WHY THIS EXISTS
---------------
`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` stores both Engram lookup tables as
MXFP4: packed E2M1 (two values per byte, even column in the low nibble) with one
E8M0 block scale per 32 elements. The overlay's Engram stack is FP8-only in four
places that must agree:

  1. `adapter/engram_backend.py` asserts weight shape [n, 256] / dtype F8_E4M3
  2. `adapter/librow_store.so` hard-codes kRowBytes = 264 (256 weight + 8 scale)
     and validates the packed-shard header against it (`row_store_open` and
     `row_store_attach_packed` both)
  3. `scripts/pack_engram.py` hard-codes WEIGHT_BYTES = 256 and asserts F8_E4M3
  4. `sglang .../engram_gather.py` loads fp8 e4m3 payload only

The compiled `.so` is the expensive one to change, so this module's design is:
**unpack MXFP4 to FP8 at pack time.** The decode is bit-exact — every E2M1 code
maps to an exact e4m3 value — so the packed shard is value-identical to an FP8
one and the store, the gather kernel and the staging path stay untouched.

Kept: the 429 GB checkpoint instead of 511 GB, and half the source bytes the
packer reads. Given up: the per-lookup byte win, since packed rows are 256 B
again. A v2 can carry 136-byte rows by giving row_store a per-store stride; the
decode below would not change.

The decode is sgl-project/sglang#47935 (2026-11-04), restated in numpy.
"""
from __future__ import annotations

import os
import shutil

PACKER = "/opt/dsv41/scripts/pack_engram.py"
BACKEND = "/opt/dsv41/adapter/engram_backend.py"
MARK = "# nvfp4-10m mod"

_UNPACK = '''

def unpack_engram_rows(block):
    """[n, 128] uint8 E2M1 pairs -> [n, 256] uint8 e4m3, bit-exact.

    Even column in the low nibble (sglang#47935). E2M1 magnitudes 0, 0.5, 1, ...
    6 are e4m3 0x00, 0x30, then 4*m + 48; the sign nibble moves to the e4m3 sign
    bit. numpy so the packer keeps its buffer-only I/O path.
    """
    nib = numpy.stack([block & 0xF, block >> 4], axis=-1).reshape(block.shape[0], -1)
    magnitude = nib & 0x7
    bits = numpy.where(magnitude >= 2, magnitude * 4 + 48, magnitude * 48)
    return (bits | ((nib & 0x8) << 4)).astype(numpy.uint8)
'''

_OLD_SPAN = """    weight, scale = header[prefix + 'weight'], header[prefix + 'scale']
    assert weight['dtype'] == 'F8_E4M3' and scale['dtype'] == 'F8_E8M0'
    assert weight['shape'][1] == WEIGHT_BYTES and scale['shape'][1] == SCALE_BYTES
    base = 8 + length
    return (path, weight['shape'][0],
            base + weight['data_offsets'][0], base + scale['data_offsets'][0])
"""

_NEW_SPAN = """    weight, scale = header[prefix + 'weight'], header[prefix + 'scale']
    assert scale['dtype'] == 'F8_E8M0' and scale['shape'][1] == SCALE_BYTES
    # nvfp4-10m mod: MXFP4 table = [n, 128] U8, expanded to e4m3 while packing so
    # the record stays 264 B and row_store's kRowBytes check still passes.
    mxfp4 = weight['dtype'] == 'U8' and weight['shape'][1] == WEIGHT_BYTES // 2
    assert mxfp4 or (weight['dtype'] == 'F8_E4M3'
                     and weight['shape'][1] == WEIGHT_BYTES), (
        weight['dtype'], weight['shape'])
    base = 8 + length
    return (path, weight['shape'][0],
            base + weight['data_offsets'][0], base + scale['data_offsets'][0], mxfp4)
"""

_OLD_BODY = """    path, rows, weight_off, scale_off = tensor_span(root, layer_id)
"""

_NEW_BODY = """    path, rows, weight_off, scale_off, mxfp4 = tensor_span(root, layer_id)
    weight_bytes = WEIGHT_BYTES // 2 if mxfp4 else WEIGHT_BYTES
"""

_OLD_READ = """            src.seek(weight_off + start * WEIGHT_BYTES)
            weights = src.read(n * WEIGHT_BYTES)
            src.seek(scale_off + start * SCALE_BYTES)
            scales = src.read(n * SCALE_BYTES)
            if len(weights) != n * WEIGHT_BYTES or len(scales) != n * SCALE_BYTES:
                raise SystemExit(f'short read from {path} at row {start}')
            block = numpy.empty((n, ROW_BYTES), dtype=numpy.uint8)
            block[:, :WEIGHT_BYTES] = numpy.frombuffer(
                weights, dtype=numpy.uint8).reshape(n, WEIGHT_BYTES)
"""

_NEW_READ = """            src.seek(weight_off + start * weight_bytes)
            weights = src.read(n * weight_bytes)
            src.seek(scale_off + start * SCALE_BYTES)
            scales = src.read(n * SCALE_BYTES)
            if len(weights) != n * weight_bytes or len(scales) != n * SCALE_BYTES:
                raise SystemExit(f'short read from {path} at row {start}')
            block = numpy.empty((n, ROW_BYTES), dtype=numpy.uint8)
            block[:, :weight_bytes] = numpy.frombuffer(
                weights, dtype=numpy.uint8).reshape(n, weight_bytes)
            if mxfp4:
                block[:, :WEIGHT_BYTES] = unpack_engram_rows(block[:, :weight_bytes])
"""

_OLD_BACKEND = """        assert w['shape'] == [num_embeddings, 256] and w['dtype'] == 'F8_E4M3'
        assert s['shape'] == [num_embeddings, 8] and s['dtype'] == 'F8_E8M0'
"""

_NEW_BACKEND = """        # nvfp4-10m mod: accept an MXFP4 table ([n,128] U8, 2 values/byte).
        assert s['shape'] == [num_embeddings, 8] and s['dtype'] == 'F8_E8M0', (
            s['shape'], s['dtype'])
        mxfp4 = w['dtype'] == 'U8' and w['shape'] == [num_embeddings, 128]
        assert mxfp4 or (w['shape'] == [num_embeddings, 256]
                         and w['dtype'] == 'F8_E4M3'), (w['shape'], w['dtype'])
        if mxfp4 and os.getenv('DSV41_REQUIRE_PACKED', '1') == '1':
            _packed = Path(os.environ.get('DSV41_PACKED_DIR', '')) / (
                f'engram-l{layer_id}-r{rank}of{self.tp_size}.bin')
            if not _packed.exists():
                raise RuntimeError(
                    f'Engram is MXFP4 and {_packed} is absent: the checkpoint '
                    'cannot be read at the FP8 row stride. Run the prepack step '
                    'from mods/dsv41-sglang-overlay-10m-nvfp4/prepack.sh.')
"""


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    tmp = path + ".nvfp4mod"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def patch_packer(path: str = PACKER) -> str:
    src = _read(path)
    if MARK in src:
        return "already patched"
    for old, new in ((_OLD_SPAN, _NEW_SPAN), (_OLD_BODY, _NEW_BODY),
                     (_OLD_READ, _NEW_READ),
                     ("        _, rows, _, _ = tensor_span(root, layer_id)",
                      "        _, rows, _, _, _ = tensor_span(root, layer_id)")):
        if old not in src:
            return f"NOT PATCHED (packer drifted: {old.splitlines()[0].strip()!r})"
        src = src.replace(old, new, 1)
    if "def unpack_engram_rows" not in src:
        src = src.replace("def tensor_span(", _UNPACK.strip() + "\n\n\ndef tensor_span(", 1)
    if "nvfp4-10m mod" not in src.splitlines()[0]:
        src = f"{MARK}: MXFP4 -> FP8 at pack time\n" + src
    _write(path, src)
    return "patched"


def patch_backend(path: str = BACKEND) -> str:
    src = _read(path)
    if MARK in src:
        return "already patched"
    if _OLD_BACKEND not in src:
        return "NOT PATCHED (backend drifted)"
    src = src.replace(_OLD_BACKEND, _NEW_BACKEND, 1)
    src = src.replace("def ", "def ", 1)
    _write(path, src)
    return "patched"


def main() -> int:
    print("packer :", patch_packer())
    print("backend:", patch_backend())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())