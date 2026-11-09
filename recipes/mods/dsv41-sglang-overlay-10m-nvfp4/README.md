# dsv41-sglang-overlay-10m-nvfp4

Fork of `dsv41-sglang-overlay-10m` for `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`.
Read `../dsv41-sglang-overlay-10m/launcher.py` first — this mod is that launcher
plus the two things this checkpoint genuinely needs, and nothing else.

## The only real port: MXFP4 Engram

The checkpoint stores both Engram tables as MXFP4 (packed E2M1, 2 values/byte,
one E8M0 scale per 32 elements) instead of FP8. The overlay's Engram stack is
FP8-only in four places that all have to agree:

| # | place | what assumes FP8 |
|---|---|---|
| 1 | `adapter/engram_backend.py` | `assert w['shape'] == [n, 256] and w['dtype'] == 'F8_E4M3'` |
| 2 | `adapter/librow_store.so` (compiled) | `kRowBytes = 264` (256 weight + 8 scale), validated against the packed shard header in **both** `row_store_open` and `row_store_attach_packed` |
| 3 | `scripts/pack_engram.py` | `WEIGHT_BYTES = 256`, same asserts |
| 4 | `sglang .../engram_gather.py` (Triton) | loads `fp8e4nv` payload |

(2) is compiled, so the mod takes the cheap road: **expand MXFP4 to FP8 at pack
time**. The decode is bit-exact — every E2M1 code is an exact e4m3 value — so a
packed shard built this way is value-identical to an FP8 one and (1)–(4) keep
working unchanged. The decode is sgl-project/sglang#47935 restated in numpy.

`engram_mxfp4.py` rewrites (1) and (3) in the container's writable layer at boot
(idempotent, per-container, warns instead of booting into a drift). `prepack.sh`
applies the same patches and then runs the packer; it exists as a host-side
script because the pack takes minutes per node and sparkrun's rendezvous wait is
~150 s, so it cannot ride inside the launch.

**Packed shards are mandatory.** With a packed shard attached, `row_store` serves
every miss from it; without one it reads the checkpoint at 256-byte stride, which
is the wrong stride for a 128-byte MXFP4 row. So a missing shard means wrong rows,
not slow rows — `gate_packed()` fails the launch with the fix in the message.

Cost: the FP8-expanded records are 264 B again, so the pack is ~95 GiB/rank, the
same as the fp8 lane, and the per-lookup byte win is NOT realised yet. What is
realised: 82 GB less checkpoint on disk, 82 GB less read by the packer, and
half-width rows available later.

v2 sketch, not done here: give `row_store` a per-store `row_bytes` read from the
packed header (drop `kRowBytes` from the two validations, take `weight_bytes =
row_bytes - 8` from it) and unpack in the gather instead of at pack time. Then the
rows stay 136 B and the pack is 49 GiB/rank.

## Experts: no patch

The experts are ModelOpt NVFP4 (E2M1 + group-16 e4m3 scale + `input_scale` +
`weight_scale_2`, tensors named `…w1.weight` / `.weight_scale` / `.weight_scale_2`
/ `.input_scale`). sglang already knows this shape of checkpoint —
`configs/model_config.py:566` auto-detects `quant_algo: MIXED_PRECISION` +
`moe_quant_algo: NVFP4` + `group_size` into `nvfp4_moe_meta`, and
`arg_groups/model_overrides/deepseek_v4.py` routes MoE to
`flashinfer_trtllm_routed`. `b12x_next` in the base overlay hooks
`FusedOpPool[("none","flashinfer_mxfp4")]`, which that runner does not use, so
`DSV41_MOE_B12X_NEXT` defaults to **0** here. Its `PackedSourceFormat` does carry
`MODELOPT_NVFP4`, so the b12x line can be restored later by forcing
`--moe-runner-backend flashinfer_mxfp4` and selecting that format in
`moe_b12x_next.py:401` — unmeasured on this lane.

## Memory

Same KV as the fp8 lane (`kv_cache_quant_algo: null` in the checkpoint; both
NVFP4 and fp8 lanes serve fp8 KV at 1670.75 B/token, 17.52 GB/node at 10.49M).
The expert scale byte costs ~4.2 GB/node of resident constant, so
`MEM_FRACTION_STATIC` is 0.86, not 0.84: 0.84 yields ~9.4M tokens and misses the
10,485,760 prompt. 0.86 projects ~11M with ~18.7 GB of host slack; the head
node's NVRM OOM is the risk, not the OOM killer.

## Use

Two recipes ride on this mod, both `--trust` (they carry `cap_add`):

| recipe | context | mem fraction | first? |
|---|---|---|---|
| `rumplestilzken--DeepSeek-V4.1-Flash-1017-abliterated-nvfp4` | 1M | 0.78 | **yes** — the verified shape |
| `nvidia--DeepSeek-V4.1-Flash-1017-NVFP4-10m` | 10M | 0.86 | after 1M serves |

Both set `DSV41_ENGRAM_PACK_KEY: nvidia__DeepSeek-V4.1-Flash-1017-NVFP4`: the
tables are not ablated, so one pack serves both. `DSV41_MEM_FRACTION`,
`DSV41_MAX_TOTAL_TOKENS`, `DSV41_MAX_RUNNING_REQUESTS` and
`DSV41_CUDA_GRAPH_MAX_BS_DECODE` are the per-context overrides of the mod's 10M
defaults.

```bash
# once per node, rank 0..3 = spark1..spark4 (pack reads the SOURCE checkpoint,
# which must be in that node's HF cache)
bash /tmp/nvfp4mod/prepack.sh 3

sparkrun run rumplestilzken--DeepSeek-V4.1-Flash-1017-abliterated-nvfp4 --trust
```

## Boot gates this mod adds

- `launcher: engram mxfp4 patch -> packer patched, backend patched`
- `packed Engram present for rank R/4 (layers [1, 14])` — or a hard stop naming
  `prepack.sh`
- engine log: `Auto-detected hybrid FP8+NVFP4 checkpoint (NVFP4 MoE group_size=16…)`
  then `Use flashinfer_trtllm_routed as MoE runner backend`
- `DSV4 memory calculation` must print `full_token` ≥ 10.49M; then
  `/v1/models` `max_model_len` 10485760 is the tell the lane is up

## Unproven, in order of risk

1. `flashinfer_trtllm_routed` on sm_121 — NVIDIA validated GB300, not GB10.
2. Quality past 1M is 10x YaRN extrapolation, same as the fp8 lane.
3. The 0.86/11M projection is arithmetic from the engine's own 0.80 measurement;
   nothing above 0.84 has been booted on either lane.
4. Prefill wall clock at 10M is estimated 1.9–3.1 h; decode floors near 15 t/s.