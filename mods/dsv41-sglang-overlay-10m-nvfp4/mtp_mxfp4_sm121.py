#!/usr/bin/env python3
"""Route the MTP (DSpark draft) MXFP4 MoE to cutlass on sm121/GB10.

WHY THIS EXISTS
---------------
This checkpoint's MTP / DSpark draft experts are MXFP4 (the quant config puts
`mtp.*` in `ignore`, so they are NOT the target's NVFP4: I8 weight + F8_E8M0
group-32 scale). The target model is NVFP4 and serves on cutlass, but the draft
dies in CUDA-graph capture with::

    TrtllmGenBatchedGemmRunner ... no kernels runnable on sm121;
    this backend ships cubins for sm100, sm103 and sm107.

The draft's MoE method is chosen in
``modelopt_quant.py :: ModelOptMixedQuantConfig.get_quant_method``. For the
excluded (MTP) FusedMoE layers it hard-routes to
``Mxfp4FlashinferTrtllmMoEMethod`` -- unconditionally, with no sm check and no
cutlass branch -- unlike ``fp8.py``'s MXFP4 dispatch which, on sm90/sm120, uses
``Mxfp4FlashinferCutlassMoEMethod`` instead. The trtllm path ships fixed cubins
(sm100/103/107); the cutlass path JITs for the live arch, which is exactly why
the NVFP4 target already runs on sm121.

THE FIX
-------
Mirror ``fp8.py`` in the MTP fallback: on ``get_platform().is_sm90 or
get_platform().is_sm120`` use the cutlass method, else keep trtllm.
``get_platform().is_sm120`` resolves to ``is_sm120_supported``
(``device_capability_majors=[12]``), which **includes sm121/GB10** -- the exact
comment in sglang's ``common.py`` says so -- so GB10 takes the cutlass branch.
The cutlass method's own ``_use_mxfp8_act_scaling = get_platform().is_sm120`` is
then also True on sm121, so it uses the SM120 W4A8 (MXFP8xMXFP4) path, which is
the arch-appropriate one for consumer Blackwell.

``get_platform`` is already imported at the top of modelopt_quant.py, so the
only change is the branch below. The target (non-excluded, NVFP4) path is
untouched. Source-only edit, no rebuild.

Applies as root in run.sh (the serve user cannot write /sgl-workspace). The
block is only ever reached when DSpark is enabled (SPEC_ALGO!=off); with it off
the MTP layers are never built and this is a dead no-op.
"""
from __future__ import annotations

import os
import shutil

MODELOPT = (
    "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py"
)
MARK = "# nvfp4-10m mod:"

_OLD = """            # Fall back to MXFP4 for MTP MoE layers
            if self.is_fp4_experts:
                from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(Fp8MoEMethod(self), prefix=prefix)
"""

_NEW = """            # Fall back to MXFP4 for MTP MoE layers
            if self.is_fp4_experts:
                from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

                # nvfp4-10m mod: the MTP (DSpark draft) experts are MXFP4; the
                # stock line hard-routes them to the trtllm GEMM, which ships no
                # sm121 (GB10) cubin, so the draft dies in CUDA-graph capture.
                # Mirror fp8.py's MXFP4 dispatch: route to the cutlass method on
                # sm90/sm120 -- get_platform().is_sm120 == is_sm120_supported,
                # which INCLUDES sm121/GB10 -- and it JITs for the live arch like
                # the NVFP4 target already does on sm121. Keep trtllm for sm100.
                if get_platform().is_sm90 or get_platform().is_sm120:
                    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
                        Mxfp4FlashinferCutlassMoEMethod,
                    )

                    return Mxfp4FlashinferCutlassMoEMethod(
                        Fp8MoEMethod(self), prefix=prefix
                    )
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(Fp8MoEMethod(self), prefix=prefix)
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


def patch_modelopt(path: str = MODELOPT) -> str:
    src = _read(path)
    if MARK in src:
        return "already patched"
    if _OLD not in src:
        return "NOT PATCHED (modelopt MTP fallback drifted)"
    _write(path, src.replace(_OLD, _NEW, 1))
    return "patched"


def main() -> int:
    print("modelopt MTP mxfp4:", patch_modelopt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
