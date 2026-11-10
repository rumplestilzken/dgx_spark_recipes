#!/usr/bin/env bash
# Prepack this node's Engram rows for the NVFP4 lane. ONE TIME per node.
#
# Why: the checkpoint's Engram tables are MXFP4 (128 B rows). The overlay's
# compiled row store reads 264 B records at an FP8 stride and validates the
# packed-shard header against exactly 264 -- so the pack step expands E2M1 to FP8
# while packing (bit-exact) and the store never sees a 4-bit row. A boot without
# these shards serves wrong rows, not slow ones; the launcher refuses to start.
#
# Usage, ON THE NODE (rank = 0..3 for spark1..spark4). Copy the mod dir to the
# node first -- it is not in a repo checkout there:
#     tar -C <mods> -czf - dsv41-sglang-overlay-10m-nvfp4 |
#         ssh nick@spark4 'rm -rf /tmp/nvfp4mod && mkdir -p /tmp/nvfp4mod &&
#                          tar -C /tmp/nvfp4mod -xzf - && mv /tmp/nvfp4mod/*/* /tmp/nvfp4mod/'
#     ssh nick@spark4 'bash /tmp/nvfp4mod/prepack.sh 3'
# or from the dev box for all four at once:
#     for i in 0 1 2 3; do ssh -o BatchMode=yes nick@spark$((i+1)) \
#         "bash /tmp/nvfp4mod/prepack.sh $i"; done
#
# Needs ~95 GiB free per node (two layers x 47 GiB; the FP8-expanded records are
# the same size as before, the checkpoint's halving does not carry through).
set -euo pipefail

RANK="${1:?usage: prepack.sh <rank 0-3>}"
TP=4
IMAGE=littlecedar/dgx-spark-dsv41:canary-roce-20261022
# The tables are not ablated: the abliterated repack carries the same ones, so the
# pack key is fixed and one pack serves every recipe on this model. The source
# is the DEPLOYED abliterated checkpoint: the NVIDIA original is on no node, and
# the ablation never touches the Engram tensors, so the tables are identical.
MODEL_ID="${PACK_SOURCE_ID:-rumplestilzken/DeepSeek-V4.1-1017-abliterated-NVFP4}"
SLUG="${PACK_KEY:-rumplestilzken__DeepSeek-V4.1-Flash-1017-NVFP4}"
SRC_SLUG="${MODEL_ID//\//--}"
HFC="${HOME}/.cache/huggingface"
REPO="${HFC}/hub/models--${SRC_SLUG}"
[[ -d "${REPO}" ]] || REPO="${HFC}/hub/models--${MODEL_ID//\//__}"
OUT_HOST="${HFC}/hub/engram-packed/${SLUG}"
# The snapshot is symlinks into ../../blobs, resolved by the RUNNING PROCESS's
# root. Bind the snapshot alone and ../../ lands on the image's /, where no blobs
# exist -> every link dangles. So: bind the CACHE ROOT at /models and address the
# snapshot by its in-cache path; then ../../blobs resolves to /models/blobs.
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -d "${REPO}" ]] || { echo "no checkpoint at ${REPO}" >&2; exit 1; }
SNAP="${REPO}/snapshots/$(cat "${REPO}/refs/main")"
[[ -f "${SNAP}/model.safetensors.index.json" ]] || { echo "snapshot incomplete: ${SNAP}" >&2; exit 1; }
SNAP_IN_CACHE="${SRC_SLUG_NAME:-$(basename "${REPO}")}/snapshots/$(basename "${SNAP}")"

# space check (the packer has its own, but fail before creating anything)
# 2 layers x ~47 GiB per rank = ~95 GiB; leave a margin.
need_gib=105
free_gib=$(df -BG --output=avail "${HFC}" | tail -1 | tr -dc '0-9')
(( free_gib >= need_gib )) || { echo "need ${need_gib} GiB free at ${HFC}, have ${free_gib}" >&2; exit 1; }

mkdir -p "${OUT_HOST}"
echo "rank ${RANK}/${TP}: packing -> ${OUT_HOST}"

docker run --rm \
  --entrypoint bash \
  -e NODE_RANK="${RANK}" -e TP_SIZE="${TP}" -e DSV41_MODEL=/models/${SNAP_IN_CACHE} \
  -v "${HFC}/hub:/models:ro" \
  -v "${HFC}:/cache/huggingface" \
  -v "${MOD_DIR}:/workspace/mods/dsv41-sglang-overlay-10m-nvfp4:ro" \
  "${IMAGE}" -lc '
    set -euo pipefail
    cd /opt/dsv41
    python3 /workspace/mods/dsv41-sglang-overlay-10m-nvfp4/engram_mxfp4.py
    # prove the patches took before writing 95 GiB
    grep -q "nvfp4-10m mod" scripts/pack_engram.py
    grep -q "MXFP4" adapter/engram_backend.py
    python3 scripts/pack_engram.py \
      --model "${DSV41_MODEL}" --out "/cache/huggingface/hub/engram-packed/'"${SLUG}"'" \
      --rank '"${RANK}"' --tp '"${TP}"'
  '

# completeness: both layers, right sizes, no .partial left behind
for layer in 1 14; do
  f="${OUT_HOST}/engram-l${layer}-r${RANK}of${TP}.bin"
  [[ -f "${f}" && ! -e "${f%.bin}.partial" ]] || { echo "missing ${f}" >&2; exit 1; }
  printf 'ok %s (%.1f GiB)\n' "$(basename "${f}")" "$(stat -c%s "${f}" | awk '{print $1/1073741824}')"
done
echo "rank ${RANK} done. The launcher will gate on these before it boots."