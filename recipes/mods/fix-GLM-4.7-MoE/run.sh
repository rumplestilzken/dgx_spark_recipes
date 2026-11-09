#!/bin/bash
echo "--- Applying GLM 4.7 MoE patch..."
grep -q 'k_scale.*v_scale.*q_scale' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe.py || \
sed -i '/# Skip non-stacked layers and experts (experts handled below)\./i\
                if name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):\
                    continue' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe.py