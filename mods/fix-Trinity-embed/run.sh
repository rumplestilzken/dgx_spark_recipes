echo "######## Applying fix-Trinity-embed #######"
#!/usr/bin/env bash
# sparkrun mod: trinity-embed-bf16
# Runtime monkeypatch so vLLM's afmoe loader builds embed_tokens/lm_head as BF16,
# matching arcee-ai/Trinity-Large-*-NVFP4 (modelopt, config_groups.targets==["Linear"]).
# Installs a .pth -> auto-imports on the driver AND every Ray worker in this container.
# Inert unless TRINITY_EMBED_BF16=1 (set in the recipe env), so it's safe to leave
# installed in an image shared across other model recipes.
set -euo pipefail

python3 <<'PYEOF'
import site, os

site_dir = site.getsitepackages()[0]
mod_path = os.path.join(site_dir, "_trinity_embed_bf16.py")
pth_path = os.path.join(site_dir, "zzz_trinity_embed_bf16.pth")   # zzz -> loads last

MODULE = '''\
import os
if os.environ.get("TRINITY_EMBED_BF16") == "1":
    import inspect
    try:
        import vllm.model_executor.layers.vocab_parallel_embedding as _vpe
    except Exception:
        _vpe = None
    if _vpe is not None:
        _cls = _vpe.VocabParallelEmbedding  # ParallelLMHead subclasses this -> both covered
        if not getattr(_cls, "_trinity_embed_bf16", False):
            _orig = _cls.__init__
            _sig = inspect.signature(_orig)
            if "quant_config" not in _sig.parameters:
                raise RuntimeError("[trinity-patch] VocabParallelEmbedding.__init__ signature changed; re-verify")
            def __init__(self, *a, **k):
                b = _sig.bind_partial(self, *a, **k)
                b.arguments["quant_config"] = None
                return _orig(*b.args, **b.kwargs)
            _cls.__init__ = __init__
            _cls._trinity_embed_bf16 = True
            print("[trinity-patch] embed_tokens/lm_head forced BF16 (quant_config=None)", flush=True)
'''

with open(mod_path, "w") as f:
    f.write(MODULE)
with open(pth_path, "w") as f:
    f.write("import _trinity_embed_bf16\n")   # .pth import-line: runs at interpreter startup

print(f"[trinity-mod] installed {mod_path}")
print(f"[trinity-mod] installed {pth_path} -> auto-import on driver + Ray workers")
PYEOF