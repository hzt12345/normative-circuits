import os
import torch
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def _env_path(*names):
    for n in names:
        v = os.environ.get(n)
        if v and os.path.isdir(v):
            return v
    return ""

MODEL_PATHS = {
    "Qwen/Qwen3-8B":              _env_path("QWEN3_8B_PATH"),
    "meta-llama/Llama-3.1-8B":    _env_path("LLAMA31_8B_PATH"),
    "meta-llama/Meta-Llama-3.1-8B": _env_path("LLAMA31_8B_PATH"),
    "mistralai/Mistral-7B-v0.1":  _env_path("MISTRAL_7B_V01_PATH"),
}

_PATCHED = False

def _install_redirect_patches():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    def _make_redirect(orig_fn):
        def _patched(cls, pretrained_model_name_or_path, *args, **kwargs):
            local = MODEL_PATHS.get(pretrained_model_name_or_path)
            if local is not None and os.path.isdir(local):
                pretrained_model_name_or_path = local
            return orig_fn(cls, pretrained_model_name_or_path, *args, **kwargs)
        return classmethod(_patched)

    AutoConfig.from_pretrained = _make_redirect(AutoConfig.from_pretrained.__func__)
    AutoModelForCausalLM.from_pretrained = _make_redirect(AutoModelForCausalLM.from_pretrained.__func__)
    AutoTokenizer.from_pretrained = _make_redirect(AutoTokenizer.from_pretrained.__func__)

def load_hooked_transformer(official_name: str, device: str = "cuda", dtype=torch.float16):
    local = MODEL_PATHS.get(official_name)
    if local and os.path.isdir(local):
        _install_redirect_patches()
        print(f"[local_model_loader] Redirecting {official_name} -> {local}")
        model = HookedTransformer.from_pretrained(
            official_name,
            device=device,
            dtype=dtype,
        )
        print(f"[local_model_loader] Loaded on {model.cfg.device}")
        return model
    else:
        print(f"[local_model_loader] No local path for {official_name}; using HF Hub.")
        return HookedTransformer.from_pretrained(official_name, device=device, dtype=dtype)
