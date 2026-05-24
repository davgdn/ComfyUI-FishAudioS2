# Adaptation Guide — INT4 Patch Integration

This document explains every change made to integrate the INT4 patch into the ComfyUI node,
and what to watch for when the upstream node releases a new version.

---

## Architecture overview

```
ComfyUI-FishAudioS2/
├── __init__.py               ← Node registration, auto-install deps
├── nodes/
│   ├── loader.py             ← [PATCHED] Model loading, HF_MODELS, resolve_bnb_mode
│   ├── model_cache.py        ← Model cache, offload/resume (unchanged)
│   ├── tts_node.py           ← TTS node UI (unchanged)
│   ├── voice_clone_node.py   ← Voice clone node UI (unchanged)
│   ├── multi_speaker_node.py ← Multi-speaker node UI (unchanged)
│   └── multi_speaker_split_node.py
└── fish_speech_src/          ← Bundled fish-speech source
    ├── fish_speech/
    │   └── models/
    │       └── text2semantic/
    │           ├── llama.py      ← [PATCHED] Added NF4 prequant functions
    │           └── inference.py  ← (unchanged, ComfyUI version kept)
    └── tools/
        └── llama/
            └── export_nf4.py ← [NEW] INT4 patch export tool
```

---

## File-by-file change reference

### 1. `nodes/loader.py`

#### 1a. `HF_MODELS` dict (line ~70)
**What was added:**
```python
"s2-pro-bnb-nf4-prequant": {
    "repo_id": "fishaudio/s2-pro",
    "description": "BNB NF4 pre-quantized (INT4 patch, ~4-5GB VRAM, fast startup). Run tools/export_nf4.py first.",
    "base_model": "s2-pro-bnb-nf4-prequant",
    "prequant": True,
},
```
**Why:** Registers the new model option in the ComfyUI dropdown.  
**What to watch:** If upstream adds new models to `HF_MODELS`, ensure they don't conflict with the `nf4-prequant` name. The `prequant: True` flag is informational only.

#### 1b. `resolve_bnb_mode()` function
**What was changed:**
```python
# Before (original):
def resolve_bnb_mode(model_name: str) -> str | None:
    name_lower = model_name.lower()
    if "bnb-int8" in name_lower or "bnb_int8" in name_lower:
        return "int8"
    if "bnb-nf4" in name_lower or "bnb_nf4" in name_lower:
        return "nf4"
    return None

# After (patched):
def resolve_bnb_mode(model_name: str) -> str | None:
    name_lower = model_name.lower()
    if "bnb-int8" in name_lower or "bnb_int8" in name_lower:
        return "int8"
    # More specific check before generic nf4
    if "nf4-prequant" in name_lower or "nf4_prequant" in name_lower:
        return "nf4-prequant"
    if "bnb-nf4" in name_lower or "bnb_nf4" in name_lower:
        return "nf4"
    return None
```
**Critical:** The `nf4-prequant` check **must come before** the `nf4` check — `"nf4-prequant"` contains `"nf4"` so the wrong branch would match otherwise.

**What to watch after upstream update:**
- If upstream changes `resolve_bnb_mode`, re-apply the prequant block in the same relative position.
- If upstream renames BNB modes, update the string checks accordingly.

---

### 2. `fish_speech_src/fish_speech/models/text2semantic/llama.py`

#### 2a. New functions added after `_apply_bnb_nf4` (line ~251)

Two functions were inserted between `_apply_bnb_nf4` and `find_multiple`:

```python
def _convert_linear_layers_to_bnb4(module, compute_dtype=torch.float16):
    """Replace nn.Linear → bnb.nn.Linear4bit (NF4) recursively."""
    ...

def _load_prequantized_bnb4_state_dict(module, state_dict):
    """Load Params4bit weights from a pre-quantized state dict."""
    ...
```

**What to watch:** If upstream refactors the BNB helpers or adds FP4 support, check for naming conflicts with `_convert_linear_layers_to_bnb4`.

#### 2b. `DualARTransformer.from_pretrained` — new `elif` branch

Inside the `if bnb_mode == "int8": ... elif bnb_mode == "nf4": ...` block, added:

```python
elif bnb_mode == "nf4-prequant":
    logger.info("Loading pre-quantized NF4 checkpoint (INT4 patch mode)...")
    _convert_linear_layers_to_bnb4(model, compute_dtype=torch.float16)
    if isinstance(weights, dict):
        _load_prequantized_bnb4_state_dict(model, weights)
    logger.info("Pre-quantized NF4 weights loaded successfully.")
```

**What to watch:**
- `weights` variable name: this must refer to the local state-dict variable at that point in `from_pretrained`. If upstream renames it, update the branch.
- The `weights` dict is mutated in-place by `_load_prequantized_bnb4_state_dict` (quantized keys are popped). The subsequent `model.load_state_dict(weights, strict=False)` call should then skip those keys cleanly. If upstream adds a strict-check after this block, you may need to pass the consumed keys set.
- If upstream changes the order of BNB `elif` branches, preserve the `nf4-prequant` check.

---

### 3. `fish_speech_src/tools/llama/export_nf4.py` *(new file)*

This file is a standalone script from the int4 patch. It does NOT integrate with the ComfyUI node at runtime — it is a one-time utility. No patching needed; just keep it in place.

---

## Automated patch script

`scripts/apply_int4_patch.py` re-applies all structural changes programmatically.
Run it after every upstream rebase to verify the patch is intact:

```bash
python scripts/apply_int4_patch.py --check   # dry-run: reports missing changes
python scripts/apply_int4_patch.py --apply   # applies any missing changes
```

The script checks for:
1. `nf4-prequant` key in `HF_MODELS` (loader.py)
2. `nf4-prequant` check in `resolve_bnb_mode` (loader.py)
3. `_convert_linear_layers_to_bnb4` function definition (llama.py)
4. `_load_prequantized_bnb4_state_dict` function definition (llama.py)
5. `nf4-prequant` branch in `from_pretrained` (llama.py)
6. `export_nf4.py` exists in tools/llama/

---

## Compatibility matrix

| Upstream version | Status | Notes |
|-----------------|--------|-------|
| 0.5.3 (base)    | ✅ Patched | This branch |
| future          | ⚠️ Rebase needed | Run `apply_int4_patch.py --check` |

---

## Common rebase conflicts and how to resolve them

### Conflict in `llama.py` near BNB helpers
The upstream may update `_apply_bnb_nf4` or add new quantization modes.
- Keep upstream's changes to `_apply_bnb_nf4` / `_apply_bnb_int8`.
- Keep our additions (`_convert_linear_layers_to_bnb4`, `_load_prequantized_bnb4_state_dict`) after those functions.
- Ensure `find_multiple` follows immediately after our additions.

### Conflict in `loader.py` `HF_MODELS`
- Keep all upstream model entries.
- Re-add `s2-pro-bnb-nf4-prequant` at the end of the dict.

### Conflict in `loader.py` `resolve_bnb_mode`
- Keep upstream's logic.
- Insert the `nf4-prequant` check immediately before the `nf4` check.

