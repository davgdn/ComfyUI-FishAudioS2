# Changelog

## [0.5.4-int4] — INT4 Patch Integration

### Overview
This branch integrates the improvements from [groxaxo/fish-speech-int4-patch](https://github.com/groxaxo/fish-speech-int4-patch) into [Saganaki22/ComfyUI-FishAudioS2](https://github.com/Saganaki22/ComfyUI-FishAudioS2).

### What changed

#### `fish_speech_src/fish_speech/models/text2semantic/llama.py`
- **Added** `_convert_linear_layers_to_bnb4(module, compute_dtype)` — replaces all `nn.Linear` layers with `bitsandbytes.nn.Linear4bit` (NF4, `compress_statistics=True`). Used to prepare the model structure before loading a pre-quantized state dict.
- **Added** `_load_prequantized_bnb4_state_dict(module, state_dict)` — loads pre-quantized `Params4bit` weights from a state dict created by `tools/export_nf4.py`, consuming those keys from the dict. Returns the set of consumed model keys so the caller can skip them in strict checks.
- **Updated** `DualARTransformer.from_pretrained` — added `bnb_mode='nf4-prequant'` branch that calls the two new functions above.

#### `nodes/loader.py`
- **Added** `"s2-pro-bnb-nf4-prequant"` entry to `HF_MODELS` dict. This model does NOT auto-download — the user must first run `tools/export_nf4.py` to create it from `s2-pro` weights.
- **Updated** `resolve_bnb_mode()` — returns `'nf4-prequant'` for model names containing `nf4-prequant` / `nf4_prequant`. Check is done **before** the existing `nf4` check (more specific match first).

#### `fish_speech_src/tools/llama/export_nf4.py` *(new file)*
- Standalone script to quantize an existing `s2-pro` checkpoint to NF4 and save it. Output is compatible with `bnb_mode='nf4-prequant'`. See usage below.

#### `__init__.py`
- Version bumped to `0.5.4-int4`.
- Docstring updated to document the integration.

---

### New model option: `s2-pro-bnb-nf4-prequant`

| Mode | VRAM | Startup | Notes |
|------|------|---------|-------|
| `s2-pro` | ~24 GB | Fast | Full precision |
| `s2-pro-bnb-nf4` | ~4–5 GB | Slow (quantizes on load) | On-the-fly NF4 |
| `s2-pro-bnb-nf4-prequant` | ~4–5 GB | **Fast** | Pre-quantized NF4 *(INT4 patch)* |

---

### How to use the pre-quantized model

**Step 1 — Export the quantized checkpoint** (one-time, ~5 min):

```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/fish_speech_src
python tools/llama/export_nf4.py \
  --input  ../../models/fishaudioS2/s2-pro \
  --output ../../models/fishaudioS2/s2-pro-bnb-nf4-prequant
```

**Step 2 — Select in ComfyUI**:
In any Fish S2 node, open the `model_name` dropdown and select `s2-pro-bnb-nf4-prequant`.

---

### Automation: keeping up with upstream updates

When `Saganaki22/ComfyUI-FishAudioS2` releases a new version, apply this patch as follows:

```bash
# 1. Add upstream remote (once)
git remote add upstream https://github.com/Saganaki22/ComfyUI-FishAudioS2.git

# 2. Fetch latest
git fetch upstream

# 3. Rebase this branch on top of new upstream main
git rebase upstream/main

# 4. Resolve any conflicts (see ADAPTATION_GUIDE.md for the files to watch)

# 5. Run the automated patch script to re-apply structural changes
python scripts/apply_int4_patch.py

# 6. Push
git push origin int4-patch --force-with-lease
```

See `ADAPTATION_GUIDE.md` for the full guide on what to check after each rebase.

