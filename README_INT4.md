# ComfyUI-FishAudioS2 — INT4 Patch Branch

**Base node:** [Saganaki22/ComfyUI-FishAudioS2](https://github.com/Saganaki22/ComfyUI-FishAudioS2)  
**INT4 patch source:** [groxaxo/fish-speech-int4-patch](https://github.com/groxaxo/fish-speech-int4-patch)  
**Branch:** `int4-patch`

---

## What this branch adds

This branch integrates pre-quantized NF4 (4-bit) checkpoint support from the fish-speech-int4-patch project into the ComfyUI node.

| Model option | VRAM | Startup time | Notes |
|---|---|---|---|
| `s2-pro` | ~24 GB | Fast | Full precision |
| `s2-pro-bnb-int8` | ~8–9 GB | Medium | On-the-fly INT8 |
| `s2-pro-bnb-nf4` | ~4–5 GB | Slow | On-the-fly NF4 quantization every startup |
| **`s2-pro-bnb-nf4-prequant`** ⭐ | ~4–5 GB | **Fast** | Pre-quantized NF4 — quantize once, load fast forever |

---

## Installation

### Option A — Install directly from this branch

```bash
cd ComfyUI/custom_nodes
git clone -b int4-patch https://github.com/YOUR_USERNAME/ComfyUI-FishAudioS2 ComfyUI-FishAudioS2
```

Restart ComfyUI. Dependencies are auto-installed on first startup.

### Option B — Switch an existing installation

```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2
git remote add int4 https://github.com/YOUR_USERNAME/ComfyUI-FishAudioS2
git fetch int4
git checkout int4/int4-patch
```

---

## Using the pre-quantized NF4 model

### Step 1 — Export the quantized checkpoint (one-time, ~5 min)

You need the base `s2-pro` weights first (auto-downloaded when you run any node with `s2-pro` selected).

```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/fish_speech_src
python tools/llama/export_nf4.py \
  --input  ../../../models/fishaudioS2/s2-pro \
  --output ../../../models/fishaudioS2/s2-pro-bnb-nf4-prequant
```

This takes 3–5 minutes and uses ~8 GB RAM during export. The output is ~2.5 GB.

### Step 2 — Select in any Fish S2 node

Open any Fish S2 node → `model_name` dropdown → `s2-pro-bnb-nf4-prequant`

First load initializes the BNB kernels (~15 s). Subsequent loads use the cached checkpoint directly.

---

## Requirements for NF4 modes

- CUDA GPU (NVIDIA only — bitsandbytes does not support MPS/CPU)
- `bitsandbytes` package (auto-installed by the node)
- Tested on: RTX 3060 (12 GB), RTX 3090, RTX 4090

---

## Keeping up to date

This branch is automatically rebased on `upstream/main` daily via GitHub Actions.

Manual update:
```bash
git fetch origin
git rebase origin/int4-patch
```

To verify the INT4 patch is intact after any rebase:
```bash
python scripts/apply_int4_patch.py --check
```

To auto-fix missing patches:
```bash
python scripts/apply_int4_patch.py --apply
```

---

## Files changed vs upstream

| File | Change |
|---|---|
| `__init__.py` | Version bump, updated docstring |
| `nodes/loader.py` | Added `s2-pro-bnb-nf4-prequant` to `HF_MODELS`, updated `resolve_bnb_mode` |
| `fish_speech_src/fish_speech/models/text2semantic/llama.py` | Added `_convert_linear_layers_to_bnb4`, `_load_prequantized_bnb4_state_dict`, `nf4-prequant` branch in `from_pretrained` |
| `fish_speech_src/tools/llama/export_nf4.py` | New file — one-time quantization export tool |
| `scripts/apply_int4_patch.py` | New file — automated patch verifier/applicator |
| `.github/workflows/sync_upstream.yml` | New file — daily upstream rebase workflow |
| `CHANGELOG.md` | New file |
| `ADAPTATION_GUIDE.md` | New file — full documentation of every change |

See `ADAPTATION_GUIDE.md` for detailed documentation of every change and how to maintain the patch across upstream updates.

