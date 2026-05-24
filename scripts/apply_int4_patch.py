#!/usr/bin/env python3
"""
apply_int4_patch.py — Automated INT4 patch verifier and applicator.

Usage:
    python scripts/apply_int4_patch.py --check    # dry-run: report missing changes
    python scripts/apply_int4_patch.py --apply    # apply any missing changes
    python scripts/apply_int4_patch.py            # same as --check

Run this after every `git rebase upstream/main` to make sure all INT4 patch
modifications are still in place.

Exit codes:
    0 — all checks passed (or all changes successfully applied)
    1 — checks failed and --apply was not requested (or apply itself failed)
"""

import argparse
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _contains(path: Path, marker: str) -> bool:
    return marker in _read(path)


def _insert_after(text: str, anchor: str, insertion: str) -> str:
    """Insert *insertion* immediately after the first occurrence of *anchor*."""
    idx = text.find(anchor)
    if idx == -1:
        raise ValueError(f"Anchor not found in file:\n  {anchor!r}")
    pos = idx + len(anchor)
    return text[:pos] + insertion + text[pos:]


def _replace_block(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Block not found:\n  {old!r}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

LLAMA_PY = HERE / "fish_speech_src/fish_speech/models/text2semantic/llama.py"
LOADER_PY = HERE / "nodes/loader.py"
EXPORT_NF4_PY = HERE / "fish_speech_src/tools/llama/export_nf4.py"

CHECKS = []


def check(name):
    """Decorator to register a check function."""
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Individual checks (each returns True if already OK, raises on apply failure)
# ---------------------------------------------------------------------------

@check("llama.py: _convert_linear_layers_to_bnb4 function")
def check_convert_linear(apply: bool) -> bool:
    marker = "def _convert_linear_layers_to_bnb4("
    if _contains(LLAMA_PY, marker):
        return True
    if not apply:
        return False

    insertion = textwrap.dedent("""

    def _convert_linear_layers_to_bnb4(
        module,
        compute_dtype=None,
    ):
        \"\"\"
        INT4 PATCH: Replace every nn.Linear inside *module* with
        bitsandbytes.nn.Linear4bit (NF4, compress_statistics=True).
        Used when bnb_mode='nf4-prequant' to prepare model structure
        before loading a pre-quantized state dict.
        \"\"\"
        import torch
        if compute_dtype is None:
            compute_dtype = torch.float16
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for NF4 prequantized loading.\\n"
                "Install it with:  pip install bitsandbytes"
            ) from exc

        import torch.nn as nn

        def _replace(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    quantized = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    quantized.load_state_dict(child.state_dict(), strict=False)
                    setattr(parent, name, quantized)
                else:
                    _replace(child)

        _replace(module)

    """)

    text = _read(LLAMA_PY)
    anchor = "\ndef find_multiple("
    new_text = _insert_after(text, anchor, insertion)
    _write(LLAMA_PY, new_text)
    return True


@check("llama.py: _load_prequantized_bnb4_state_dict function")
def check_load_prequant(apply: bool) -> bool:
    marker = "def _load_prequantized_bnb4_state_dict("
    if _contains(LLAMA_PY, marker):
        return True
    if not apply:
        return False

    insertion = textwrap.dedent("""

    def _load_prequantized_bnb4_state_dict(module, state_dict):
        \"\"\"
        INT4 PATCH: Load pre-quantized Params4bit weights from a state dict
        created by tools/export_nf4.py.  Quantized keys are consumed (popped)
        from state_dict in-place so the caller's subsequent load_state_dict
        call does not trip on them.
        Returns the set of model parameter keys that were successfully loaded.
        \"\"\"
        try:
            import bitsandbytes as bnb
            from bitsandbytes.nn import Params4bit
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for loading prequantized NF4 checkpoints."
            ) from exc

        consumed_model_keys = set()
        consumed_state_keys = set()

        for name, child in module.named_modules():
            if not isinstance(child, bnb.nn.Linear4bit):
                continue
            prefix = f"{name}." if name else ""
            weight_key = prefix + "weight"
            quant_prefix = weight_key + "."
            quantized_stats = {
                k[len(quant_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(quant_prefix)
            }
            quantized_state_keys = [prefix + "weight." + key for key in quantized_stats]
            if weight_key not in state_dict or not quantized_stats:
                continue
            child.weight = Params4bit.from_prequantized(
                data=state_dict[weight_key],
                quantized_stats=quantized_stats,
                requires_grad=False,
                device=child.weight.device,
                module=child,
            )
            consumed_model_keys.add(weight_key)
            consumed_state_keys.add(weight_key)
            consumed_state_keys.update(quantized_state_keys)
            bias_key = prefix + "bias"
            if child.bias is not None and bias_key in state_dict:
                child.bias.data.copy_(state_dict[bias_key].to(dtype=child.bias.dtype))
                consumed_model_keys.add(bias_key)
                consumed_state_keys.add(bias_key)

        for key in consumed_state_keys:
            state_dict.pop(key, None)

        return consumed_model_keys

    """)

    text = _read(LLAMA_PY)
    anchor = "\ndef _convert_linear_layers_to_bnb4("
    if anchor not in text:
        print("  ERROR: _convert_linear_layers_to_bnb4 must be added first.")
        return False
    # Insert right after the whole _convert_linear_layers_to_bnb4 function.
    # Find the next def/class after that anchor.
    idx = text.find(anchor)
    next_def = text.find("\ndef ", idx + 1)
    if next_def == -1:
        next_def = text.find("\nclass ", idx + 1)
    new_text = text[:next_def] + insertion + text[next_def:]
    _write(LLAMA_PY, new_text)
    return True


@check("llama.py: nf4-prequant branch in from_pretrained")
def check_from_pretrained_branch(apply: bool) -> bool:
    marker = "nf4-prequant"
    if _contains(LLAMA_PY, marker):
        return True
    if not apply:
        return False

    insertion = textwrap.dedent("""
        elif bnb_mode == "nf4-prequant":
            # INT4 PATCH: load a pre-quantized NF4 checkpoint (export_nf4.py).
            logger.info(
                "Loading pre-quantized NF4 checkpoint (INT4 patch mode). "
                "Converts model structure to Linear4bit and loads quantized weights."
            )
            _convert_linear_layers_to_bnb4(model, compute_dtype=torch.float16)
            if isinstance(weights, dict):
                _load_prequantized_bnb4_state_dict(model, weights)
            logger.info("Pre-quantized NF4 weights loaded successfully.")
    """)

    text = _read(LLAMA_PY)
    # Find the end of the `elif bnb_mode == "nf4":` block
    anchor = 'elif bnb_mode == "nf4":'
    idx = text.find(anchor)
    if idx == -1:
        print("  ERROR: could not find 'elif bnb_mode == \"nf4\":' anchor in llama.py")
        return False
    # Find next elif/if/else/setup_lora after this block
    search_start = idx + len(anchor)
    for keyword in ("\n        elif ", "\n        if lora_config", "\n        return model"):
        next_kw = text.find(keyword, search_start)
        if next_kw != -1:
            new_text = text[:next_kw] + "\n" + insertion + text[next_kw:]
            _write(LLAMA_PY, new_text)
            return True
    print("  ERROR: could not find insertion point after nf4 branch in from_pretrained")
    return False


@check("loader.py: s2-pro-bnb-nf4-prequant in HF_MODELS")
def check_hf_models_entry(apply: bool) -> bool:
    marker = '"s2-pro-bnb-nf4-prequant"'
    if _contains(LOADER_PY, marker):
        return True
    if not apply:
        return False

    insertion = textwrap.dedent("""
    # INT4 PATCH: pre-quantized NF4 checkpoint (created with tools/export_nf4.py).
    "s2-pro-bnb-nf4-prequant": {
        "repo_id": "fishaudio/s2-pro",
        "description": "BNB NF4 pre-quantized (INT4 patch, ~4-5GB VRAM, fast startup). Run tools/export_nf4.py first.",
        "base_model": "s2-pro-bnb-nf4-prequant",
        "prequant": True,
    },
""")

    text = _read(LOADER_PY)
    anchor = "}\nHF_DEFAULT_MODEL_NAME"
    if anchor not in text:
        print("  ERROR: could not find HF_MODELS closing brace in loader.py")
        return False
    new_text = _replace_block(text, anchor, insertion + "}\nHF_DEFAULT_MODEL_NAME")
    _write(LOADER_PY, new_text)
    return True


@check("loader.py: nf4-prequant check in resolve_bnb_mode (before nf4 check)")
def check_resolve_bnb_mode(apply: bool) -> bool:
    marker = '"nf4-prequant"'
    if _contains(LOADER_PY, marker):
        # Also verify ordering: nf4-prequant check must come before nf4 check
        text = _read(LOADER_PY)
        prequant_pos = text.find('"nf4-prequant"')
        nf4_pos = text.find('"nf4"', prequant_pos + 1) if prequant_pos != -1 else -1
        if prequant_pos != -1 and (nf4_pos == -1 or prequant_pos < nf4_pos):
            return True
        # Wrong order — fall through to apply
        if not apply:
            print("  WARNING: nf4-prequant check exists but may be in wrong order")
            return False

    if not apply:
        return False

    text = _read(LOADER_PY)
    old_nf4_block = '    if "bnb-nf4" in name_lower or "bnb_nf4" in name_lower:\n        return "nf4"'
    new_nf4_block = (
        '    # INT4 PATCH: check prequant before generic nf4 (more specific first)\n'
        '    if "nf4-prequant" in name_lower or "nf4_prequant" in name_lower:\n'
        '        return "nf4-prequant"\n'
        '    if "bnb-nf4" in name_lower or "bnb_nf4" in name_lower:\n'
        '        return "nf4"'
    )
    if old_nf4_block not in text:
        print("  ERROR: could not find nf4 block in resolve_bnb_mode in loader.py")
        return False
    _write(LOADER_PY, text.replace(old_nf4_block, new_nf4_block, 1))
    return True


@check("tools/export_nf4.py exists")
def check_export_nf4(apply: bool) -> bool:
    if EXPORT_NF4_PY.is_file():
        return True
    if not apply:
        return False
    print(f"  WARNING: {EXPORT_NF4_PY} not found. Copy it from the int4 patch repo:")
    print(f"    cp fish-speech-int4-patch/tools/llama/export_nf4.py {EXPORT_NF4_PY}")
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", default=True, help="Dry-run only (default)")
    group.add_argument("--apply", action="store_true", help="Apply missing changes")
    args = parser.parse_args()

    do_apply = args.apply
    all_ok = True

    print(f"\n{'='*60}")
    print(f"INT4 Patch verifier — {'APPLY' if do_apply else 'CHECK'} mode")
    print(f"{'='*60}\n")

    for name, fn in CHECKS:
        try:
            ok = fn(apply=do_apply)
        except Exception as e:
            ok = False
            print(f"  [ERROR] {name}\n    Exception: {e}")
        status = "✅" if ok else ("✍️  APPLIED" if do_apply else "❌ MISSING")
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("✅  All INT4 patch checks passed.")
        sys.exit(0)
    elif do_apply:
        # Re-run checks after apply
        print("Re-checking after apply...\n")
        all_ok_after = True
        for name, fn in CHECKS:
            try:
                ok = fn(apply=False)
            except Exception:
                ok = False
            if not ok:
                all_ok_after = False
                print(f"  ❌ Still missing: {name}")
        if all_ok_after:
            print("\n✅  All changes applied successfully.")
            sys.exit(0)
        else:
            print("\n❌  Some changes could not be applied automatically.")
            print("    See ADAPTATION_GUIDE.md for manual steps.")
            sys.exit(1)
    else:
        print("❌  Some INT4 patch changes are missing.")
        print("    Run with --apply to fix automatically, or see ADAPTATION_GUIDE.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()
