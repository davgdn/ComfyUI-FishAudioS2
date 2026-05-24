"""Installation script for FishAudioS2 custom node.

ComfyUI Manager runs this at install time, before the node's __init__.py
loads.  This guarantees all pip dependencies are available on first start.

WHY THIS EXISTS:
- descript-audio-codec and descript-audiotools pin protobuf<5 which conflicts
  with other ComfyUI nodes (tensorflow, mediapipe, florence2).  We install
  them with --no-deps and pre-install their transitive deps manually.
- bitsandbytes may pull in a CPU-only torch in embedded Python environments
  where the CUDA torch has no pip metadata.  We detect and fix that.

__init__.py still has a fallback auto-installer for git-clone installs.
"""

import importlib
import subprocess
import sys


def run_cmd(cmd, timeout=300):
    """Run a command and return (success, stdout, stderr)."""
    print(f"[FishAudioS2] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            print("[FishAudioS2] Success")
            return True, result.stdout, result.stderr
        else:
            print(f"[FishAudioS2] Failed: {result.stderr}")
            return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[FishAudioS2] Timeout after {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"[FishAudioS2] Error: {e}")
        return False, "", str(e)


def is_installed(package_name):
    """Check if a package is importable right now."""
    import importlib.util
    return importlib.util.find_spec(package_name) is not None


def pip_install(package, no_deps=False):
    """Install a package. Tries uv first, falls back to pip. Returns True on success."""
    python = sys.executable
    flags = []
    if no_deps:
        flags.append("--no-deps")

    # Split package string so multi-pkg specs like "torch torchaudio --index-url ..."
    # are passed as separate args (matching __init__.py's spec.split() behavior).
    pkg_args = package.split()

    # Try uv first (faster)
    cmd = [python, "-s", "-m", "uv", "pip", "install"] + pkg_args + flags
    success, _, _ = run_cmd(cmd)
    if success:
        return True

    # Fall back to pip
    cmd = [python, "-s", "-m", "pip", "install"] + pkg_args + flags
    success, _, _ = run_cmd(cmd)
    return success


def check_torch():
    """Check PyTorch installation. Returns (version, has_cuda)."""
    try:
        import torch
        version = torch.__version__
        has_cuda = "+cu" in version
        return version, has_cuda
    except ImportError:
        return None, False


def main():
    # Early exit: if all critical packages import cleanly, skip.
    try:
        import dac              # noqa: F401
        import audiotools       # noqa: F401
        import bitsandbytes     # noqa: F401
        import transformers     # noqa: F401
        import librosa          # noqa: F401
        print("[FishAudioS2] All dependencies already installed. Skipping.")
        return
    except ImportError:
        pass

    print("=" * 60)
    print("[FishAudioS2] Installing dependencies...")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Check PyTorch (we do NOT modify it)
    # ------------------------------------------------------------------
    print("\n[FishAudioS2] Step 1: Checking PyTorch...")
    torch_version, has_cuda = check_torch()

    if torch_version is None:
        print("[FishAudioS2] ERROR: PyTorch is not installed!")
        print("[FishAudioS2] ComfyUI requires PyTorch. Please check your installation.")
        return

    if has_cuda:
        print(f"[FishAudioS2] PyTorch {torch_version} (CUDA) - OK")
    else:
        print(f"[FishAudioS2] PyTorch {torch_version} - No CUDA detected")

    # ------------------------------------------------------------------
    # Step 2: Install standard dependencies
    # ------------------------------------------------------------------
    print("\n[FishAudioS2] Step 2: Installing standard dependencies...")

    standard_packages = [
        ("numpy",               "numpy"),
        ("tqdm",                "tqdm"),
        ("soundfile",           "soundfile"),
        ("loguru",              "loguru"),
        ("transformers",        "transformers>=4.45.2"),
        ("einops",              "einops>=0.7.0"),
        ("librosa",             "librosa>=0.10.1"),
        ("rich",                "rich>=13.5.3"),
        ("ormsgpack",           "ormsgpack"),
        ("pydantic",            "pydantic==2.9.2"),
        ("tiktoken",            "tiktoken>=0.8.0"),
        ("cachetools",          "cachetools"),
        ("zstandard",           "zstandard>=0.22.0"),
        ("resampy",             "resampy>=0.4.3"),
        ("safetensors",         "safetensors>=0.4.0"),
        ("pyrootutils",         "pyrootutils>=1.0.4"),
        ("natsort",             "natsort>=8.4.0"),
        ("loralib",             "loralib>=0.1.2"),
        ("hydra",               "hydra-core>=1.3.2"),
    ]

    for import_name, pip_name in standard_packages:
        if is_installed(import_name):
            print(f"[FishAudioS2] {pip_name} - already installed")
        else:
            print(f"[FishAudioS2] Installing {pip_name}...")
            pip_install(pip_name)

    # ------------------------------------------------------------------
    # Step 3: Install transitive deps of dac/audiotools
    # These MUST be installed before the --no-deps packages below.
    # ------------------------------------------------------------------
    print("\n[FishAudioS2] Step 3: Installing dac/audiotools transitive deps...")

    transitive_deps = [
        ("flatten_dict",        "flatten-dict"),
        ("importlib_resources", "importlib-resources"),
        ("julius",              "julius"),
        ("randomname",          "randomname"),
        ("ffmpy",               "ffmpy"),
        ("argbind",             "argbind"),
        ("tensorboard",         "tensorboard"),
    ]

    for import_name, pip_name in transitive_deps:
        if is_installed(import_name):
            print(f"[FishAudioS2] {pip_name} - already installed")
        else:
            print(f"[FishAudioS2] Installing {pip_name}...")
            pip_install(pip_name)

    # ------------------------------------------------------------------
    # Step 4: Install audiotools and dac with --no-deps
    # audiotools FIRST because dac imports audiotools at module load.
    # ------------------------------------------------------------------
    print("\n[FishAudioS2] Step 4: Installing audiotools and dac (with --no-deps)...")
    print("[FishAudioS2] Using --no-deps to avoid protobuf<5 constraint conflicts")

    if is_installed("audiotools"):
        print("[FishAudioS2] descript-audiotools - already installed")
    else:
        pip_install("descript-audiotools>=0.7.2", no_deps=True)

    if is_installed("dac"):
        print("[FishAudioS2] descript-audio-codec - already installed")
    else:
        pip_install("descript-audio-codec", no_deps=True)

    # ------------------------------------------------------------------
    # Step 5: Install bitsandbytes
    # ------------------------------------------------------------------
    print("\n[FishAudioS2] Step 5: Installing bitsandbytes...")
    if is_installed("bitsandbytes"):
        print("[FishAudioS2] bitsandbytes - already installed")
    else:
        pip_install("bitsandbytes")

    # ------------------------------------------------------------------
    # Step 6: Verify torch is still a CUDA build — auto-restore if not
    # ------------------------------------------------------------------
    print("\n[FishAudioS2] Step 6: Verifying PyTorch CUDA...")
    torch_version_after, has_cuda_after = check_torch()

    if torch_version_after and has_cuda_after:
        print(f"[FishAudioS2] PyTorch {torch_version_after} (CUDA) - still OK")
    elif torch_version_after and not has_cuda_after:
        print("=" * 60)
        print("[FishAudioS2] PyTorch lost CUDA — auto-restoring...")
        print("=" * 60)
        print(f"[FishAudioS2] A dependency downgraded PyTorch {torch_version_after} to CPU-only.")
        cuda_tag = "cu128"
        index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
        print(f"[FishAudioS2] Restoring torch with: --index-url {index_url}")
        pip_install(f"torch torchaudio --index-url {index_url}")
        # Verify restore worked
        torch_version_fixed, has_cuda_fixed = check_torch()
        if has_cuda_fixed:
            print(f"[FishAudioS2] PyTorch {torch_version_fixed} (CUDA) - restored!")
        else:
            print(f"[FishAudioS2] WARNING: PyTorch still not CUDA after restore.")
            print("  Try manually:")
            print(f"    {sys.executable} -m pip install torch torchaudio "
                  f"--index-url {index_url}")
        print("")
    else:
        print("[FishAudioS2] ERROR: PyTorch is missing!")

    # ------------------------------------------------------------------
    # Step 7: Final verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[FishAudioS2] Installation complete!")
    print("=" * 60)
    print("\n[FishAudioS2] Verification:")

    importlib.invalidate_caches()

    critical = [
        ("dac",           "descript-audio-codec"),
        ("audiotools",    "descript-audiotools"),
        ("bitsandbytes",  "bitsandbytes"),
        ("transformers",  "transformers"),
        ("librosa",       "librosa"),
    ]

    all_ok = True
    for import_name, display_name in critical:
        try:
            __import__(import_name)
            print(f"  [OK] {display_name}")
        except ImportError as e:
            print(f"  [FAIL] {display_name} - {e}")
            all_ok = False

    if not all_ok:
        print("\n[FishAudioS2] Some packages failed. Try restarting ComfyUI.")
        print(f"  If still broken: {sys.executable} -m pip install "
              "descript-audiotools>=0.7.2 descript-audio-codec bitsandbytes")

    print("")


if __name__ == "__main__":
    main()
