#!/usr/bin/env python3
"""
FakeLens Environment & Project Verification Script
Verifies Python version, directory structure, and essential packages.
"""

import sys
from pathlib import Path

REQUIRED_DIRECTORIES = [
    "backend",
    "backend/api",
    "backend/datasets",
    "backend/preprocessing",
    "backend/features",
    "backend/models",
    "backend/training",
    "backend/evaluation",
    "backend/utils",
    "frontend",
    "notebooks",
    "configs",
    "results",
    "tests",
]

CORE_MODULES = [
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("torch", "PyTorch"),
    ("torchvision", "Torchvision"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("fastapi", "FastAPI"),
]


def check_python_version() -> bool:
    print(f"[1/4] Checking Python version... Current: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  [FAIL] Python 3.10 or higher is required.")
        return False
    print("  [OK] Python version is compatible.")
    return True


def check_directories() -> bool:
    print("\n[2/4] Verifying project directory layout...")
    root = Path(__file__).resolve().parent.parent
    all_present = True
    for folder in REQUIRED_DIRECTORIES:
        p = root / folder
        if p.exists() and p.is_dir():
            print(f"  [OK] {folder}/")
        else:
            print(f"  [FAIL] Missing directory: {folder}/")
            all_present = False
    return all_present


def check_package_imports() -> dict:
    print("\n[3/4] Checking availability of ML and project libraries...")
    installed = {}
    for mod_name, label in CORE_MODULES:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {label} ({ver})")
            installed[label] = True
        except ImportError:
            print(f"  [NOT INSTALLED] {label} (run 'pip install -r requirements.txt')")
            installed[label] = False
    return installed


def check_accelerator() -> None:
    print("\n[4/4] Checking compute accelerator availability...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  [OK] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("  [INFO] CUDA GPU not detected. CPU mode will be used for local execution.")
    except ImportError:
        print("  [SKIP] PyTorch not yet installed.")


def main():
    print("=" * 60)
    print("  FakeLens Project Setup & Environment Verification")
    print("=" * 60)

    py_ok = check_python_version()
    dirs_ok = check_directories()
    installed = check_package_imports()
    check_accelerator()

    print("\n" + "=" * 60)
    if py_ok and dirs_ok:
        print("Project structure and Python baseline verified.")
        if all(installed.values()):
            print("All core dependencies are installed and ready!")
        else:
            print("Note: To install remaining dependencies, run:")
            print("      pip install -r requirements.txt")
        print("=" * 60)
        sys.exit(0)
    else:
        print("Project setup check encountered issues. Please review above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
