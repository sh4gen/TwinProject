#!/usr/bin/env python3
"""
Setup verification script for TransReID training on LTCC dataset
Checks all requirements before starting training
"""

import os
import sys
from pathlib import Path


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_check(name, status, details=""):
    """Print a check result"""
    icon = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
    status_text = f"{Colors.GREEN}OK{Colors.RESET}" if status else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"{icon} {name:.<50} {status_text}")
    if details:
        print(f"  {Colors.YELLOW}{details}{Colors.RESET}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 6)
    status = version >= required
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    if not status:
        details += f" (Required: >= {required[0]}.{required[1]})"
    return status, details


def check_pytorch():
    """Check PyTorch installation"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        details = f"PyTorch {version}"
        if cuda_available:
            details += f", CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)"
        else:
            details += ", CUDA not available"
        return True, details
    except ImportError:
        return False, "PyTorch not installed"


def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown version')
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} not installed"


def check_dataset_structure():
    """Check LTCC dataset structure"""
    dataset_path = Path('/home/ika/yzlm/TwinProject/experiments/reid/ltcc/data')

    if not dataset_path.exists():
        return False, f"Dataset directory not found: {dataset_path}"

    train_dir = dataset_path / 'bounding_box_train'
    query_dir = dataset_path / 'query'
    gallery_dir = dataset_path / 'bounding_box_test'

    missing = []
    if not train_dir.exists():
        missing.append('bounding_box_train')
    if not query_dir.exists():
        missing.append('query')
    if not gallery_dir.exists():
        missing.append('bounding_box_test')

    if missing:
        return False, f"Missing directories: {', '.join(missing)}"

    # Count images
    train_count = len(list(train_dir.glob('*.jpg'))) if train_dir.exists() else 0
    query_count = len(list(query_dir.glob('*.jpg'))) if query_dir.exists() else 0
    gallery_count = len(list(gallery_dir.glob('*.jpg'))) if gallery_dir.exists() else 0

    details = f"Train: {train_count}, Query: {query_count}, Gallery: {gallery_count} images"
    return True, details


def check_pretrained_model():
    """Check if pre-trained model exists"""
    pretrain_path = Path.home() / '.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'

    if pretrain_path.exists():
        size_mb = pretrain_path.stat().st_size / (1024 * 1024)
        return True, f"Found at {pretrain_path} ({size_mb:.1f} MB)"
    else:
        return False, f"Not found at {pretrain_path}"


def check_transreid_files():
    """Check TransReID files"""
    script_dir = Path(__file__).parent
    transreid_dir = script_dir / 'TransReID'

    checks = []

    # Check TransReID directory
    if not transreid_dir.exists():
        return False, "TransReID directory not found"

    # Check important files
    files = {
        'train.py': transreid_dir / 'train.py',
        'test.py': transreid_dir / 'test.py',
        'LTCC dataset': transreid_dir / 'datasets/ltcc.py',
        'LTCC config': transreid_dir / 'configs/LTCC/vit_transreid_stride.yml',
    }

    missing = [name for name, path in files.items() if not path.exists()]

    if missing:
        return False, f"Missing files: {', '.join(missing)}"

    return True, "All files present"


def check_disk_space():
    """Check available disk space"""
    import shutil
    stats = shutil.disk_usage('/')
    free_gb = stats.free / (1024 ** 3)
    total_gb = stats.total / (1024 ** 3)
    used_percent = (stats.used / stats.total) * 100

    details = f"Free: {free_gb:.1f} GB / Total: {total_gb:.1f} GB ({used_percent:.1f}% used)"

    # Warn if less than 10 GB free
    if free_gb < 10:
        return False, details + " - WARNING: Low disk space!"

    return True, details


def print_installation_instructions():
    """Print installation instructions for missing components"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Installation Instructions:{Colors.RESET}\n")

    print("1. Install PyTorch (with CUDA support):")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n")

    print("2. Install required packages:")
    print("   cd TransReID")
    print("   pip install -r requirements.txt\n")

    print("3. Download pre-trained ViT-Base model:")
    print("   mkdir -p ~/.cache/torch/checkpoints")
    print("   cd ~/.cache/torch/checkpoints")
    print("   wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth\n")


def main():
    """Main function"""
    print_header("TransReID Setup Verification for LTCC Dataset")

    all_passed = True

    # Python version
    print(f"{Colors.BOLD}System Checks:{Colors.RESET}")
    status, details = check_python_version()
    print_check("Python Version", status, details)
    all_passed &= status

    status, details = check_disk_space()
    print_check("Disk Space", status, details)
    all_passed &= status

    # PyTorch and dependencies
    print(f"\n{Colors.BOLD}Package Checks:{Colors.RESET}")
    status, details = check_pytorch()
    print_check("PyTorch", status, details)
    all_passed &= status

    packages = [
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('yacs', 'yacs'),
        ('opencv-python', 'cv2'),
    ]

    for pkg_name, import_name in packages:
        status, details = check_package(pkg_name, import_name)
        print_check(pkg_name, status, details)
        all_passed &= status

    # TransReID files
    print(f"\n{Colors.BOLD}TransReID Files:{Colors.RESET}")
    status, details = check_transreid_files()
    print_check("TransReID Setup", status, details)
    all_passed &= status

    # Dataset
    print(f"\n{Colors.BOLD}Dataset Checks:{Colors.RESET}")
    status, details = check_dataset_structure()
    print_check("LTCC Dataset", status, details)
    all_passed &= status

    # Pre-trained model
    print(f"\n{Colors.BOLD}Pre-trained Model:{Colors.RESET}")
    status, details = check_pretrained_model()
    print_check("ViT-Base Model", status, details)
    all_passed &= status

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! You're ready to start training.{Colors.RESET}")
        print(f"\n{Colors.BOLD}Quick start:{Colors.RESET}")
        print(f"  python train_ltcc.py --mode train --gpu 0")
        print(f"  or")
        print(f"  ./quick_train.sh 0 64 120")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed. Please fix the issues above.{Colors.RESET}")
        print_installation_instructions()

    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
