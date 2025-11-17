#!/usr/bin/env python3
"""
Training Management Script for TransReID on LTCC Dataset

This script provides an easy-to-use interface for training TransReID on the LTCC dataset.
It handles configuration, training execution, model evaluation, and automatic archiving.

Each training run is automatically archived with:
- Training logs and tensorboard files
- Model checkpoints
- Configuration file copy
- Dataset information
- Training metadata (date, duration, parameters)

Usage:
    # Train with automatic archiving
    python train_ltcc.py --mode train --gpu 0

    # Train and evaluate
    python train_ltcc.py --mode train_eval --gpu 0,1 --batch_size 128

    # Evaluate existing checkpoint
    python train_ltcc.py --mode eval --checkpoint path/to/model.pth --gpu 0

    # View configuration info
    python train_ltcc.py --mode info

    # List all training archives
    python train_ltcc.py --mode list_archives
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime


class LTCCTrainer:
    """Training manager for TransReID on LTCC dataset"""

    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent
        self.transreid_dir = self.project_root / "TransReID"
        self.config_file = self.transreid_dir / "configs/LTCC/vit_transreid_stride.yml"
        self.train_script = self.transreid_dir / "train.py"
        self.test_script = self.transreid_dir / "test.py"

        # Archive configuration
        self.archive_root = Path("/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/archives")
        self.current_archive_dir = None
        self.training_start_time = None

        # Verify paths exist
        self._verify_setup()

    def _verify_setup(self):
        """Verify that all necessary files and directories exist"""
        if not self.transreid_dir.exists():
            raise FileNotFoundError(f"TransReID directory not found: {self.transreid_dir}")

        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        if not self.train_script.exists():
            raise FileNotFoundError(f"Train script not found: {self.train_script}")

        print("✓ Setup verified successfully")
        print(f"✓ Config file: {self.config_file}")
        print(f"✓ TransReID directory: {self.transreid_dir}")

    def _create_training_archive(self):
        """Create a new training archive directory"""
        # Create timestamp-based directory name
        self.training_start_time = datetime.now()
        timestamp = self.training_start_time.strftime("%Y%m%d_%H%M%S")

        # Create archive directory structure
        self.current_archive_dir = self.archive_root / f"train_{timestamp}"
        self.current_archive_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.current_archive_dir / "checkpoints").mkdir(exist_ok=True)
        (self.current_archive_dir / "logs").mkdir(exist_ok=True)
        (self.current_archive_dir / "config").mkdir(exist_ok=True)

        # Copy configuration file
        shutil.copy2(self.config_file, self.current_archive_dir / "config" / self.config_file.name)

        # Get dataset information
        dataset_info = self._get_dataset_info()

        # Create metadata file
        metadata = {
            "training_start": self.training_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_info,
            "configuration": {
                "config_file": str(self.config_file),
                "batch_size": self.args.batch_size,
                "max_epochs": self.args.max_epochs,
                "base_lr": self.args.base_lr,
                "gpu": self.args.gpu,
                "num_workers": self.args.num_workers,
                "eval_during_training": self.args.eval_during_training
            },
            "command_line_args": vars(self.args),
            "archive_location": str(self.current_archive_dir)
        }

        # Save metadata
        with open(self.current_archive_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create README
        self._create_archive_readme(metadata)

        print(f"\n✓ Training archive created: {self.current_archive_dir}")
        return self.current_archive_dir

    def _get_dataset_info(self):
        """Gather information about the dataset"""
        dataset_path = Path("/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/data")

        info = {
            "name": "LTCC (Long-Term Cloth-Changing)",
            "path": str(dataset_path),
            "exists": dataset_path.exists()
        }

        # Try to get dataset statistics if directory exists
        if dataset_path.exists():
            try:
                # Count train/test splits if they exist
                train_dir = dataset_path / "bounding_box_train"
                test_dir = dataset_path / "bounding_box_test"
                query_dir = dataset_path / "query"

                if train_dir.exists():
                    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
                    info["train_images"] = len(train_images)

                if test_dir.exists():
                    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                    info["gallery_images"] = len(test_images)

                if query_dir.exists():
                    query_images = list(query_dir.glob("*.jpg")) + list(query_dir.glob("*.png"))
                    info["query_images"] = len(query_images)
            except Exception as e:
                info["error"] = str(e)

        return info

    def _create_archive_readme(self, metadata):
        """Create a README file for the archive"""
        readme_content = f"""# Training Archive - {metadata['training_start']}

## Overview
This directory contains a complete archive of a TransReID training run on the LTCC dataset.

## Directory Structure
```
.
├── checkpoints/          # Model checkpoints saved during training
├── logs/                # Training logs and tensorboard files
├── config/              # Configuration files used
├── training_metadata.json  # Complete training metadata
└── README.md           # This file
```

## Training Configuration
- **Start Time**: {metadata['training_start']}
- **Dataset**: {metadata['dataset']['name']}
- **Batch Size**: {metadata['configuration']['batch_size']}
- **Max Epochs**: {metadata['configuration']['max_epochs']}
- **Base Learning Rate**: {metadata['configuration']['base_lr']}
- **GPU(s)**: {metadata['configuration']['gpu']}

## Dataset Information
- **Dataset Path**: {metadata['dataset']['path']}
- **Train Images**: {metadata['dataset'].get('train_images', 'N/A')}
- **Gallery Images**: {metadata['dataset'].get('gallery_images', 'N/A')}
- **Query Images**: {metadata['dataset'].get('query_images', 'N/A')}

## Configuration File
The original configuration file is stored in: `config/{Path(self.config_file).name}`

## Checkpoints
Model checkpoints are saved in the `checkpoints/` directory according to the checkpoint period specified in the configuration.

## Logs
Training logs, including console output and tensorboard files, are stored in the `logs/` directory.

## Reproduction
To reproduce this training run, use the configuration file in the `config/` directory with the parameters specified in `training_metadata.json`.
"""

        with open(self.current_archive_dir / "README.md", 'w') as f:
            f.write(readme_content)

    def _finalize_archive(self, success=True):
        """Finalize the training archive with completion information"""
        if not self.current_archive_dir or not self.current_archive_dir.exists():
            return

        training_end_time = datetime.now()
        duration = training_end_time - self.training_start_time

        # Update metadata with completion info
        metadata_file = self.current_archive_dir / "training_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata["training_end"] = training_end_time.strftime("%Y-%m-%d %H:%M:%S")
        metadata["training_duration"] = str(duration)
        metadata["training_success"] = success

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Training archive finalized: {self.current_archive_dir}")
        print(f"  Duration: {duration}")
        print(f"  Status: {'Success' if success else 'Failed'}")

    def _copy_checkpoints_to_archive(self):
        """Copy model checkpoints from logs directory to archive/checkpoints"""
        if not self.current_archive_dir:
            return

        logs_dir = self.current_archive_dir / "logs"
        checkpoints_dir = self.current_archive_dir / "checkpoints"

        if not logs_dir.exists():
            return

        # Find all .pth checkpoint files
        checkpoint_files = list(logs_dir.glob("**/*.pth"))

        if checkpoint_files:
            print(f"\n✓ Copying {len(checkpoint_files)} checkpoint(s) to archive...")
            for ckpt in checkpoint_files:
                dest = checkpoints_dir / ckpt.name
                shutil.copy2(ckpt, dest)
                print(f"  - {ckpt.name}")

    def _build_command(self, mode='train', checkpoint=None):
        """Build the training/testing command"""

        # Base command
        if mode in ['train', 'train_eval']:
            script = str(self.train_script)
        else:
            script = str(self.test_script)

        cmd = [sys.executable, script, "--config_file", str(self.config_file)]

        # Add device ID
        gpu_id = self.args.gpu
        cmd.extend(["MODEL.DEVICE_ID", f"('{gpu_id}')"])

        # Add custom parameters based on args
        if self.args.batch_size:
            cmd.extend(["SOLVER.IMS_PER_BATCH", str(self.args.batch_size)])

        if self.args.max_epochs:
            cmd.extend(["SOLVER.MAX_EPOCHS", str(self.args.max_epochs)])

        if self.args.base_lr:
            cmd.extend(["SOLVER.BASE_LR", str(self.args.base_lr)])

        # Use archive directory for training output if in training mode
        if mode in ['train', 'train_eval'] and self.current_archive_dir:
            output_dir = str(self.current_archive_dir / "logs")
            cmd.extend(["OUTPUT_DIR", output_dir])
        elif self.args.output_dir:
            cmd.extend(["OUTPUT_DIR", str(self.args.output_dir)])

        if self.args.num_workers is not None:
            cmd.extend(["DATALOADER.NUM_WORKERS", str(self.args.num_workers)])

        # Add checkpoint for evaluation
        if checkpoint:
            cmd.extend(["TEST.WEIGHT", str(checkpoint)])

        # Disable evaluation during training if specified
        if mode == 'train' and not self.args.eval_during_training:
            cmd.extend(["TEST.EVAL", "False"])

        return cmd

    def train(self):
        """Start training"""
        print("\n" + "="*70)
        print("Starting TransReID Training on LTCC Dataset")
        print("="*70)
        print(f"GPU: {self.args.gpu}")
        print(f"Batch Size: {self.args.batch_size}")
        print(f"Max Epochs: {self.args.max_epochs}")
        print(f"Base LR: {self.args.base_lr}")
        print("="*70 + "\n")

        # Create training archive
        self._create_training_archive()

        cmd = self._build_command(mode='train')

        print("Executing command:")
        print(" ".join(cmd))
        print("\n")

        # Change to TransReID directory and run
        original_dir = os.getcwd()
        success = False
        try:
            os.chdir(self.transreid_dir)
            result = subprocess.run(cmd, check=True)
            success = result.returncode == 0

            # Copy checkpoints to archive
            if success and self.current_archive_dir:
                self._copy_checkpoints_to_archive()

            return success
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error: {e}")
            success = False
            return False
        finally:
            os.chdir(original_dir)
            # Finalize archive
            self._finalize_archive(success=success)

    def evaluate(self, checkpoint=None):
        """Evaluate a trained model"""
        if checkpoint is None and self.args.checkpoint is None:
            print("❌ Error: No checkpoint specified for evaluation")
            print("Please provide checkpoint path with --checkpoint argument")
            return False

        checkpoint = checkpoint or self.args.checkpoint

        print("\n" + "="*70)
        print("Starting TransReID Evaluation on LTCC Dataset")
        print("="*70)
        print(f"Checkpoint: {checkpoint}")
        print(f"GPU: {self.args.gpu}")
        print("="*70 + "\n")

        cmd = self._build_command(mode='eval', checkpoint=checkpoint)

        print("Executing command:")
        print(" ".join(cmd))
        print("\n")

        # Change to TransReID directory and run
        original_dir = os.getcwd()
        try:
            os.chdir(self.transreid_dir)
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Evaluation failed with error: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def train_and_evaluate(self):
        """Train and then evaluate the model"""
        print("\n" + "="*70)
        print("Training and Evaluation Pipeline")
        print("="*70 + "\n")

        # Train
        success = self.train()
        if not success:
            print("\n❌ Training failed. Skipping evaluation.")
            return False

        print("\n" + "="*70)
        print("Training completed successfully!")
        print("="*70 + "\n")

        # Find the latest checkpoint from the archive
        if self.current_archive_dir:
            checkpoint_path = self._find_latest_checkpoint(str(self.current_archive_dir / "checkpoints"))
            # If no checkpoints in archive/checkpoints, try logs directory
            if not checkpoint_path:
                checkpoint_path = self._find_latest_checkpoint(str(self.current_archive_dir / "logs"))
        else:
            output_dir = self.args.output_dir or "/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/logs/ltcc_vit_transreid_stride"
            checkpoint_path = self._find_latest_checkpoint(output_dir)

        if checkpoint_path:
            print(f"\nFound checkpoint: {checkpoint_path}")
            print("Starting evaluation...\n")
            return self.evaluate(checkpoint=checkpoint_path)
        else:
            print("\n⚠ Warning: No checkpoint found for evaluation")
            return False

    def _find_latest_checkpoint(self, output_dir):
        """Find the latest checkpoint in the output directory"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return None

        checkpoints = list(output_path.glob("*.pth"))
        if not checkpoints:
            return None

        # Return the checkpoint with the highest epoch number
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])

    def print_config_info(self):
        """Print information about the current configuration"""
        print("\n" + "="*70)
        print("TransReID Training Configuration for LTCC")
        print("="*70)
        print(f"\nConfig File: {self.config_file}")
        print(f"\nDataset Path: /home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/data")
        print(f"\nArchive Directory: {self.archive_root}")
        print(f"Output Directory: {self.args.output_dir or 'Will be created in archive'}")
        print("\nModel Configuration:")
        print("  - Model: ViT-Base TransReID")
        print("  - Pretrained: ImageNet ViT-Base")
        print("  - Stride Size: [12, 12]")
        print("  - SIE Camera: Enabled")
        print("  - JPM: Enabled")
        print("\nTraining Configuration:")
        print(f"  - Batch Size: {self.args.batch_size}")
        print(f"  - Max Epochs: {self.args.max_epochs}")
        print(f"  - Base LR: {self.args.base_lr}")
        print(f"  - Optimizer: SGD")
        print(f"  - Loss: Triplet Loss")
        print("\nArchiving:")
        print("  Each training run will be archived with:")
        print("  - Training logs and checkpoints")
        print("  - Configuration copy")
        print("  - Dataset information")
        print("  - Training metadata (date, duration, parameters)")
        print("="*70 + "\n")

    def list_archives(self):
        """List all training archives"""
        if not self.archive_root.exists():
            print(f"\nNo archives found at {self.archive_root}")
            return

        archives = sorted(self.archive_root.glob("train_*"), reverse=True)

        if not archives:
            print(f"\nNo training archives found at {self.archive_root}")
            return

        print("\n" + "="*70)
        print("Training Archives")
        print("="*70 + "\n")

        for archive in archives:
            metadata_file = archive / "training_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                print(f"Archive: {archive.name}")
                print(f"  Start: {metadata.get('training_start', 'N/A')}")
                print(f"  End: {metadata.get('training_end', 'In Progress' if not metadata.get('training_end') else 'N/A')}")
                print(f"  Duration: {metadata.get('training_duration', 'N/A')}")
                print(f"  Status: {'✓ Success' if metadata.get('training_success') else '✗ Failed' if metadata.get('training_success') is False else 'Unknown'}")
                print(f"  Epochs: {metadata['configuration'].get('max_epochs', 'N/A')}")
                print(f"  Batch Size: {metadata['configuration'].get('batch_size', 'N/A')}")
                print(f"  Path: {archive}")
                print()
            else:
                print(f"Archive: {archive.name}")
                print(f"  (No metadata found)")
                print(f"  Path: {archive}")
                print()

        print("="*70 + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Training Management Script for TransReID on LTCC Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'train_eval', 'info', 'list_archives'],
        help='Operation mode: train, eval, train_eval, info, or list_archives'
    )

    # GPU configuration
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device ID (e.g., "0" or "0,1")'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=120,
        help='Maximum number of training epochs (default: 120)'
    )

    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.008,
        help='Base learning rate (default: 0.008)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of data loading workers (default: from config)'
    )

    # Evaluation parameters
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation'
    )

    parser.add_argument(
        '--eval_during_training',
        action='store_true',
        help='Enable evaluation during training'
    )

    # Output configuration
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for logs and checkpoints (default: from config)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Create trainer
    try:
        trainer = LTCCTrainer(args)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure that:")
        print("  1. TransReID is properly set up in the TransReid/ directory")
        print("  2. The LTCC config file exists at TransReID/configs/LTCC/vit_transreid_stride.yml")
        print("  3. The dataset is located at ReID_Experiments/LTCC_ReID/data/")
        return 1

    # Execute based on mode
    if args.mode == 'info':
        trainer.print_config_info()
        return 0
    elif args.mode == 'list_archives':
        trainer.list_archives()
        return 0
    elif args.mode == 'train':
        success = trainer.train()
    elif args.mode == 'eval':
        success = trainer.evaluate()
    elif args.mode == 'train_eval':
        success = trainer.train_and_evaluate()
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return 1

    if success:
        print("\n✓ Operation completed successfully!")
        return 0
    else:
        print("\n❌ Operation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
