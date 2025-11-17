# Training Archive System

## Overview

The TransReID training script now includes an automatic archiving system that creates a complete, organized record of each training run. This ensures reproducibility and makes it easy to track and compare different experiments.

## Archive Structure

Each training run creates a timestamped archive directory with the following structure:

```
archives/
└── train_YYYYMMDD_HHMMSS/
    ├── checkpoints/          # Model checkpoints (.pth files)
    ├── logs/                 # Training logs and TensorBoard files
    ├── config/               # Copy of configuration file used
    ├── training_metadata.json # Complete training metadata
    └── README.md             # Human-readable archive summary
```

## What's Archived

Each archive contains:

1. **Checkpoints**: All model checkpoints saved during training
2. **Logs**: Complete training logs including TensorBoard files
3. **Configuration**: Exact copy of the configuration file used
4. **Metadata**: JSON file with:
   - Training start/end time
   - Training duration
   - Dataset information (path, number of images)
   - All training parameters (batch size, epochs, learning rate, etc.)
   - Command line arguments
   - Success/failure status

## Usage

### Start a Training Run

Training runs are automatically archived:

```bash
# Basic training
python train_ltcc.py --mode train --gpu 0

# Training with custom parameters
python train_ltcc.py --mode train --gpu 0 --batch_size 128 --max_epochs 100
```

The archive directory will be created at:
```
/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/archives/train_YYYYMMDD_HHMMSS/
```

### List All Archives

View all training archives with their metadata:

```bash
python train_ltcc.py --mode list_archives
```

This shows:
- Archive name and timestamp
- Training start/end time
- Duration
- Success/failure status
- Key parameters (epochs, batch size)
- Full path to archive

### View Training Info

See what will be archived in the next training run:

```bash
python train_ltcc.py --mode info
```

## Archive Contents

### training_metadata.json

Complete JSON record of the training run:

```json
{
  "training_start": "2025-11-17 15:30:00",
  "training_end": "2025-11-17 18:45:30",
  "training_duration": "3:15:30",
  "training_success": true,
  "dataset": {
    "name": "LTCC (Long-Term Cloth-Changing)",
    "path": "/path/to/dataset",
    "train_images": 9576,
    "gallery_images": 493,
    "query_images": 493
  },
  "configuration": {
    "batch_size": 64,
    "max_epochs": 120,
    "base_lr": 0.008,
    "gpu": "0"
  }
}
```

### README.md

Each archive includes a human-readable README with:
- Training overview
- Directory structure explanation
- Configuration details
- Dataset statistics
- Instructions for reproduction

## Benefits

1. **Reproducibility**: Every training run is fully documented with exact parameters and configuration
2. **Comparison**: Easy to compare different runs using metadata
3. **Organization**: All related files (logs, checkpoints, configs) in one place
4. **Traceability**: Know exactly when training ran, how long it took, and what parameters were used
5. **Recovery**: If training fails, you know exactly what was attempted

## Examples

### Example 1: Basic Training

```bash
python train_ltcc.py --mode train --gpu 0 --batch_size 64 --max_epochs 120
```

Creates archive at:
```
archives/train_20251117_153000/
├── checkpoints/
│   ├── checkpoint_epoch_20.pth
│   ├── checkpoint_epoch_40.pth
│   └── ...
├── logs/
│   ├── transreid.log
│   └── events.out.tfevents...
├── config/
│   └── vit_transreid_stride.yml
├── training_metadata.json
└── README.md
```

### Example 2: Finding Best Checkpoint

```bash
# List all archives to find the best training run
python train_ltcc.py --mode list_archives

# Navigate to the archive
cd archives/train_20251117_153000/checkpoints/

# Use the checkpoint for evaluation
python train_ltcc.py --mode eval --checkpoint archives/train_20251117_153000/checkpoints/best_model.pth
```

### Example 3: Reproducing a Training Run

```bash
# Find the archive of interest
python train_ltcc.py --mode list_archives

# Check the metadata
cat archives/train_20251117_153000/training_metadata.json

# Reproduce using the same config
python train_ltcc.py --mode train \
  --gpu 0 \
  --batch_size 64 \
  --max_epochs 120 \
  --base_lr 0.008
```

## Archive Location

Default location: `/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/archives/`

Archives are named with timestamp: `train_YYYYMMDD_HHMMSS`

## Tips

1. **Regular Cleanup**: Archives can take significant disk space. Periodically review and remove unsuccessful runs.

2. **Backup Important Runs**: Copy successful training archives to a backup location for long-term storage.

3. **Use Metadata**: The JSON metadata makes it easy to write scripts to analyze and compare multiple training runs.

4. **Documentation**: The auto-generated README in each archive is helpful when sharing results with others.

5. **Checkpoints**: Checkpoints are duplicated (in both `logs/` and `checkpoints/`) for convenience. The `checkpoints/` directory is the canonical location.

## Notes

- Archives are created at the start of training
- Metadata is updated when training completes (or fails)
- If training is interrupted, the archive will exist but may be marked as incomplete
- The system automatically handles directory creation and organization
