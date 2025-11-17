# TransReID Training Guide for LTCC Dataset

This guide explains how to train the TransReID model on your LTCC (Long-Term Cloth-Changing) Person Re-Identification dataset.

## Setup Complete ✓

The following components have been set up for you:

1. **LTCC Dataset Class**: [TransReID/datasets/ltcc.py](TransReID/datasets/ltcc.py)
   - Handles LTCC dataset structure
   - Extracts person IDs and camera IDs from filename format: `XXXX_cYYsZ_XX_XX.jpg`

2. **Configuration File**: [TransReID/configs/LTCC/vit_transreid_stride.yml](TransReID/configs/LTCC/vit_transreid_stride.yml)
   - Pre-configured for LTCC dataset
   - Uses ViT-Base with TransReID enhancements
   - Optimized hyperparameters

3. **Training Management Script**: [train_ltcc.py](train_ltcc.py)
   - Easy-to-use Python interface
   - Handles training, evaluation, and configuration

## Prerequisites

### 1. Download Pre-trained Model

You need to download the ImageNet pre-trained ViT-Base model:

```bash
mkdir -p ~/.cache/torch/checkpoints
cd ~/.cache/torch/checkpoints
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

### 2. Verify Dataset Structure

Your LTCC dataset should be organized as follows:

```
ReID_Experiments/LTCC_ReID/data/
├── bounding_box_train/  (training images)
├── query/               (query images)
└── bounding_box_test/   (gallery/test images)
```

The filename format should be: `XXXX_cYYsZ_XX_XX.jpg`
- `XXXX`: Person ID
- `YY`: Camera ID
- `Z`: Session/Sequence ID

### 3. Install Dependencies

Ensure you have the required packages:

```bash
cd TransReID
pip install -r requirements.txt
```

Required packages include:
- PyTorch >= 1.6
- torchvision >= 0.7
- timm == 0.3.2
- yacs
- opencv-python

## Quick Start

### View Configuration Info

```bash
python train_ltcc.py --mode info
```

This displays the current configuration without starting training.

### Basic Training

Start training with default settings:

```bash
python train_ltcc.py --mode train --gpu 0
```

### Training with Custom Parameters

```bash
python train_ltcc.py --mode train \
    --gpu 0 \
    --batch_size 64 \
    --max_epochs 120 \
    --base_lr 0.008
```

### Training with Evaluation

Train and then automatically evaluate on the best checkpoint:

```bash
python train_ltcc.py --mode train_eval --gpu 0
```

### Evaluation Only

Evaluate a trained model:

```bash
python train_ltcc.py --mode eval \
    --checkpoint /path/to/checkpoint.pth \
    --gpu 0
```

## Training Parameters

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | Operation mode: `train`, `eval`, `train_eval`, or `info` |
| `--gpu` | `0` | GPU device ID (e.g., "0" or "0,1") |
| `--batch_size` | `64` | Batch size for training |
| `--max_epochs` | `120` | Maximum number of training epochs |
| `--base_lr` | `0.008` | Base learning rate |
| `--num_workers` | `8` | Number of data loading workers |
| `--checkpoint` | `None` | Path to checkpoint for evaluation |
| `--output_dir` | Config default | Custom output directory for logs |
| `--eval_during_training` | `False` | Enable periodic evaluation during training |

## Model Configuration

The default configuration uses:

- **Architecture**: ViT-Base (Vision Transformer)
- **Input Size**: 256x128 pixels
- **Stride Size**: [12, 12] (for more fine-grained features)
- **Special Features**:
  - **SIE (Side Information Embedding)**: Camera information embedding
  - **JPM (Jigsaw Patch Module)**: Enhanced feature learning
- **Loss Function**: Triplet loss with softmax
- **Optimizer**: SGD with learning rate 0.008
- **Training Schedule**: 120 epochs with warmup

## Output Structure

Training outputs will be saved to:

```
ReID_Experiments/LTCC_ReID/logs/ltcc_vit_transreid_stride/
├── transformer_20.pth    (checkpoint at epoch 20)
├── transformer_40.pth
├── ...
├── transformer_120.pth   (final checkpoint)
└── log.txt              (training log)
```

## Monitoring Training

During training, you'll see:
- Loss values (ID loss, Triplet loss, Total loss)
- Training accuracy
- Learning rate schedule
- Checkpoint saving notifications

Example output:
```
Epoch[1] Iteration[50/150] Loss: 6.123 (6.234), Acc: 0.123 (0.112), Base Lr: 0.00016
```

## Advanced Usage

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
# Modify the config to enable distributed training
python train_ltcc.py --mode train --gpu 0,1
```

Note: Multi-GPU training requires additional configuration in the YAML file.

### Custom Output Directory

```bash
python train_ltcc.py --mode train \
    --output_dir /custom/path/to/logs \
    --gpu 0
```

### Modify Learning Rate Schedule

Edit the configuration file directly:
[TransReID/configs/LTCC/vit_transreid_stride.yml](TransReID/configs/LTCC/vit_transreid_stride.yml)

```yaml
SOLVER:
  OPTIMIZER_NAME: 'SGD'
  MAX_EPOCHS: 120
  BASE_LR: 0.008      # Adjust learning rate
  WARMUP_EPOCHS: 5    # Warmup period
  CHECKPOINT_PERIOD: 20  # Save checkpoint every N epochs
  EVAL_PERIOD: 20     # Evaluate every N epochs
```

## Expected Performance

For LTCC dataset, TransReID typically achieves:
- **Training time**: ~1-2 hours per epoch on V100 GPU (depending on dataset size)
- **GPU Memory**: ~12GB for stride [12,12], ~7GB for stride [16,16]
- **Convergence**: Usually converges within 80-120 epochs

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python train_ltcc.py --mode train --batch_size 32 --gpu 0
```

Or modify stride size in config to [16, 16] (uses less memory):
```yaml
MODEL:
  STRIDE_SIZE: [16, 16]
```

### Issue: Pre-trained Model Not Found

**Solution**: Download the pre-trained model as shown in Prerequisites section.

### Issue: Dataset Not Found

**Solution**: Verify your dataset path in the config file:
```yaml
DATASETS:
  ROOT_DIR: ('/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/data')
```

### Issue: Slow Training

**Solutions**:
1. Increase number of workers: `--num_workers 16`
2. Enable AMP (Automatic Mixed Precision) - already enabled in TransReID
3. Ensure data is on SSD rather than HDD

## Next Steps

After training:

1. **Evaluate the model**:
   ```bash
   python train_ltcc.py --mode eval \
       --checkpoint logs/ltcc_vit_transreid_stride/transformer_120.pth \
       --gpu 0
   ```

2. **Test with re-ranking** (improves accuracy but slower):
   - Edit config file and set `TEST.RE_RANKING: True`

3. **Fine-tune hyperparameters**:
   - Adjust learning rate
   - Try different batch sizes
   - Experiment with stride sizes

4. **Export model for inference**:
   - The trained `.pth` files can be loaded for inference
   - See TransReID test.py for inference examples

## References

- **TransReID Paper**: [arXiv:2102.04378](https://arxiv.org/abs/2102.04378)
- **LTCC Dataset**: Long-Term Cloth-Changing Person Re-ID
- **Original TransReID Code**: [GitHub Repository](https://github.com/damo-cv/TransReID)

## Support

If you encounter issues:
1. Check the log file in the output directory
2. Verify dataset structure and file naming
3. Ensure all dependencies are installed correctly
4. Check GPU memory availability

Happy training! 🚀
