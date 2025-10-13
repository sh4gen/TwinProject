# Synthetic Data Enhanced Multi-Camera Intruder Detection Using Edge AI

[![NVIDIA](https://img.shields.io/badge/NVIDIA-TAO_Toolkit-76B900?logo=nvidia)](https://developer.nvidia.com/tao-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)

> **A real-time multi-camera intruder detection system optimized for edge devices using person detection and re-identification.**

⚠️ **NOTE**: This is a messy experiment code saving repository, not a plug-and-play Re-ID solution. The code represents research experiments and may require significant adaptation for your use case.

This research project develops an Edge AI-powered surveillance system that can track and re-identify individuals across multiple camera views. The system combines state-of-the-art object detection (YOLOv11) with person re-identification models, optimized for deployment on resource-constrained edge devices like NVIDIA Jetson.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Highlights](#research-highlights)
- [System Architecture](#system-architecture)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This research introduces a multi-camera intruder detection system powered by Edge AI. The core challenge in surveillance is tracking individuals as they move between camera fields of view. Our solution:

1. **Person Detection**: Uses YOLOv11n optimized with TensorRT for real-time detection
2. **Person Re-Identification (Re-ID)**: Employs deep learning to match individuals across cameras
3. **Edge Deployment**: Optimized for NVIDIA Jetson and similar edge devices
4. **Synthetic Data Enhancement**: Incorporates synthetic datasets to improve model robustness

### Key Contributions

- ✅ **mAP×FPS Metric**: Novel metric balancing accuracy and throughput for edge AI model selection
- ✅ **Comprehensive Dataset Analysis**: Evaluation of cloth-changing and synthetic Re-ID datasets
- ✅ **TensorRT Optimization**: Average 2.69x speedup across YOLO architectures
- ✅ **Gallery/Query System**: Persistent tracking across camera views with feature storage

---

## 🔬 Research Highlights

### Object Detection Performance

| Model | mAP@50 | TensorRT FPS | mAP×FPS Score | Inference (ms) |
|-------|--------|--------------|---------------|----------------|
| **YOLOv11n** | 60.29% | 159.76 | **96.32** | 6.25 |
| YOLOv10n | 56.78% | 167.29 | 94.99 | 5.97 |
| YOLOv11s | 63.20% | 145.90 | 92.20 | 6.85 |
| YOLOv8x | 67.02% | 72.70 | 48.73 | 13.75 |

**Hardware**: NVIDIA GeForce RTX 3050 Mobile (4GB VRAM, 60W)

### Re-Identification Results

#### PRCC Dataset (Cloth-Changing)
- **mAP**: 61.2%
- **Rank-1**: 81.7%
- **Rank-5**: 90.0%
- **Training Time**: 56 minutes (60 epochs)

#### LTCC Dataset (Long-Term Cloth-Changing)
- **mAP**: 23.6%
- **Rank-1**: 46.7%
- **Rank-5**: 61.9%
- **Comparison**: Competitive with UCAS baseline (29.4% mAP)

#### ULI-RI Synthetic Dataset
- **Observation**: Near-perfect scores (97.7% mAP, 100% Rank-1) indicate dataset limitations
- **Conclusion**: Current synthetic data quality insufficient for real-world Re-ID challenges

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-Camera System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Camera 1 ──┐                                                   │
│  Camera 2 ──┼──► YOLOv11n ──► Person Re-ID ──► Gallery/Query    │
│  Camera N ──┘    Detection     Network          System          │
│                                                                 │
│              ┌──────────────┐  ┌──────────────┐                 │
│              │  TensorRT    │  │  Feature     │                 │
│              │  Optimized   │  │  Extraction  │                 │
│              └──────────────┘  └──────────────┘                 │
│                                                                 │
│                          ▼                                      │
│                  ┌───────────────┐                              │
│                  │  Persistent   │                              │
│                  │   Tracking    │                              │
│                  └───────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
1. **Detection Module**: YOLOv11n for real-time person detection
2. **Re-ID Module**: ResNet50-based feature extraction
3. **Gallery System**: Persistent storage of person embeddings
4. **Query System**: Real-time matching against gallery
5. **Tracking**: Cross-camera identity association

---

## 📊 Datasets

### Supported Datasets

#### 1. **PRCC** (Person Re-identification with Clothes Changing)
- **Images**: 33,769
- **Identities**: 221
- **Cameras**: 3
- **Focus**: Clothing change scenarios
- **Best Performance**: 81.7% Rank-1 accuracy

#### 2. **LTCC** (Long-Term Cloth-Changing)
- **Images**: 17,119
- **Identities**: 14
- **Cameras**: 12
- **Focus**: Long-term appearance changes
- **Challenge**: High diversity in capture angles and lighting

#### 3. **CCVID** (Cloth-Changing Video-based)
- **Images**: 347,833
- **Identities**: 226
- **Cameras**: 12
- **Challenge**: Large-scale dataset with memory constraints

#### 4. **ULI-RI** (Synthetic Dataset)
- **Images**: 58,880
- **Identities**: 115
- **Cameras**: 8
- **Generation**: Unreal Engine
- **Limitation**: Low realism for Re-ID tasks

### Dataset Format

All datasets are converted to **Market-1501 format**:

```
dataset/
├── bounding_box_train/
│   ├── 0001_c1s1_001051_00.jpg
│   ├── 0001_c1s1_001052_00.jpg
│   └── ...
├── bounding_box_test/  (gallery)
│   └── ...
└── query/
    └── ...
```

**Filename Convention**: `{person_id}_c{camera_id}s{sequence_id}_{frame_number}_{bbox_id}.jpg`

### Dataset Preparation

```bash
# Example: Prepare PRCC dataset
cd Dataset_Utils
python prepare_prcc.py

# Analyze dataset quality
python Analyse_dataset.py --dataset_path /path/to/dataset
```

---

## 🎓 Training

### Using TAO Toolkit

#### 1. Configure Training

Create a YAML configuration file (e.g., `reid_config.yaml`):

```yaml
model:
  backbone: "resnet50"
  pretrained: true

dataset:
  train_dataset: "/path/to/data/bounding_box_train"
  query_dataset: "/path/to/data/query"
  test_dataset: "/path/to/data/bounding_box_test"
  num_classes: 221  # Number of identities in training set
  pixel_mean: [0.444, 0.438, 0.457]
  pixel_std: [0.288, 0.280, 0.275]

train:
  num_epochs: 60
  batch_size: 64
  learning_rate: 0.00035
  optimizer: "SGD"
  weight_decay: 0.0005

augmentation:
  padding: 10
  random_crop_prob: 0.6
  random_erase_prob: 0.6
```

#### 2. Start Training

```bash
# Using TAO Toolkit
tao model re_identification train -e reid_config.yaml

# Or using pipeline
python ReID_Pipeline/pipeline.py
```

#### 3. Monitor Progress

```bash
# Training logs and checkpoints saved to:
# results/train/
#   ├── model_epoch_009_step_XXX.pth
#   ├── model_epoch_019_step_XXX.pth
#   └── status.json
```

### Training Tips

- **Batch Size**: Use 64 for 12GB GPU, 32 for 4GB GPU
- **Learning Rate**: Start with 0.00035, reduce if loss doesn't converge
- **Epochs**: 60 epochs typically sufficient for cloth-changing datasets
- **Validation**: Run every 10 epochs to track performance

---

## 📈 Evaluation

### Using TAO Toolkit

```bash
tao model re_identification evaluate \
    -e reid_config.yaml \
    evaluate.checkpoint=/path/to/model.pth \
    evaluate.query_dataset=/path/to/query \
    evaluate.test_dataset=/path/to/gallery \
    re_ranking.re_ranking=true
```

### Using Custom Evaluation Pipeline

```python
from ReID_Pipeline.Pipes.EvaluateTAO import EvaluatePipeTAO

evaluator = EvaluatePipeTAO(
    config_file="reid_config.yaml",
    checkpoint_dir="results/train",
    query_dir="data/query",
    gallery_dir="data/bounding_box_test",
    results_dir="evaluation_results",
    use_rerank=True
)

evaluator.run()
```

### Metrics

- **mAP** (mean Average Precision): Overall retrieval accuracy
- **Rank-1**: Percentage where correct match is top result
- **Rank-5**: Percentage where correct match is in top 5
- **Rank-10**: Percentage where correct match is in top 10

### Results Interpretation

```
╒════════════════════╤═════════╕
│ Name               │ Score   │
╞════════════════════╪═════════╡
│ mAP                │ 61.2%   │
├────────────────────┼─────────┤
│ CMC curve, Rank-1  │ 81.7%   │
├────────────────────┼─────────┤
│ CMC curve, Rank-5  │ 90.0%   │
├────────────────────┼─────────┤
│ CMC curve, Rank-10 │ 94.4%   │
╘════════════════════╧═════════╛
```

---

## 🚢 Deployment

### Export to ONNX

```bash
tao model re_identification export \
    -e export_config.yaml \
    export.checkpoint=/path/to/model.pth \
    export.onnx_file=reid_model.onnx
```

### TensorRT Optimization

```python
from ultralytics import YOLO

# Convert YOLO to TensorRT
model = YOLO('yolo11n.pt')
model.export(format='engine', device=0)  # Creates yolo11n.engine

# Use TensorRT engine
model = YOLO('yolo11n.engine')
results = model(image)
```

### Edge Deployment (NVIDIA Jetson)

```bash
# On Jetson device
pip install ultralytics opencv-python
python ReID_Inference/pipeline.py \
    --yolo_model yolo11n.engine \
    --reid_model reid_model.onnx \
    --input /dev/video0 \
    --output output.mp4
```

### Docker Deployment

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "ReID_Inference/pipeline.py"]
```

---

## 📊 Results

### Detection Performance

**YOLOv11n on WiderPerson Dataset:**
- **mAP@50**: 60.29%
- **FPS (PyTorch)**: 83.87
- **FPS (TensorRT)**: 159.76
- **Speedup**: 1.90x
- **Inference Time**: 6.25ms

**Comparison with Other Models:**

| Model | Parameters | mAP@50 | TensorRT FPS | mAP×FPS |
|-------|-----------|--------|--------------|---------|
| YOLOv11n | 2.6M | 60.29% | 159.76 | **96.32** |
| YOLOv10n | 2.3M | 56.78% | 167.29 | 94.99 |
| YOLOv11s | 9.4M | 63.20% | 145.90 | 92.20 |
| YOLOv8x | 68.2M | 67.02% | 72.70 | 48.73 |

### Re-Identification Performance

#### Cross-Dataset Evaluation

**Model trained on LTCC, tested on PRCC:**

| Train Dataset | Test Dataset | mAP | Rank-1 | Rank-5 |
|--------------|--------------|-----|--------|--------|
| LTCC | LTCC | 23.6% | 46.7% | 61.9% |
| PRCC | PRCC | 61.2% | 81.7% | 90.0% |
| 3-Dataset Combined | LTCC | 23.9% | 47.3% | 62.1% |
| 3-Dataset Combined | PRCC | 56.6% | 76.1% | 81.7% |

#### Synthetic Data Analysis

**ULI-RI Performance:**
- **Standalone Training**: Poor generalization (5.0% mAP on LTCC)
- **Combined Training**: No improvement over real data alone
- **Conclusion**: Quality limitations prevent effective use

### Key Findings

1. **Throughput is King**: For edge AI, balanced mAP×FPS score outperforms raw accuracy
2. **Fine-tuning Works**: 137% improvement on PRCC with proper hyperparameters
3. **Synthetic Data Challenge**: Current synthetic datasets insufficient for Re-ID
4. **Hardware Matters**: Large datasets (CCVID) require high-end GPUs (12GB+ VRAM)

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@techreport{acar2025synthetic,
  title={Synthetic Data Enhanced Multi-Camera Intruder Detection Using Edge AI},
  author={Acar, İsmail Kağan and Berbergil, Aşkın Ali},
  year={2025},
  institution={NVIDIA Academic Grant Program}
}
```

---

## 🙏 Acknowledgments

This research was conducted within the scope of **Synthetic Data Enhanced Multi-Camera Intruder Detection Using Edge AI**, supported by the **NVIDIA Academic Grant Program**.

**Hardware Support:**
- NVIDIA Academic Grant: 4× RTX PRO 6000 Blackwell Max-Q Workstation Edition Graphics Cards (pending delivery at time of publication)
- Development Hardware: NVIDIA GeForce RTX 3050 Mobile (4GB VRAM)

**Frameworks & Tools:**
- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [TensorRT](https://developer.nvidia.com/tensorrt)

**Datasets:**
- PRCC: Person Re-identification with Clothes Changing
- LTCC: Long-Term Cloth-Changing Dataset
- CCVID: Cloth-Changing Video-based Dataset
- ULI-RI: Unreal Engine Synthetic Dataset
- WiderPerson: Pedestrian Detection Dataset

---

## 📄 License

This project is licensed under the Academic Research License. See `LICENSE` file for details.

**Note**: This research is intended for academic and research purposes. Commercial use requires additional licensing considerations.

---

## 🔗 Links

- **Research Report**: [Progress_Report.pdf](docs/Progress_Report.pdf)
- **NVIDIA TAO Toolkit**: https://developer.nvidia.com/tao-toolkit
- **YOLOv11 Documentation**: https://docs.ultralytics.com/

---

**Authors**: 
- [İsmail Kağan Acar](https://github.com/ikaganacar1) 
- [Aşkın Ali Berbergil](https://github.com/sh4gen)

**Last Updated**: August 23, 2025

---

<div align="center">
  <sub>Built with ❤️ for the computer vision community</sub>
</div>
