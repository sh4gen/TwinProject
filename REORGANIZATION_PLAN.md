# TwinProject Repository Reorganization Plan

## Current Issues

1. **Root Directory Clutter**: Model weight file (`yolo11n.pt`) in root
2. **Inconsistent Naming**: Mixed case conventions (`ReID_Experiments` vs `datasets`)
3. **Scattered Components**: Pipeline code in multiple locations
4. **Unclear Structure**: Hard to understand project organization
5. **Cache Files**: `__pycache__` directories not ignored
6. **Mixed Content**: Experiments, inference, and pipelines mixed together

## Proposed New Structure

```
TwinProject/
├── docs/                          # All documentation
│   ├── README.md                  # Main project documentation
│   ├── ARCHITECTURE.md            # System architecture
│   ├── SETUP.md                   # Installation & setup guide
│   └── CONTRIBUTING.md            # Contribution guidelines
│
├── models/                        # Pre-trained model weights
│   ├── detection/                 # Object detection models
│   │   └── yolo11n.pt
│   └── reid/                      # ReID models
│       └── .gitkeep
│
├── src/                           # Source code
│   ├── detection/                 # Object detection module
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   │
│   ├── reid/                      # ReID module
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   └── matching.py
│   │
│   ├── pipeline/                  # Main pipeline implementation
│   │   ├── __init__.py
│   │   ├── pipeline.py            # From ReID_Pipeline
│   │   └── inference.py           # From ReID_Inference
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       └── dataset_utils.py       # From Dataset_Utils
│
├── experiments/                   # Training experiments & results
│   ├── object_detection/
│   │   └── widerperson/           # From Object_Detection/WiderPersonYOLO
│   │       ├── configs/
│   │       ├── notebooks/         # .ipynb files
│   │       ├── results/
│   │       └── docs/              # evaluation reports
│   │
│   └── reid/                      # From ReID_Experiments
│       ├── ltcc/                  # LTCC_ReID
│       │   ├── configs/
│       │   ├── data/
│       │   ├── results/
│       │   └── archives/
│       ├── prcc/                  # PRCC
│       ├── uliri/                 # ULIRI
│       └── ccvid/                 # CCVID
│
├── training/                      # Training frameworks & scripts
│   └── transreid/                 # From TransReid/
│       ├── TransReID/             # Original repo
│       ├── train_ltcc.py
│       ├── check_setup.py
│       ├── LTCC_TRAINING_GUIDE.md
│       └── ARCHIVING_GUIDE.md
│
├── datasets/                      # Dataset-related code & metadata
│   ├── posetrack/                 # Keep as is
│   └── README.md                  # Dataset documentation
│
├── notebooks/                     # Exploratory notebooks
│   ├── detection/                 # From Object_Detection/*.ipynb
│   ├── reid/                      # From ReID_Pipeline/*.ipynb
│   └── README.md
│
├── tests/                         # Unit tests (to be created)
│   └── __init__.py
│
├── scripts/                       # Utility scripts
│   └── setup_environment.sh
│
├── .gitignore                     # Updated gitignore
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # Updated main README
```

## Migration Steps

### Phase 1: Preparation
1. Create new directory structure
2. Update .gitignore
3. Create placeholder files

### Phase 2: Move Files
1. Move model weights to `models/`
2. Consolidate source code to `src/`
3. Reorganize experiments
4. Move notebooks
5. Organize documentation

### Phase 3: Updates
1. Update import paths in Python files
2. Update documentation
3. Create new comprehensive README
4. Test that code still works

### Phase 4: Cleanup
1. Remove old empty directories
2. Remove __pycache__ directories
3. Clean up redundant files

## Benefits

1. **Clear Separation**: Source code, experiments, and training separate
2. **Better Discoverability**: Easy to find components
3. **Scalability**: Easy to add new models/experiments
4. **Professional**: Industry-standard structure
5. **Maintainability**: Easier to maintain and extend
6. **Documentation**: Centralized docs directory

## Files to Move

### To `models/detection/`
- `yolo11n.pt`

### To `src/`
- `ReID_Inference/*.py` → `src/pipeline/`
- `ReID_Pipeline/pipeline.py` → `src/pipeline/`
- `Dataset_Utils/*` → `src/utils/`
- Extract detection code → `src/detection/`
- Extract reid code → `src/reid/`

### To `notebooks/`
- `Object_Detection/*.ipynb`
- `ReID_Pipeline/*.ipynb`

### To `experiments/`
- `Object_Detection/WiderPersonYOLO/` → `experiments/object_detection/widerperson/`
- `ReID_Experiments/LTCC_ReID/` → `experiments/reid/ltcc/`
- `ReID_Experiments/PRCC/` → `experiments/reid/prcc/`
- `ReID_Experiments/ULIRI/` → `experiments/reid/uliri/`
- `ReID_Experiments/CCVID/` → `experiments/reid/ccvid/`

### To `training/`
- `TransReid/` → `training/transreid/`

### Keep in Place
- `datasets/posetrack/`
- `.git/`

## Next Steps

1. Review this plan
2. Backup current state (git commit)
3. Execute migration in phases
4. Test after each phase
5. Update all documentation
