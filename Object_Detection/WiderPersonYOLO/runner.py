# simple_yolov8_evaluation.py
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
from datetime import datetime

def evaluate_trt(model_name,model_path, data_yaml, output_dir="results"):
    """Simple evaluation of YOLOv8 on WiderPerson dataset."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    is_engine = str(model_path).endswith('.engine')
    
    # Load model
    model = YOLO(model_path, task='detect')
    
    # Load test images
    dataset_root = Path(data_yaml).parent
    with open(dataset_root / 'test.txt', 'r') as f:
        test_images = [str(dataset_root / line.strip()) for line in f if line.strip()]
    test_images = [p for p in test_images if Path(p).exists()]
    
    metrics = {'model': str(model_path), 'device': str(device), 'is_tensorrt': is_engine}
    
    # 1. Get mAP metrics (only for .pt models)
    if not is_engine:
        val_results = model.val(data=data_yaml, imgsz=640, batch=16, device=device, 
                               conf=0.001, iou=0.5, verbose=False)
        metrics['mAP'] = {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr)
        }
    
    # 2. Measure inference speed
    # Detect batch size for TensorRT
    batch_size = 16
    if is_engine:
        try:
            _ = model.predict(test_images[0], imgsz=640, verbose=False)
        except AssertionError as e:
            if "max model size" in str(e):
                import re
                match = re.search(r"max model size KATEX_INLINE_OPEN(\d+),", str(e))
                if match:
                    batch_size = int(match.group(1))
    
    # Warmup
    for _ in range(5):
        if batch_size > 1:
            batch = test_images[:batch_size] if len(test_images) >= batch_size else test_images * batch_size
            _ = model.predict(batch[:batch_size], imgsz=640, verbose=False)
        else:
            _ = model.predict(test_images[0], imgsz=640, verbose=False)
    
    # Time inference
    inference_times = []
    sample_size = min(500, len(test_images))
    
    if batch_size > 1:
        # Batch processing
        for i in range(0, sample_size, batch_size):
            batch = test_images[i:i+batch_size]
            if len(batch) < batch_size:
                batch = batch + [batch[-1]] * (batch_size - len(batch))
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            _ = model.predict(batch, imgsz=640, verbose=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            batch_time = time.perf_counter() - start
            inference_times.extend([batch_time/batch_size] * min(len(test_images[i:i+batch_size]), batch_size))
    else:
        # Single image processing
        for img in test_images[:sample_size]:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            _ = model.predict(img, imgsz=640, verbose=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_times.append(time.perf_counter() - start)
    
    # Calculate speed metrics
    times_ms = np.array(inference_times) * 1000
    metrics['speed'] = {
        'batch_size': batch_size,
        'mean_ms': float(np.mean(times_ms)),
        'std_ms': float(np.std(times_ms)),
        'fps': float(1000 / np.mean(times_ms))
    }
    
    # 3. Generate predictions
    pred_dir = output_dir / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    total_detections = 0
    
    if batch_size > 1:
        # Batch predictions
        for i in tqdm(range(0, len(test_images), batch_size), desc="Predictions"):
            batch = test_images[i:i+batch_size]
            batch_names = [Path(p).stem for p in batch]
            
            if len(batch) < batch_size:
                batch = batch + [batch[-1]] * (batch_size - len(batch))
            
            results = model.predict(batch, imgsz=640, conf=0.001, verbose=False)
            
            for result, name in zip(results[:len(batch_names)], batch_names):
                with open(pred_dir / f"{name}.txt", 'w') as f:
                    if result.boxes is not None:
                        xyxy = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        f.write(f"{len(xyxy)}\n")
                        total_detections += len(xyxy)
                        for box, conf in zip(xyxy, confs):
                            f.write(f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f} {conf:.6f}\n")
                    else:
                        f.write("0\n")
    else:
        # Single predictions
        for img_path in tqdm(test_images, desc="Predictions"):
            results = model.predict(img_path, imgsz=640, conf=0.001, verbose=False)
            
            with open(pred_dir / f"{Path(img_path).stem}.txt", 'w') as f:
                if results[0].boxes is not None:
                    xyxy = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    f.write(f"{len(xyxy)}\n")
                    total_detections += len(xyxy)
                    for box, conf in zip(xyxy, confs):
                        f.write(f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f} {conf:.6f}\n")
                else:
                    f.write("0\n")
    
    metrics['predictions'] = {
        'output_dir': str(pred_dir),
        'total_detections': total_detections,
        'avg_per_image': total_detections / len(test_images)
    }
    
    # Save results
    with open(output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

