import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

class ObjectDetector:
    
    def __init__(self, 
                 model_path: str = 'yolo11n.pt',
                 device: str = 'auto',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
      
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing YOLOv11n on {self.device}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO class names
        self.class_names = self.model.names
        
    def detect(self, 
               frame: np.ndarray, 
               target_classes: Optional[List[int]] = None) -> List[Detection]:
       
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=target_classes,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes
                
                for i in range(len(boxes)):
                    # Extract detection info
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Create Detection object
                    detection = Detection(
                        bbox=bbox,
                        confidence=float(confidence),
                        class_id=class_id,
                        class_name=self.class_names[class_id]
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def detect_persons(self, frame: np.ndarray) -> List[Detection]:
      
        return self.detect(frame, target_classes=[0])  # Person class is 0 in COCO
    
    def draw_detections(self, 
                       frame: np.ndarray, 
                       detections: List[Detection],
                       draw_labels: bool = True) -> np.ndarray:
    
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for detections
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            if draw_labels:
                # Prepare label
                label = f"{det.class_name}: {det.confidence:.2f}"
                if det.track_id is not None:
                    label = f"ID:{det.track_id} {label}"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label,
                          (x1, y1 - 2),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, 
                     video_path: str,
                     output_path: Optional[str] = None,
                     target_classes: Optional[List[int]] = None,
                     display: bool = True,
                     frame_callback=None):
   
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                detections = self.detect(frame, target_classes)
                
                # Apply callback if provided
                if frame_callback:
                    processed_frame = frame_callback(frame, detections)
                else:
                    processed_frame = self.draw_detections(frame, detections)
                
                # Save frame
                if out:
                    out.write(processed_frame)
                
                # Display
                if display:
                    cv2.imshow('Object Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
        print(f"Processed {frame_count} frames")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector(
        model_path='yolo11n.pt',
        conf_threshold=0.5
    )
    
    # Test on image
    image = cv2.imread('test_image.jpg')
    if image is not None:
        detections = detector.detect_persons(image)
        result = detector.draw_detections(image, detections)
        cv2.imshow('Detections', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Test on video
    # detector.process_video('test_video.mp4', target_classes=[0])  # Detect persons only