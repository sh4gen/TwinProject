import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from object_detection import ObjectDetector, Detection
from reid import PersonReID

class ReIDPipeline:
    """Complete Person Re-Identification Pipeline"""
    
    def __init__(self,
                 yolo_model_path: str = 'yolo11n.pt',
                 reid_model_path: Optional[str] = None,
                 device: str = 'auto',
                 detection_conf: float = 0.5,
                 reid_threshold: float = 0.7):
       
        print("Initializing Person Re-Identification Pipeline...")
        
        # Initialize object detector
        self.detector = ObjectDetector(
            model_path=yolo_model_path,
            device=device,
            conf_threshold=detection_conf
        )
        
        # Initialize ReID module
        self.reid = PersonReID(
            model_path=reid_model_path,
            device=device,
            similarity_threshold=reid_threshold
        )
        
        # Pipeline statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'processing_times': []
        }
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        
        start_time = time.time()
        
        # Step 1: Detect persons
        detections = self.detector.detect_persons(frame)
        
        # Step 2: Re-identify persons
        detections = self.reid.process_detections(frame, detections)
        
        # Step 3: Draw results
        annotated_frame = self.reid.draw_tracks(frame, detections, draw_trails=True)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        self.stats['frames_processed'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['processing_times'].append(processing_time)
        
        # Add frame info
        frame_stats = {
            'detections': len(detections),
            'active_persons': self.reid.get_statistics()['active_persons'],
            'processing_time': processing_time,
            'fps': 1.0 / processing_time if processing_time > 0 else 0
        }
        
        # Draw statistics on frame
        self._draw_stats(annotated_frame, frame_stats)
        
        return annotated_frame, frame_stats
    
    def _draw_stats(self, frame: np.ndarray, stats: Dict[str, Any]):
        """Draw statistics on frame"""
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw stats text
        y_offset = 30
        texts = [
            f"FPS: {stats['fps']:.1f}",
            f"Detections: {stats['detections']}",
            f"Active Persons: {stats['active_persons']}",
            f"Frame: {self.stats['frames_processed']}"
        ]
        
        for text in texts:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 20
    
    def process_video(self,
                     input_path: str,
                     output_path: Optional[str] = None,
                     display: bool = True,
                     save_stats: bool = True):
        
        # Open video
        if isinstance(input_path, int) or input_path.isdigit():
            cap = cv2.VideoCapture(int(input_path))
            print(f"Opening camera {input_path}")
        else:
            cap = cv2.VideoCapture(input_path)
            print(f"Processing video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        # Processing loop
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, frame_stats = self.process_frame(frame)
                
                # Save frame
                if out:
                    out.write(processed_frame)
                
                # Display
                if display:
                    cv2.imshow('Person Re-Identification Pipeline', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing interrupted by user")
                        break
                
                # Progress update
                frame_count += 1
                if frame_count % 30 == 0 and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print("\n" + "="*50)
            print("Processing Complete!")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total unique persons detected: {self.reid.next_person_id}")
            print("="*50)
            
            # Save statistics
            if save_stats:
                self._save_statistics(output_path)
    
    def _save_statistics(self, output_path: Optional[str]):
        """Save pipeline statistics to JSON"""
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'frames_processed': self.stats['frames_processed'],
            'total_detections': self.stats['total_detections'],
            'unique_persons': self.reid.next_person_id,
            'avg_processing_time': np.mean(self.stats['processing_times']),
            'avg_fps': 1.0 / np.mean(self.stats['processing_times'])
        }
        
        # Determine stats file path
        if output_path:
            stats_path = Path(output_path).with_suffix('.json')
        else:
            stats_path = Path('pipeline_stats.json')
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Statistics saved to: {stats_path}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ReIDPipeline(
        yolo_model_path='yolo11n.pt',
        reid_model_path='/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.1.4/train/model_epoch_039_step_21863.pth', 
        detection_conf=0.5,
        reid_threshold=0.7
    )
    
    # Process video file
    pipeline.process_video(
        input_path='/home/ika/yzlm/TwinProject/ReID_Inference/testing_videos/20.avi',
        output_path='/home/ika/yzlm/TwinProject/ReID_Inference/result_videos/20.avi',
        display=True,
        save_stats=True
    )
    
    # Or process webcam
    # pipeline.process_video(
    #     input_path=0,  # Webcam
    #     output_path='webcam_reid.mp4',
    #     display=True
    # )