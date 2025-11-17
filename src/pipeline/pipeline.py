import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from src.pipeline.object_detection import ObjectDetector, Detection
from src.pipeline.reid import GalleryQueryReID

class ReIDPipeline:
    """Complete Person Re-Identification Pipeline with Gallery/Query System"""
    
    def __init__(self,
                 yolo_model_path: str = 'yolo11n.pt',
                 reid_model_path: Optional[str] = None,
                 device: str = 'auto',
                 detection_conf: float = 0.5,
                 reid_threshold: float = 0.7,
                 gallery_path: str = './reid_gallery'):
       
        print("Initializing Person Re-Identification Pipeline with Gallery/Query System...")
        
        # Initialize object detector
        self.detector = ObjectDetector(
            model_path=yolo_model_path,
            device=device,
            conf_threshold=detection_conf
        )
        
        # Initialize Enhanced ReID module with Gallery/Query system
        self.reid = GalleryQueryReID(
            model_path=reid_model_path,
            device=device,
            similarity_threshold=reid_threshold,
            gallery_path=gallery_path,
            save_crops=True  # Save person crops for visualization
        )
        
        # Pipeline statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'processing_times': [],
            'gallery_size_history': []
        }
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        
        start_time = time.time()
        
        # Step 1: Detect persons
        detections = self.detector.detect_persons(frame)
        
        # Step 2: Process with Gallery/Query system
        detections = self.reid.process_detections(frame, detections)
        
        # Step 3: Draw results
        annotated_frame = self.reid.draw_tracks(frame, detections, draw_trails=True)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        self.stats['frames_processed'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['processing_times'].append(processing_time)
        self.stats['gallery_size_history'].append(len(self.reid.gallery))
        
        # Get ReID statistics
        reid_stats = self.reid.get_statistics()
        
        # Add frame info
        frame_stats = {
            'detections': len(detections),
            'active_persons': reid_stats['active_persons'],
            'gallery_size': reid_stats['gallery_size'],
            'total_persons_seen': reid_stats['total_persons_seen'],
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
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw stats text
        y_offset = 30
        texts = [
            f"FPS: {stats['fps']:.1f}",
            f"Detections: {stats['detections']}",
            f"Active Persons: {stats['active_persons']}",
            f"Gallery Size: {stats['gallery_size']}",
            f"Total Seen: {stats['total_persons_seen']}",
            f"Frame: {self.stats['frames_processed']}"
        ]
        
        for text in texts:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 15
    
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
                    cv2.imshow('Person Re-Identification Pipeline (Gallery/Query)', processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing interrupted by user")
                        break
                    elif key == ord('s'):
                        # Save current gallery state
                        self.reid.save_gallery()
                        print("Gallery saved!")
                    elif key == ord('e'):
                        # Export gallery info
                        self.reid.export_gallery_info(f'gallery_info_frame_{frame_count}.json')
                
                # Progress update
                frame_count += 1
                if frame_count % 30 == 0 and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Gallery: {len(self.reid.gallery)} persons")
        
        finally:
            # Final save of gallery
            self.reid.save_gallery()
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print("\n" + "="*60)
            print("Processing Complete!")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Gallery size: {len(self.reid.gallery)} persons")
            print(f"Total unique persons detected: {self.reid.next_person_id}")
            print("="*60)
            
            # Save statistics
            if save_stats:
                self._save_statistics(output_path)
                
            # Export final gallery info
            self.reid.export_gallery_info('final_gallery_info.json')
    
    def _save_statistics(self, output_path: Optional[str]):
        """Save pipeline statistics to JSON"""
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'frames_processed': self.stats['frames_processed'],
            'total_detections': self.stats['total_detections'],
            'unique_persons': self.reid.next_person_id,
            'final_gallery_size': len(self.reid.gallery),
            'avg_processing_time': np.mean(self.stats['processing_times']),
            'avg_fps': 1.0 / np.mean(self.stats['processing_times']),
            'gallery_growth': {
                'initial_size': self.stats['gallery_size_history'][0] if self.stats['gallery_size_history'] else 0,
                'final_size': self.stats['gallery_size_history'][-1] if self.stats['gallery_size_history'] else 0,
                'max_size': max(self.stats['gallery_size_history']) if self.stats['gallery_size_history'] else 0
            }
        }
        
        # Determine stats file path
        if output_path:
            stats_path = Path(output_path).with_suffix('.json')
        else:
            stats_path = Path('pipeline_stats_gallery.json')
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Pipeline statistics saved to: {stats_path}")

    def load_existing_gallery(self, gallery_path: str):
        """Load an existing gallery for continued processing"""
        self.reid.gallery_path = Path(gallery_path)
        self.reid.load_gallery()
        print(f"Loaded existing gallery from {gallery_path} with {len(self.reid.gallery)} persons")

    def reset_gallery(self):
        """Reset the gallery (useful for testing)"""
        self.reid.gallery.clear()
        self.reid.gallery_embeddings.clear()
        self.reid.next_person_id = 0
        print("Gallery reset!")

    def get_gallery_summary(self):
        """Print summary of current gallery"""
        print("\n" + "="*50)
        print("Gallery Summary:")
        print(f"Total persons in gallery: {len(self.reid.gallery)}")
        print(f"Next person ID: {self.reid.next_person_id}")
        print(f"Gallery path: {self.reid.gallery_path}")
        
        if self.reid.gallery:
            print("\nPersons in gallery:")
            for pid, entry in self.reid.gallery.items():
                print(f"  Person {pid}: conf={entry.confidence:.3f}, added={entry.timestamp}")
        print("="*50)

# Example usage
if __name__ == "__main__":
    # Initialize pipeline with gallery system
    pipeline = ReIDPipeline(
        yolo_model_path='yolo11n.pt',
        reid_model_path='/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.1.4/train/model_epoch_039_step_21863.pth', 
        detection_conf=0.5,
        reid_threshold=0.7,
        gallery_path='./reid_gallery'  # Directory to store gallery data
    )
    
    # Show initial gallery state
    pipeline.get_gallery_summary()
    
    # Process video file
    pipeline.process_video(
        input_path='/home/ika/yzlm/TwinProject/ReID_Inference/mot16-05/MOT16-05.mp4',
        output_path='/home/ika/yzlm/TwinProject/ReID_Inference/mot16-05/MOT16-05_result.mp4',
        display=True,
        save_stats=True
    )
    
    # Show final gallery state
    pipeline.get_gallery_summary()
    
    # Or process webcam (uncomment to use)
    # pipeline.process_video(
    #     input_path=0,  # Webcam
    #     output_path='webcam_reid_gallery.mp4',
    #     display=True
    # )
    
    # Example of loading existing gallery for continued processing
    # pipeline2 = ReIDPipeline(gallery_path='./reid_gallery')
    # pipeline2.load_existing_gallery('./reid_gallery')
    # pipeline2.process_video('another_video.mp4')