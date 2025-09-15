import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Person:
    """Data class for storing person information"""
    person_id: int
    features: np.ndarray
    last_seen_frame: int
    bboxes: List[np.ndarray]
    confidence_scores: List[float]

class ReIDModel(nn.Module):
    """Person Re-Identification Model"""
    
    def __init__(self, embedding_dim: int = 2048, pretrained: bool = True):
        super(ReIDModel, self).__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the last FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add embedding layer
        self.embedding = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        
        # Generate embeddings
        embeddings = self.embedding(x)
        embeddings = self.bn(embeddings)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class PersonReID:
    """Person Re-Identification module"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 similarity_threshold: float = 0.7,
                 max_absent_frames: int = 30):
        """
        Initialize ReID module
        
        Args:
            model_path: Path to trained ReID model
            device: Device to run inference on
            similarity_threshold: Threshold for matching persons
            max_absent_frames: Maximum frames a person can be absent before removal
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Initializing ReID model on {self.device}...")
        
        # Initialize model
        self.model = ReIDModel().to(self.device)
        self.model.eval()
        
        # Load weights if provided
        if model_path:
            self.load_model(model_path)
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Tracking parameters
        self.similarity_threshold = similarity_threshold
        self.max_absent_frames = max_absent_frames
        
        # Person tracking data
        self.persons: Dict[int, Person] = {}
        self.next_person_id = 0
        self.current_frame = 0
        
    def load_model(self, model_path: str):
        """Load model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded ReID model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized model")
    
    def extract_features(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ReID features from person crop
        
        Args:
            image: Full frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector or None if extraction fails
        """
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Crop person
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0 or person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
            return None
        
        try:
            # Preprocess
            person_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(person_tensor)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors"""
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
    def match_person(self, features: np.ndarray, bbox: np.ndarray, confidence: float) -> int:
        """
        Match features to existing persons or create new person
        
        Args:
            features: Feature vector
            bbox: Bounding box
            confidence: Detection confidence
            
        Returns:
            Person ID
        """
        best_match_id = None
        best_similarity = 0
        
        # Compare with existing persons
        for person_id, person in self.persons.items():
            # Skip if person hasn't been seen for too long
            if self.current_frame - person.last_seen_frame > self.max_absent_frames:
                continue
                
            similarity = self.compute_similarity(features, person.features)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = person_id
        
        if best_match_id is not None:
            # Update existing person
            person = self.persons[best_match_id]
            
            # Update features with exponential moving average
            alpha = 0.1
            person.features = alpha * features + (1 - alpha) * person.features
            
            # Update tracking info
            person.last_seen_frame = self.current_frame
            person.bboxes.append(bbox)
            person.confidence_scores.append(confidence)
            
            # Keep only recent history
            if len(person.bboxes) > 100:
                person.bboxes.pop(0)
                person.confidence_scores.pop(0)
                
            return best_match_id
        else:
            # Create new person
            new_person = Person(
                person_id=self.next_person_id,
                features=features,
                last_seen_frame=self.current_frame,
                bboxes=[bbox],
                confidence_scores=[confidence]
            )
            
            self.persons[self.next_person_id] = new_person
            self.next_person_id += 1
            
            return new_person.person_id
    
    def update_frame(self):
        """Update frame counter and clean up old persons"""
        self.current_frame += 1
        
        # Remove persons not seen for too long
        persons_to_remove = []
        for person_id, person in self.persons.items():
            if self.current_frame - person.last_seen_frame > self.max_absent_frames:
                persons_to_remove.append(person_id)
        

        for person_id in persons_to_remove:
            del self.persons[person_id]
    
    def process_detections(self, 
                          frame: np.ndarray, 
                          detections: List['Detection']) -> List['Detection']:
     
        self.update_frame()
        
        for detection in detections:
            # Extract features for this detection
            features = self.extract_features(frame, detection.bbox)
            
            if features is not None:
                # Match to person
                person_id = self.match_person(
                    features, 
                    detection.bbox, 
                    detection.confidence
                )
                
                # Update detection with person ID
                detection.track_id = person_id
        
        return detections
    
    def get_person_color(self, person_id: int) -> Tuple[int, int, int]:
        """Get consistent color for person ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (255, 0, 128), (128, 255, 0),
            (0, 255, 128), (128, 0, 255), (0, 128, 255)
        ]
        return colors[person_id % len(colors)]
    
    def draw_tracks(self, 
                   frame: np.ndarray, 
                   detections: List['Detection'],
                   draw_trails: bool = True) -> np.ndarray:
    
        annotated_frame = frame.copy()
        
        # Draw trails first (so they appear behind boxes)
        if draw_trails:
            for person_id, person in self.persons.items():
                if len(person.bboxes) > 1:
                    color = self.get_person_color(person_id)
                    
                    # Draw trail
                    for i in range(1, min(len(person.bboxes), 20)):
                        if i < len(person.bboxes):
                            # Get center points
                            bbox1 = person.bboxes[-i-1]
                            bbox2 = person.bboxes[-i]
                            
                            center1 = (
                                int((bbox1[0] + bbox1[2]) / 2),
                                int((bbox1[1] + bbox1[3]) / 2)
                            )
                            center2 = (
                                int((bbox2[0] + bbox2[2]) / 2),
                                int((bbox2[1] + bbox2[3]) / 2)
                            )
                            
                            # Draw line with decreasing opacity
                            thickness = max(1, 3 - i // 5)
                            cv2.line(annotated_frame, center1, center2, color, thickness)
        
        # Draw current detections
        for det in detections:
            if det.track_id is not None:
                x1, y1, x2, y2 = det.bbox.astype(int)
                color = self.get_person_color(det.track_id)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"Person {det.track_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(annotated_frame,
                            (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                cv2.putText(annotated_frame, label,
                          (x1, y1 - 2),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_statistics(self) -> Dict:
        """Get current tracking statistics"""
        active_persons = sum(
            1 for p in self.persons.values() 
            if self.current_frame - p.last_seen_frame <= self.max_absent_frames
        )
        
        return {
            'total_persons_seen': self.next_person_id,
            'active_persons': active_persons,
            'current_frame': self.current_frame
        }

# Example usage
if __name__ == "__main__":
    # Test ReID module
    reid = PersonReID(
        model_path='reid_model.pth',
        similarity_threshold=0.7
    )
    
    # Create dummy detections
    from object_detection import Detection
    
    frame = cv2.imread('test_frame.jpg')
    if frame is not None:
        # Simulate detections
        detections = [
            Detection(
                bbox=np.array([100, 100, 200, 300]),
                confidence=0.9,
                class_id=0,
                class_name='person'
            )
        ]
        
        # Process detections
        updated_detections = reid.process_detections(frame, detections)
        
        # Draw results
        result = reid.draw_tracks(frame, updated_detections)
        cv2.imshow('ReID Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()