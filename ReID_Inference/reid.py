import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
from dataclasses import dataclass
from collections import defaultdict
import pickle
import os
from pathlib import Path
import json
from datetime import datetime

@dataclass
class Person:
    """Data class for storing person information"""
    person_id: int
    features: np.ndarray
    last_seen_frame: int
    bboxes: List[np.ndarray]
    confidence_scores: List[float]

@dataclass
class GalleryEntry:
    """Data class for gallery entries"""
    person_id: int
    embedding: np.ndarray
    image_crop: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    confidence: float = 0.0
    timestamp: Optional[str] = None

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

class GalleryQueryReID:
    """Enhanced Person Re-Identification with Gallery/Query System"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 similarity_threshold: float = 0.7,
                 max_absent_frames: int = 30,
                 gallery_path: str = './reid_gallery',
                 save_crops: bool = True):
        """
        Initialize ReID module with Gallery/Query system
        
        Args:
            model_path: Path to trained ReID model
            device: Device to run inference on
            similarity_threshold: Threshold for matching persons
            max_absent_frames: Maximum frames a person can be absent before removal
            gallery_path: Path to save gallery data
            save_crops: Whether to save image crops for visualization
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Initializing Gallery/Query ReID system on {self.device}...")
        
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
        
        # Parameters
        self.similarity_threshold = similarity_threshold
        self.max_absent_frames = max_absent_frames
        self.save_crops = save_crops
        
        # Gallery system
        self.gallery_path = Path(gallery_path)
        self.gallery_path.mkdir(exist_ok=True)
        self.gallery: Dict[int, GalleryEntry] = {}
        self.gallery_embeddings: Dict[int, np.ndarray] = {}
        
        # Load existing gallery
        self.load_gallery()
        
        # Person tracking data (current session)
        self.persons: Dict[int, Person] = {}
        self.next_person_id = max(self.gallery.keys()) + 1 if self.gallery else 0
        self.current_frame = 0
        
        # Query cache for current detections
        self.query_embeddings: List[np.ndarray] = []
        self.query_bboxes: List[np.ndarray] = []
        self.query_confidences: List[float] = []
        
    def load_model(self, model_path: str):
        """Load model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded ReID model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized model")
    
    def save_gallery(self):
        """Save gallery to disk"""
        # Save gallery embeddings and metadata
        gallery_data = {
            'embeddings': self.gallery_embeddings,
            'metadata': {
                pid: {
                    'person_id': entry.person_id,
                    'confidence': entry.confidence,
                    'timestamp': entry.timestamp,
                    'bbox': entry.bbox.tolist() if entry.bbox is not None else None
                }
                for pid, entry in self.gallery.items()
            },
            'next_person_id': self.next_person_id
        }
        
        gallery_file = self.gallery_path / 'gallery.pkl'
        with open(gallery_file, 'wb') as f:
            pickle.dump(gallery_data, f)
        
        print(f"Gallery saved to {gallery_file}")
    
    def load_gallery(self):
        """Load gallery from disk"""
        gallery_file = self.gallery_path / 'gallery.pkl'
        
        if gallery_file.exists():
            try:
                with open(gallery_file, 'rb') as f:
                    gallery_data = pickle.load(f)
                
                self.gallery_embeddings = gallery_data['embeddings']
                metadata = gallery_data['metadata']
                
                # Reconstruct gallery entries
                for pid, meta in metadata.items():
                    self.gallery[pid] = GalleryEntry(
                        person_id=meta['person_id'],
                        embedding=self.gallery_embeddings[pid],
                        confidence=meta['confidence'],
                        timestamp=meta['timestamp'],
                        bbox=np.array(meta['bbox']) if meta['bbox'] else None
                    )
                
                self.next_person_id = gallery_data.get('next_person_id', max(self.gallery.keys()) + 1 if self.gallery else 0)
                print(f"Loaded gallery with {len(self.gallery)} persons")
                
            except Exception as e:
                print(f"Error loading gallery: {e}")
                print("Starting with empty gallery")
    
    def extract_features(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract ReID features from person crop"""
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
    
    def compute_similarity_matrix(self, query_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity matrix between query embeddings and gallery
        
        Returns:
            similarity_matrix: Shape (n_queries, n_gallery_persons)
        """
        if not self.gallery_embeddings or not query_embeddings:
            return np.array([])
        
        # Convert to matrices
        query_matrix = np.vstack(query_embeddings)  # (n_queries, embedding_dim)
        gallery_ids = list(self.gallery_embeddings.keys())
        gallery_matrix = np.vstack([self.gallery_embeddings[pid] for pid in gallery_ids])  # (n_gallery, embedding_dim)
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(query_matrix, gallery_matrix.T)  # (n_queries, n_gallery)
        
        return similarity_matrix, gallery_ids
    
    def query_gallery(self, query_embeddings: List[np.ndarray]) -> List[Optional[int]]:
        """
        Query gallery with current detections
        
        Returns:
            List of person IDs or None for each query
        """
        if not self.gallery_embeddings:
            return [None] * len(query_embeddings)
        
        similarity_matrix, gallery_ids = self.compute_similarity_matrix(query_embeddings)
        
        if similarity_matrix.size == 0:
            return [None] * len(query_embeddings)
        
        matches = []
        
        for i in range(len(query_embeddings)):
            # Find best match for this query
            best_gallery_idx = np.argmax(similarity_matrix[i])
            best_similarity = similarity_matrix[i, best_gallery_idx]
            
            if best_similarity > self.similarity_threshold:
                matched_person_id = gallery_ids[best_gallery_idx]
                matches.append(matched_person_id)
            else:
                matches.append(None)
        
        return matches
    
    def add_to_gallery(self, person_id: int, embedding: np.ndarray, 
                      image_crop: Optional[np.ndarray] = None, 
                      bbox: Optional[np.ndarray] = None,
                      confidence: float = 0.0):
        """Add person to gallery"""
        
        # Create gallery entry
        gallery_entry = GalleryEntry(
            person_id=person_id,
            embedding=embedding,
            image_crop=image_crop,
            bbox=bbox,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        # Update gallery
        self.gallery[person_id] = gallery_entry
        self.gallery_embeddings[person_id] = embedding
        
        # Save image crop if enabled
        if self.save_crops and image_crop is not None:
            crop_path = self.gallery_path / f'person_{person_id}.jpg'
            cv2.imwrite(str(crop_path), image_crop)
        
        print(f"Added person {person_id} to gallery")
    
    def update_gallery_embedding(self, person_id: int, new_embedding: np.ndarray, alpha: float = 0.1):
        """Update existing gallery embedding with exponential moving average"""
        if person_id in self.gallery_embeddings:
            # Exponential moving average
            old_embedding = self.gallery_embeddings[person_id]
            updated_embedding = alpha * new_embedding + (1 - alpha) * old_embedding
            
            # Re-normalize
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            
            self.gallery_embeddings[person_id] = updated_embedding
            self.gallery[person_id].embedding = updated_embedding
    
    def process_detections(self, 
                          frame: np.ndarray, 
                          detections: List['Detection']) -> List['Detection']:
        """Process detections using Gallery/Query system"""
        self.update_frame()
        
        if not detections:
            return detections
        
        # Step 1: Extract query embeddings for all detections
        self.query_embeddings = []
        self.query_bboxes = []
        self.query_confidences = []
        query_crops = []
        
        for detection in detections:
            embedding = self.extract_features(frame, detection.bbox)
            if embedding is not None:
                self.query_embeddings.append(embedding)
                self.query_bboxes.append(detection.bbox)
                self.query_confidences.append(detection.confidence)
                
                # Extract crop for potential gallery storage
                x1, y1, x2, y2 = detection.bbox.astype(int)
                crop = frame[y1:y2, x1:x2]
                query_crops.append(crop)
            else:
                self.query_embeddings.append(None)
                query_crops.append(None)
        
        # Step 2: Query gallery for matches
        valid_queries = [emb for emb in self.query_embeddings if emb is not None]
        matches = self.query_gallery(valid_queries)
        
        # Step 3: Process results
        valid_idx = 0
        for i, detection in enumerate(detections):
            if self.query_embeddings[i] is not None:
                matched_person_id = matches[valid_idx]
                
                if matched_person_id is not None:
                    # Existing person found in gallery
                    detection.track_id = matched_person_id
                    
                    # Update gallery embedding
                    self.update_gallery_embedding(matched_person_id, self.query_embeddings[i])
                    
                    # Update current session tracking
                    if matched_person_id not in self.persons:
                        self.persons[matched_person_id] = Person(
                            person_id=matched_person_id,
                            features=self.query_embeddings[i],
                            last_seen_frame=self.current_frame,
                            bboxes=[detection.bbox],
                            confidence_scores=[detection.confidence]
                        )
                    else:
                        person = self.persons[matched_person_id]
                        person.last_seen_frame = self.current_frame
                        person.bboxes.append(detection.bbox)
                        person.confidence_scores.append(detection.confidence)
                        
                        if len(person.bboxes) > 100:
                            person.bboxes.pop(0)
                            person.confidence_scores.pop(0)
                
                else:
                    # New person - add to gallery
                    new_person_id = self.next_person_id
                    detection.track_id = new_person_id
                    
                    # Add to gallery
                    self.add_to_gallery(
                        person_id=new_person_id,
                        embedding=self.query_embeddings[i],
                        image_crop=query_crops[i],
                        bbox=detection.bbox,
                        confidence=detection.confidence
                    )
                    
                    # Add to current session tracking
                    self.persons[new_person_id] = Person(
                        person_id=new_person_id,
                        features=self.query_embeddings[i],
                        last_seen_frame=self.current_frame,
                        bboxes=[detection.bbox],
                        confidence_scores=[detection.confidence]
                    )
                    
                    self.next_person_id += 1
                
                valid_idx += 1
        
        # Save gallery periodically (every 100 frames)
        if self.current_frame % 100 == 0:
            self.save_gallery()
        
        return detections
    
    def update_frame(self):
        """Update frame counter and clean up old persons"""
        self.current_frame += 1
        
        # Remove persons not seen for too long (only from current session)
        persons_to_remove = []
        for person_id, person in self.persons.items():
            if self.current_frame - person.last_seen_frame > self.max_absent_frames:
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del self.persons[person_id]
    
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
        """Draw tracking results"""
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
                
                # Check if person is in gallery
                in_gallery = det.track_id in self.gallery
                
                # Draw bounding box
                thickness = 3 if in_gallery else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label = f"Person {det.track_id}"
                if in_gallery:
                    label += " (G)"  # Indicate gallery presence
                    
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
            'gallery_size': len(self.gallery),
            'active_persons': active_persons,
            'current_frame': self.current_frame
        }
    
    def export_gallery_info(self, output_path: str = './gallery_info.json'):
        """Export gallery information for analysis"""
        gallery_info = {
            'gallery_size': len(self.gallery),
            'persons': [
                {
                    'person_id': entry.person_id,
                    'confidence': entry.confidence,
                    'timestamp': entry.timestamp,
                    'has_crop': f'person_{entry.person_id}.jpg' if self.save_crops else None
                }
                for entry in self.gallery.values()
            ],
            'statistics': self.get_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(gallery_info, f, indent=2)
        
        print(f"Gallery info exported to {output_path}")

# Example usage
if __name__ == "__main__":
    # Test Enhanced ReID module
    reid = GalleryQueryReID(
        model_path='reid_model.pth',
        similarity_threshold=0.7,
        gallery_path='./reid_gallery'
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
        cv2.imshow('Enhanced ReID Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Export gallery info
        reid.export_gallery_info()