import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import glob
from tqdm import tqdm
import re
import os

class BatchReIDEvaluator:
    def __init__(self, onnx_path, batch_size=32):
        self.session = ort.InferenceSession(onnx_path, 
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.226, 0.226, 0.226])
        ])
        
    def extract_features_batch(self, image_dir):
        """Extract features in batches to avoid OOM"""
        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        image_paths.extend(sorted(glob.glob(os.path.join(image_dir, '*.png'))))
        
        all_features = []
        all_paths = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"Extracting from {image_dir}"):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_imgs = []
            
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                batch_imgs.append(img)
            
            if batch_imgs:
                batch_tensor = torch.stack(batch_imgs).numpy()
                input_name = self.session.get_inputs()[0].name
                features = self.session.run(None, {input_name: batch_tensor})[0]
                all_features.append(features)
                all_paths.extend(batch_paths)
        
        return np.concatenate(all_features, axis=0), all_paths
    
    def compute_distance_matrix_batch(self, query_feats, gallery_feats, batch_size=512):
        """Compute distance matrix in batches"""
        num_query = query_feats.shape[0]
        num_gallery = gallery_feats.shape[0]
        distmat = np.zeros((num_query, num_gallery), dtype=np.float32)
        
        for i in tqdm(range(0, num_query, batch_size), desc="Computing distances"):
            q_batch = query_feats[i:i+batch_size]
            dist = np.power(q_batch, 2).sum(axis=1, keepdims=True) + \
                   np.power(gallery_feats, 2).sum(axis=1) - \
                   2 * np.dot(q_batch, gallery_feats.T)
            distmat[i:i+batch_size] = np.sqrt(np.maximum(dist, 0))
        
        return distmat
    
    def evaluate(self, query_dir, gallery_dir):
        # Extract features
        print("Extracting query features...")
        query_feats, query_paths = self.extract_features_batch(query_dir)
        
        print("Extracting gallery features...")
        gallery_feats, gallery_paths = self.extract_features_batch(gallery_dir)
        
        # Get IDs
        query_ids, query_cams = self.get_ids_and_cams(query_paths)
        gallery_ids, gallery_cams = self.get_ids_and_cams(gallery_paths)
        
        # Compute distance matrix in batches
        distmat = self.compute_distance_matrix_batch(query_feats, gallery_feats)
        
        # Calculate metrics
        results = self.calculate_metrics(distmat, query_ids, gallery_ids, 
                                        query_cams, gallery_cams)
        
        return results
    
    def get_ids_and_cams(self, img_paths):
        ids, cams = [], []
        pattern = re.compile(r'([-\d]+)_c(\d)')
        for path in img_paths:
            fname = os.path.basename(path)
            m = pattern.search(fname)
            if m:
                ids.append(int(m.group(1)))
                cams.append(int(m.group(2)))
            else:
                ids.append(-1)
                cams.append(-1)
        return np.array(ids), np.array(cams)
    
    def calculate_metrics(self, distmat, query_ids, gallery_ids, query_cams, gallery_cams):
        num_query = distmat.shape[0]
        CMC = np.zeros(len(gallery_ids))
        AP = 0.0
        valid_queries = 0
        
        for i in range(num_query):
            if query_ids[i] == -1: continue
            valid_queries += 1
            
            order = np.argsort(distmat[i])
            remove = (gallery_ids[order] == query_ids[i]) & (gallery_cams[order] == query_cams[i])
            keep = np.invert(remove)
            match = (gallery_ids[order] == query_ids[i])[keep]
            
            if not np.any(match): continue
            
            first_index = np.where(match)[0][0]
            CMC[first_index:] += 1
            
            num_rel = match.sum()
            tmp_cmc = match.cumsum()
            precision = tmp_cmc / (np.arange(len(match)) + 1)
            AP += (precision * match).sum() / num_rel
        
        mAP = AP / valid_queries
        CMC = CMC / valid_queries
        
        return {
            'mAP': mAP,
            'Rank-1': CMC[0],
            'Rank-5': CMC[4] if len(CMC) > 4 else 1.0,
            'Rank-10': CMC[9] if len(CMC) > 9 else 1.0
        }

# Kullanım
evaluator = BatchReIDEvaluator('model.onnx', batch_size=32)
results = evaluator.evaluate(
    '/home/ika/yzlm/TwinProject/CCVID/CCVID_market1501_format/query',
    '/home/ika/yzlm/TwinProject/CCVID/CCVID_market1501_format/bounding_box_test'
)

print("\n==== Evaluation Results ====")
for k, v in results.items():
    print(f"{k}: {v:.4f}")