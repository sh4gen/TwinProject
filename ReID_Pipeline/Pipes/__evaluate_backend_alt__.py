import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from torchvision import transforms
import torch

# =========================
# DATASET
# =========================
class ReIDImageDataset:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.img_files.sort()
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.444, 0.438, 0.457], [0.288, 0.280, 0.275])
        ])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        pid = int(self.img_files[idx].split('_')[0])
        return img, pid, self.img_files[idx]

# =========================
# FEATURE SAVE/LOAD
# =========================
def save_features(features, pids, fnames, save_path):
    np.savez_compressed(save_path, features=features, pids=pids, fnames=fnames)
    print(f"Features saved to: {save_path}")

def load_features(load_path):
    data = np.load(load_path)
    features = data['features']
    pids = data['pids']
    fnames = data['fnames']
    print(f"Features loaded from: {load_path}")
    return features, pids, fnames

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features_onnx(onnx_path, dataset, batch_size=32, save_path=None):
    if save_path and os.path.exists(save_path):
        print(f"Loading existing features from: {save_path}")
        return load_features(save_path)
    
    session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )   
    input_name = session.get_inputs()[0].name
    features, pids, fnames = [], [], []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting features"):
        batch_imgs = []
        batch_pids = []
        batch_fnames = []
        for j in range(i, min(i+batch_size, len(dataset))):
            img, pid, fname = dataset[j]
            batch_imgs.append(img.numpy())
            batch_pids.append(pid)
            batch_fnames.append(fname)
        batch_imgs = np.stack(batch_imgs)
        feats = session.run(None, {input_name: batch_imgs.astype(np.float32)})[0]
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        features.append(feats)
        pids.extend(batch_pids)
        fnames.extend(batch_fnames)
    
    features = np.vstack(features)
    pids = np.array(pids)
    if save_path:
        save_features(features, pids, fnames, save_path)
    return features, pids, fnames

# =========================
# RE-RANKING (opsiyonel)
# =========================
def re_ranking(query_features, gallery_features, k1=20, k2=6, lambda_value=0.3, batch_size=256):
    # GPU'da çalışır, memory friendly
    import faiss
    qf = query_features.astype(np.float32)
    gf = gallery_features.astype(np.float32)
    all_features = np.vstack([qf, gf])
    all_num = all_features.shape[0]
    q_num = qf.shape[0]
    g_num = gf.shape[0]
    # FAISS ile hızlı mesafe matrisi
    index = faiss.IndexFlatL2(all_features.shape[1])
    index.add(all_features)
    _, initial_rank = index.search(all_features, k1 + 1)
    V = np.zeros((all_num, all_num), dtype=np.float32)
    for i in tqdm(range(all_num), desc="Re-ranking"):
        k_reciprocal_index = initial_rank[i, 1:k1+1]
        V[i, k_reciprocal_index] = 1
    V = V[:q_num, q_num:]
    jaccard_dist = 1 - V
    original_dist = np.linalg.norm(qf[:, None] - gf[None, :], axis=2)
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
    return final_dist

# =========================
# EVALUATION
# =========================
def evaluate_gpu_memory_efficient(query_features, gallery_features, query_pids, gallery_pids,
                                 batch_size=512, topk=(1, 5, 10), distmat=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation device: {device}")

    qf = torch.from_numpy(query_features).to(device)
    gf = torch.from_numpy(gallery_features).to(device)
    q_pids = torch.from_numpy(query_pids).to(device)
    g_pids = torch.from_numpy(gallery_pids).to(device)

    num_query = qf.shape[0]
    max_rank = max(topk)
    total_cmc = torch.zeros(max_rank, device=device)
    total_ap = 0.0
    valid_queries = 0

    for i in tqdm(range(0, num_query, batch_size), desc="Evaluating Batches"):
        batch_qf = qf[i : i + batch_size]
        batch_q_pids = q_pids[i : i + batch_size]
        if distmat is not None:
            batch_distmat = torch.from_numpy(distmat[i : i + batch_size]).to(device)
        else:
            batch_distmat = torch.cdist(batch_qf, gf, p=2)
        indices = torch.argsort(batch_distmat, dim=1)
        matches = (g_pids[indices] == batch_q_pids.view(-1, 1)).int()
        num_rel = matches.sum(dim=1)
        has_match = num_rel > 0
        if not has_match.any():
            continue
        valid_queries += has_match.sum().item()
        positions = torch.arange(1, matches.shape[1] + 1, device=device).expand_as(matches)
        precision = matches.cumsum(dim=1) / positions
        ap = (precision * matches).sum(dim=1) / num_rel.clamp(min=1)
        total_ap += ap[has_match].sum().item()
        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        total_cmc += cmc[has_match, :max_rank].sum(dim=0)
    mAP = total_ap / valid_queries if valid_queries > 0 else 0.0
    cmc_scores = (total_cmc / valid_queries).cpu().numpy()
    final_scores = [cmc_scores[k-1] for k in topk]
    return mAP, final_scores

# =========================
# PIPELINE
# =========================
def reid_pipeline(
    onnx_path, query_dir, gallery_dir,
    query_features_path=None, gallery_features_path=None,
    feature_batch_size=256, eval_batch_size=1024,
    topk=(1, 5, 10), use_rerank=False
):
    print("\nLoading datasets...")
    query_dataset = ReIDImageDataset(query_dir)
    gallery_dataset = ReIDImageDataset(gallery_dir)
    print(f"Query: {len(query_dataset)} images")
    print(f"Gallery: {len(gallery_dataset)} images")

    print("\nExtracting/Loading query features...")
    query_features, query_pids, query_fnames = extract_features_onnx(
        onnx_path, query_dataset, 
        batch_size=feature_batch_size,
        save_path=query_features_path
    )
    print("\nExtracting/Loading gallery features...")
    gallery_features, gallery_pids, gallery_fnames = extract_features_onnx(
        onnx_path, gallery_dataset, 
        batch_size=feature_batch_size,
        save_path=gallery_features_path
    )
    print(f"\nFeature shapes:")
    print(f"Query: {query_features.shape}")
    print(f"Gallery: {gallery_features.shape}")

    if use_rerank:
        print("\nApplying re-ranking...")
        distmat = re_ranking(query_features, gallery_features)
        mAP, cmc_scores = evaluate_gpu_memory_efficient(
            query_features, gallery_features, query_pids, gallery_pids,
            batch_size=eval_batch_size, topk=topk, distmat=distmat
        )
    else:
        mAP, cmc_scores = evaluate_gpu_memory_efficient(
            query_features, gallery_features, query_pids, gallery_pids,
            batch_size=eval_batch_size, topk=topk
        )
    print("\nResults:")
    print(f"mAP: {mAP:.4f}")
    for i, k in enumerate(topk):
        print(f"Rank-{k}: {cmc_scores[i]:.4f}")
    return mAP, cmc_scores

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    onnx_path = "/home/ika/yzlm/TwinProject/CCVID/results/exported/model49.onnx"
    query_dir = "/home/ika/yzlm/TwinProject/CCVID/data/query"
    gallery_dir = "/home/ika/yzlm/TwinProject/CCVID/data/bounding_box_test"
    query_features_path = "/home/ika/yzlm/TwinProject/CCVID/features/query_features.npz"
    gallery_features_path = "/home/ika/yzlm/TwinProject/CCVID/features/gallery_features.npz"
    os.makedirs(os.path.dirname(query_features_path), exist_ok=True)
    feature_batch_size = 256
    eval_batch_size = 1024
    topk = (1, 5, 10)
    use_rerank = False  # True yaparsan re-ranking aktif olur

    reid_pipeline(
        onnx_path, query_dir, gallery_dir,
        query_features_path, gallery_features_path,
        feature_batch_size, eval_batch_size, topk, use_rerank
    )