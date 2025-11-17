import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from torchvision import transforms
import torch
import re
from concurrent.futures import ThreadPoolExecutor
import warnings

def evaluate_reid_pipeline_optimized(
    onnx_path, 
    query_dir, 
    gallery_dir,
    batch_size=256,
    eval_batch_size=1024,
    topk=(1, 5, 10),
    use_gpu=True,
    save_features=False,
    features_dir=None,
    num_workers=4,
    use_camera_filter=False,
    use_reranking=True,
    rerank_k1=20,
    rerank_k2=6,
    rerank_lambda=0.3
):
    """
    Optimize edilmiş ReID pipeline with GPU-based re-ranking
    """
    
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.444, 0.438, 0.457], [0.288, 0.280, 0.275])
    ])
    
    def extract_pid_camid(filename):
        """PID ve Camera ID çıkar"""
        try:
            parts = filename.split('_')
            pid = int(parts[0])
            
            camid = 0
            if len(parts) > 1 and 'c' in parts[1]:
                camid = int(parts[1].split('s')[0].replace('c', ''))
            
            return pid, camid
        except:
            warnings.warn(f"Could not parse: {filename}")
            return -1, -1
    
    def load_single_image(args):
        """Tek görüntü yükle"""
        img_file, img_dir, transform = args
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        pid, camid = extract_pid_camid(img_file)
        return img_tensor.numpy(), pid, camid, img_file
    
    def parallel_load_images(img_dir, transform, num_workers):
        """Paralel görüntü yükleme"""
        img_files = sorted([f for f in os.listdir(img_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        args_list = [(f, img_dir, transform) for f in img_files]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(load_single_image, args_list),
                total=len(img_files),
                desc=f"Loading {os.path.basename(img_dir)}"
            ))
        
        images, pids, camids, fnames = zip(*results)
        return np.array(images), np.array(pids), np.array(camids), list(fnames)
    
    def extract_features_optimized(session, images, batch_size):
        """Optimize edilmiş feature extraction"""
        input_name = session.get_inputs()[0].name
        features = []
        
        use_fp16 = 'CUDAExecutionProvider' in session.get_providers()
        
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size]
            
            if use_fp16:
                batch = batch.astype(np.float16)
            else:
                batch = batch.astype(np.float32)
            
            feats = session.run(None, {input_name: batch})[0]
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
            features.append(feats)
        
        features = np.vstack(features)
        return features
    
    def gpu_euclidean_dist_batch(x, y, batch_size=512):
        """Memory-efficient batch euclidean distance computation on GPU"""
        m, n = x.shape[0], y.shape[0]
        device = x.device
        
        # Eğer küçükse direkt hesapla
        if m * n < 1e7:  # ~40MB for float32
            xx = (x * x).sum(dim=1, keepdim=True)
            yy = (y * y).sum(dim=1, keepdim=True)
            dist = xx + yy.t() - 2 * torch.mm(x, y.t())
            dist = dist.clamp(min=1e-12).sqrt()
            return dist
        
        # Büyükse batch'le
        dist = torch.zeros(m, n, device=device)
        for i in range(0, m, batch_size):
            end_i = min(i + batch_size, m)
            batch_x = x[i:end_i]
            
            xx = (batch_x * batch_x).sum(dim=1, keepdim=True)
            yy = (y * y).sum(dim=1, keepdim=True)
            
            batch_dist = xx + yy.t() - 2 * torch.mm(batch_x, y.t())
            batch_dist = batch_dist.clamp(min=1e-12).sqrt()
            dist[i:end_i] = batch_dist
            
        return dist
    
    def gpu_reranking(qf, gf, k1=20, k2=6, lambda_value=0.3, batch_size=256):
        """
        GPU-based re-ranking implementation
        Reference: Re-ranking Person Re-identification with k-reciprocal Encoding
        """
        device = qf.device
        query_num = qf.shape[0]
        gallery_num = gf.shape[0]
        
        print("Computing initial distances...")
        # Initial distance matrix
        if query_num + gallery_num < 10000:
            # Küçük dataset için tüm feature'ları birleştir
            feat = torch.cat([qf, gf], dim=0)
            dist = gpu_euclidean_dist_batch(feat, feat, batch_size=batch_size)
            original_dist = dist[:query_num, query_num:].clone()
        else:
            # Büyük dataset için ayrı hesapla
            original_dist = gpu_euclidean_dist_batch(qf, gf, batch_size=batch_size)
            dist_qq = gpu_euclidean_dist_batch(qf, qf, batch_size=batch_size)
            dist_gg = gpu_euclidean_dist_batch(gf, gf, batch_size=batch_size)
        
        print("Computing k-reciprocal features...")
        # k-reciprocal features için batch processing
        V = torch.zeros(query_num, gallery_num, device=device)
        
        # Query için k-reciprocal
        for i in tqdm(range(0, query_num, batch_size), desc="Query k-reciprocal"):
            end_i = min(i + batch_size, query_num)
            batch_size_i = end_i - i
            
            # Her query için k1 nearest neighbors
            if query_num + gallery_num < 10000:
                batch_dist = dist[i:end_i]
            else:
                batch_dist = torch.cat([
                    dist_qq[i:end_i], 
                    original_dist[i:end_i]
                ], dim=1)
            
            # k1 nearest neighbors
            _, initial_rank = batch_dist.topk(k1+1, dim=1, largest=False, sorted=True)
            
            # Her query için işlem
            for j in range(batch_size_i):
                q_idx = i + j
                k_reciprocal_index = initial_rank[j]
                
                # k-reciprocal expansion
                k_reciprocal_expansion_index = k_reciprocal_index.clone()
                for candidate in k_reciprocal_index[:k2]:
                    if candidate >= query_num:  # Gallery'de
                        candidate_idx = candidate - query_num
                        if query_num + gallery_num < 10000:
                            candidate_dist = dist[candidate]
                        else:
                            candidate_dist = torch.cat([
                                original_dist[:, candidate_idx],
                                dist_gg[candidate_idx]
                            ])
                    else:  # Query'de
                        if query_num + gallery_num < 10000:
                            candidate_dist = dist[candidate]
                        else:
                            candidate_dist = torch.cat([
                                dist_qq[candidate],
                                original_dist[candidate]
                            ])
                    
                    _, candidate_rank = candidate_dist.topk(k1+1, largest=False)
                    k_reciprocal_expansion_index = torch.unique(
                        torch.cat([k_reciprocal_expansion_index, candidate_rank[:int(k1/2)+1]])
                    )
                
                # Weight calculation
                weight = torch.exp(-batch_dist[j, k_reciprocal_expansion_index])
                V[q_idx, k_reciprocal_expansion_index[k_reciprocal_expansion_index >= query_num] - query_num] = \
                    weight[k_reciprocal_expansion_index >= query_num] / weight.sum()
        
        print("Computing Jaccard distance...")
        # Jaccard distance - batch processing
        jaccard_dist = torch.zeros(query_num, gallery_num, device=device)
        
        for i in tqdm(range(0, query_num, batch_size), desc="Jaccard distance"):
            end_i = min(i + batch_size, query_num)
            batch_V_q = V[i:end_i]
            
            for j in range(0, gallery_num, batch_size):
                end_j = min(j + batch_size, gallery_num)
                batch_V_g = V[:, j:end_j].t()  # (batch_j, query_num)
                
                # Compute min and max
                for qi in range(end_i - i):
                    for gi in range(end_j - j):
                        v_qi = batch_V_q[qi]
                        v_gi = V[:, j + gi]
                        
                        # Jaccard = 1 - (intersection / union)
                        min_vg_vq = torch.minimum(v_qi, v_gi)
                        max_vg_vq = torch.maximum(v_qi, v_gi)
                        
                        jaccard_dist[i + qi, j + gi] = 1 - min_vg_vq.sum() / (max_vg_vq.sum() + 1e-12)
        
        # Final distance
        final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
        
        return final_dist
    
    def compute_metrics_with_reranking(qf, gf, q_pids, g_pids, q_camids, g_camids, 
                                      batch_size, topk, use_camera_filter, use_reranking,
                                      rerank_k1, rerank_k2, rerank_lambda):
        """Metrik hesaplama with optional re-ranking"""
        device = qf.device
        num_query = qf.shape[0]
        max_rank = max(topk)
        
        if use_reranking:
            print("\nApplying re-ranking...")
            # Re-ranking ile mesafe matrisi
            distmat = gpu_reranking(qf, gf, k1=rerank_k1, k2=rerank_k2, 
                                   lambda_value=rerank_lambda, batch_size=batch_size)
        else:
            # Normal mesafe matrisi
            distmat = gpu_euclidean_dist_batch(qf, gf, batch_size=batch_size)
        
        # Camera filtering
        if use_camera_filter:
            for i in range(num_query):
                mask = (g_pids == q_pids[i]) & (g_camids == q_camids[i])
                distmat[i, mask] = float('inf')
        
        print("\nComputing final metrics...")
        total_cmc = torch.zeros(max_rank, device=device)
        total_ap = 0.0
        valid_queries = 0
        
        # Batch processing for metrics
        for i in tqdm(range(0, num_query, batch_size), desc="Computing metrics"):
            end_i = min(i + batch_size, num_query)
            batch_distmat = distmat[i:end_i]
            batch_q_pids = q_pids[i:end_i]
            
            # Sıralama
            indices = torch.argsort(batch_distmat, dim=1)
            matches = (g_pids[indices] == batch_q_pids.view(-1, 1)).int()
            
            # Geçerli sorguları bul
            num_rel = matches.sum(dim=1)
            has_match = num_rel > 0
            
            if not has_match.any():
                continue
            
            valid_queries += has_match.sum().item()
            
            # AP hesapla
            positions = torch.arange(1, matches.shape[1] + 1, device=device, dtype=torch.float32)
            positions = positions.expand_as(matches)
            
            precision = matches.cumsum(dim=1).float() / positions
            ap = (precision * matches).sum(dim=1) / num_rel.clamp(min=1).float()
            total_ap += ap[has_match].sum().item()
            
            # CMC hesapla
            cmc = matches.cumsum(dim=1).clamp(max=1)
            total_cmc += cmc[has_match, :max_rank].sum(dim=0)
        
        return total_ap, total_cmc, valid_queries
    
    # Ana pipeline
    print("="*60)
    print("Optimized ReID Evaluation Pipeline with Re-ranking")
    print("="*60)
    
    # ONNX session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Using providers: {session.get_providers()}")
    
    # Görüntüleri yükle
    print("\nLoading images...")
    query_images, query_pids, query_camids, query_fnames = parallel_load_images(
        query_dir, transform, num_workers
    )
    gallery_images, gallery_pids, gallery_camids, gallery_fnames = parallel_load_images(
        gallery_dir, transform, num_workers
    )
    
    print(f"Query: {len(query_images)} images")
    print(f"Gallery: {len(gallery_images)} images")
    
    # Feature extraction
    print("\nExtracting features...")
    query_features = extract_features_optimized(session, query_images, batch_size)
    gallery_features = extract_features_optimized(session, gallery_images, batch_size)
    
    # GPU'ya taşı
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluation device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Memory before evaluation: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    qf = torch.from_numpy(query_features).float().to(device)
    gf = torch.from_numpy(gallery_features).float().to(device)
    q_pids = torch.from_numpy(query_pids).long().to(device)
    g_pids = torch.from_numpy(gallery_pids).long().to(device)
    q_camids = torch.from_numpy(query_camids).long().to(device)
    g_camids = torch.from_numpy(gallery_camids).long().to(device)
    
    # Metrikleri hesapla
    print("\nComputing metrics...")
    total_ap, total_cmc, valid_queries = compute_metrics_with_reranking(
        qf, gf, q_pids, g_pids, q_camids, g_camids,
        eval_batch_size, topk, use_camera_filter, use_reranking,
        rerank_k1, rerank_k2, rerank_lambda
    )
    
    # Final sonuçlar
    if valid_queries == 0:
        print("ERROR: No valid queries found!")
        return 0.0, {k: 0.0 for k in topk}, {}
    
    mAP = total_ap / valid_queries
    cmc_scores = (total_cmc / valid_queries).cpu().numpy()
    cmc_dict = {k: float(cmc_scores[k-1]) for k in topk}
    
    # GPU bellek kullanımı
    if device.type == 'cuda':
        print(f"\nPeak GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        torch.cuda.empty_cache()
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mAP: {mAP:.4f}")
    for k, score in cmc_dict.items():
        print(f"Rank-{k}: {score:.4f}")
    print(f"Valid queries: {valid_queries}/{len(query_pids)}")
    print(f"Re-ranking: {'Enabled' if use_reranking else 'Disabled'}")
    if use_reranking:
        print(f"Re-ranking params: k1={rerank_k1}, k2={rerank_k2}, lambda={rerank_lambda}")
    
    results = {
        'mAP': mAP,
        'cmc_scores': cmc_dict,
        'num_query': len(query_features),
        'num_gallery': len(gallery_features),
        'valid_queries': valid_queries,
        'unique_query_pids': len(np.unique(query_pids.cpu())),
        'unique_gallery_pids': len(np.unique(gallery_pids.cpu())),
        're_ranking_used': use_reranking
    }
    
    # Feature'ları kaydet
    if save_features and features_dir:
        os.makedirs(features_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(features_dir, "query_features.npz"),
            features=query_features,
            pids=query_pids.cpu().numpy(),
            camids=query_camids.cpu().numpy(),
            fnames=query_fnames
        )
        np.savez_compressed(
            os.path.join(features_dir, "gallery_features.npz"),
            features=gallery_features,
            pids=gallery_pids.cpu().numpy(),
            camids=gallery_camids.cpu().numpy(),
            fnames=gallery_fnames
        )
        print(f"\nFeatures saved to: {features_dir}")
    
    return mAP, cmc_dict, results


# Daha da optimize edilmiş re-ranking için alternatif implementasyon
def gpu_reranking_optimized(qf, gf, k1=20, k2=6, lambda_value=0.3, batch_size=128):
    """
    Ultra memory-efficient GPU re-ranking
    Daha az bellek kullanır ama biraz daha yavaş olabilir
    """
    device = qf.device
    query_num = qf.shape[0]
    gallery_num = gf.shape[0]
    feat_dim = qf.shape[1]
    
    print("Computing initial distances...")
    
    # Original distance matrix - batch computed
    original_dist = torch.zeros(query_num, gallery_num, device=device)
    
    for i in range(0, query_num, batch_size):
        end_i = min(i + batch_size, query_num)
        original_dist[i:end_i] = torch.cdist(qf[i:end_i], gf, p=2)
    
    # k-reciprocal features
    print("Computing k-reciprocal features...")
    
    # Sparse representation için
    V_rows = []
    V_cols = []
    V_vals = []
    
    for i in tqdm(range(query_num), desc="Query k-reciprocal"):
        # Query-gallery distances
        q_g_dist = original_dist[i]
        
        # Query-query distances (sadece bu query için)
        q_q_dist = torch.cdist(qf[i:i+1], qf, p=2).squeeze(0)
        
        # Combined distances
        q_dist = torch.cat([q_q_dist, q_g_dist])
        
        # Initial ranking
        _, initial_rank = q_dist.topk(k1+1, largest=False)
        
        # k-reciprocal neighbors
        k_reciprocal_index = []
        
        for idx in initial_rank[:k1+1]:
            if idx == i:  # Skip self
                continue
                
            # Reciprocal check
            if idx < query_num:  # Query
                idx_dist = torch.cat([
                    torch.cdist(qf[idx:idx+1], qf, p=2).squeeze(0),
                    original_dist[idx]
                ])
            else:  # Gallery
                g_idx = idx - query_num
                idx_dist = torch.cat([
                    torch.cdist(gf[g_idx:g_idx+1], qf, p=2).squeeze(0),
                    torch.cdist(gf[g_idx:g_idx+1], gf, p=2).squeeze(0)
                ])
            
            _, idx_rank = idx_dist.topk(k1+1, largest=False)
            
            if i in idx_rank or (i + query_num) in idx_rank:
                k_reciprocal_index.append(idx)
        
        if len(k_reciprocal_index) == 0:
            continue
        
        k_reciprocal_index = torch.tensor(k_reciprocal_index, device=device)
        
        # k-reciprocal expansion
        k_reciprocal_expansion_index = k_reciprocal_index.clone()
        
        for j in k_reciprocal_index[:min(k2, len(k_reciprocal_index))]:
            if j < query_num:
                j_dist = torch.cat([
                    torch.cdist(qf[j:j+1], qf, p=2).squeeze(0),
                    original_dist[j]
                ])
            else:
                g_j = j - query_num
                j_dist = torch.cat([
                    torch.cdist(gf[g_j:g_j+1], qf, p=2).squeeze(0),
                    torch.cdist(gf[g_j:g_j+1], gf, p=2).squeeze(0)
                ])
            
            _, j_rank = j_dist.topk(int(k1/2)+1, largest=False)
            k_reciprocal_expansion_index = torch.unique(
                torch.cat([k_reciprocal_expansion_index, j_rank])
            )
        
        # Weight calculation
        weight = torch.exp(-q_dist[k_reciprocal_expansion_index])
        weight = weight / weight.sum()
        
        # Store in sparse format
        gallery_indices = k_reciprocal_expansion_index[k_reciprocal_expansion_index >= query_num] - query_num
        if len(gallery_indices) > 0:
            for idx, g_idx in enumerate(gallery_indices):
                idx_in_expansion = (k_reciprocal_expansion_index == g_idx + query_num).nonzero()[0]
                V_rows.append(i)
                V_cols.append(g_idx.item())
                V_vals.append(weight[idx_in_expansion].item())
    
    # Create sparse V matrix
    V_indices = torch.tensor([V_rows, V_cols], device=device)
    V_values = torch.tensor(V_vals, device=device)
    V = torch.sparse_coo_tensor(V_indices, V_values, (query_num, gallery_num), device=device)
    V = V.to_dense()
    
    print("Computing Jaccard distance...")
    # Jaccard distance computation
    jaccard_dist = torch.zeros(query_num, gallery_num, device=device)
    
    # Batch Jaccard computation
    for i in tqdm(range(0, query_num, batch_size), desc="Jaccard distance"):
        end_i = min(i + batch_size, query_num)
        
        for j in range(end_i - i):
            q_idx = i + j
            v_qi = V[q_idx]
            
            # Vectorized Jaccard
            v_g = V  # All gallery vectors
            
            # Min and max operations
            v_min = torch.minimum(v_qi.unsqueeze(0), v_g)
            v_max = torch.maximum(v_qi.unsqueeze(0), v_g)
            
            # Jaccard distance
            jaccard_dist[q_idx] = 1 - v_min.sum(dim=1) / (v_max.sum(dim=1) + 1e-12)
    
    # Final distance
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
    
    return final_dist


# Kullanım örneği
if __name__ == "__main__":
    # Basit kullanım
    mAP, cmc_scores, results = evaluate_reid_pipeline_optimized(
        onnx_path="/path/to/model.onnx",
        query_dir="/path/to/query",
        gallery_dir="/path/to/gallery",
        batch_size=256,
        eval_batch_size=512,
        topk=(1, 5, 10, 20),
        use_gpu=True,
        use_camera_filter=True,
        use_reranking=True,
        rerank_k1=20,
        rerank_k2=6,
        rerank_lambda=0.3
    )
    
    print(f"\nFinal Results:")
    print(f"mAP: {mAP:.4f}")
    print(f"Rank-1: {cmc_scores[1]:.4f}")
    print(f"Rank-5: {cmc_scores[5]:.4f}")
    print(f"Rank-10: {cmc_scores[10]:.4f}")