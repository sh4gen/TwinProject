import os
import sys
import glob
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from __pipe_structure__ import Pipe
from ReID_Pipeline.Pipes import __evaluate_backend_alt__ as backend

class EvaluatePipe(Pipe):
    def __init__(self, onnx_dir, query_dir, gallery_dir, results_dir,
                 feature_batch_size=256, eval_batch_size=1024, topk=(1,5,10), use_rerank=False):
        super().__init__("evaluate")
        self.onnx_dir = onnx_dir
        self.query_dir = query_dir
        self.gallery_dir = gallery_dir
        self.results_dir = results_dir
        self.feature_batch_size = feature_batch_size
        self.eval_batch_size = eval_batch_size
        self.topk = topk
        self.use_rerank = use_rerank

    def run(self):
        os.makedirs(self.results_dir, exist_ok=True)
        onnx_files = glob.glob(os.path.join(self.onnx_dir, "*.onnx"))
        all_results = {}

        for onnx_file in onnx_files:
            model_name = os.path.basename(onnx_file).replace('.onnx', '')
            print(f"Evaluating {model_name} ...")
            result = self.evaluate_model(onnx_file, model_name)
            all_results[model_name] = result

        results_file = os.path.join(self.results_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to {results_file}")

    def evaluate_model(self, onnx_path, model_name):
        query_features_path = os.path.join(self.results_dir, f"{model_name}_query_features.npz")
        gallery_features_path = os.path.join(self.results_dir, f"{model_name}_gallery_features.npz")

        mAP, cmc_scores = backend.reid_pipeline(
            onnx_path=onnx_path,
            query_dir=self.query_dir,
            gallery_dir=self.gallery_dir,
            query_features_path=query_features_path,
            gallery_features_path=gallery_features_path,
            feature_batch_size=self.feature_batch_size,
            eval_batch_size=self.eval_batch_size,
            topk=self.topk,
            use_rerank=self.use_rerank
        )

        result = {
            "mAP": mAP,
            "cmc": {f"Rank-{k}": float(cmc_scores[i]) for i, k in enumerate(self.topk)}
        }
        return result

