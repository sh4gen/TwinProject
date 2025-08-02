import os
import json
from datetime import datetime

from Pipes.Evaluate import EvaluatePipe
from Pipes.Train import TrainPipe
from Pipes.Export import ExportPipe

class Pipeline:
    def __init__(self, experiment_name="0.0.0"):
        self.experiment_name = experiment_name
        self.pipes = []
        self.results = {}
        
    def add_pipe(self, pipe):
        self.pipes.append(pipe)
        
    def run(self):
        for pipe in self.pipes:
            pipe.run()
            
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"experiment_{self.experiment_name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_file}")


def create_reid_pipeline():
    pipeline = Pipeline("0.0.1")
    
    train_pipe = TrainPipe("/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/ltcc.yaml")
    pipeline.add_pipe(train_pipe)
    
    export_pipe = ExportPipe(
        export_config_path="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/export.yaml",
        checkpoint_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.0.1/train",
        export_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.0.1/exported",
        num_classes=75
    )
    pipeline.add_pipe(export_pipe)
    

    evaluate_pipe = EvaluatePipe(
        onnx_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.0.1/exported",
        query_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/query",
        gallery_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/test",
        results_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.0.1",

        feature_batch_size=256,
        eval_batch_size=1024,
        topk=(1,5,10),
        use_rerank=False
    )
    pipeline.add_pipe(evaluate_pipe)
    
    return pipeline

if __name__ == "__main__":
    pipeline = create_reid_pipeline()
    pipeline.run()
    pipeline.save_results()
    