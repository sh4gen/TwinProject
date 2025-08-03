import os
import json
from datetime import datetime

from Pipes.EvaluateTAO import EvaluatePipeTAO
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
    
    """export_pipe = ExportPipe(
        export_config_path="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/export.yaml",
        checkpoint_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.0.1/train",
        export_dir="/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.0.1/exported",
        num_classes=75
    )
    pipeline.add_pipe(export_pipe)"""
    

    CONFIG_FILE = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/ltcc.yaml"
    
    # 2. Directory where your trained .pth models are saved
    CHECKPOINT_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/results_0.1.1/train"
    
    # 3. Path to the query image folder
    QUERY_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/data/query"
    
    # 4. Path to the gallery (test) image folder
    GALLERY_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/data/bounding_box_test"
    
    # 5. Directory where you want to save the final summary and intermediate logs
    RESULTS_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/evaluation_results"
    
    # 6. Set to True or False to enable/disable re-ranking
    USE_RERANK = True

    # =================================================================
    # --- RUN THE EVALUATION ---
    # =================================================================
    
    # Initialize the evaluation pipe with your configuration
    eval_pipe = EvaluatePipeTAO(
        config_file=CONFIG_FILE,
        checkpoint_dir=CHECKPOINT_DIR,
        query_dir=QUERY_DIR,
        gallery_dir=GALLERY_DIR,
        results_dir=RESULTS_DIR,
        use_rerank=USE_RERANK
    )
    pipeline.add_pipe(eval_pipe)
    
    return pipeline

if __name__ == "__main__":
    pipeline = create_reid_pipeline()
    pipeline.run()
    pipeline.save_results()
    