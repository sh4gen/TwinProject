import os
import json
import glob
import re
from datetime import datetime

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

class Pipe:
    def __init__(self, name):
        self.name = name
        
    def run(self):
        raise NotImplementedError
        
    def exec(self, script):
        return os.system(script)

class TrainPipe(Pipe):
    def __init__(self, config_path):
        super().__init__("train")
        self.config_path = config_path
        
    def run(self):
        cmd = f"tao model re_identification train -e {self.config_path}"
        self.exec(cmd)

class ExportPipe(Pipe):
    def __init__(self, export_config_path, checkpoint_dir, export_dir, num_classes=75):
        super().__init__("export")
        self.export_config_path = export_config_path
        self.checkpoint_dir = checkpoint_dir
        self.export_dir = export_dir
        self.num_classes = num_classes
        
    def run(self):
        checkpoints = self.get_checkpoints()
        for ckpt in checkpoints:
            self.export_checkpoint(ckpt)
            
    def get_checkpoints(self):
        pattern = os.path.join(self.checkpoint_dir, "model_epoch_*_step_*.pth")
        checkpoints = glob.glob(pattern)
        filtered = []
        for ckpt in checkpoints:
            match = re.search(r'epoch_(\d+)_', ckpt)
            if match:
                epoch = int(match.group(1))
                if epoch % 5 == 4 or epoch % 10 == 9:
                    filtered.append(ckpt)
        return filtered
        
    def export_checkpoint(self, checkpoint_path):
        epoch = re.search(r'epoch_(\d+)_', checkpoint_path).group(1)
        onnx_path = os.path.join(self.export_dir, f"model_epoch_{epoch}.onnx")
        
        with open(self.export_config_path, 'r') as f:
            config = f.read()
        
        config = re.sub(r'checkpoint:.*', f'checkpoint: "{checkpoint_path}"', config)
        config = re.sub(r'onnx_file:.*', f'onnx_file: "{onnx_path}"', config)
        
        temp_config = f"/tmp/export_epoch_{epoch}.yaml"
        with open(temp_config, 'w') as f:
            f.write(config)
            
        cmd = f"tao model re_identification export -e {temp_config} dataset.num_classes={self.num_classes}"
        self.exec(cmd)
        os.remove(temp_config)

class EvaluatePipe(Pipe):
    def __init__(self, evaluate_script, onnx_dir, query_dir, gallery_dir, results_dir):
        super().__init__("evaluate")
        self.evaluate_script = evaluate_script
        self.onnx_dir = onnx_dir
        self.query_dir = query_dir
        self.gallery_dir = gallery_dir
        self.results_dir = results_dir
        
    def run(self):
        onnx_files = glob.glob(os.path.join(self.onnx_dir, "*.onnx"))
        all_results = {}
        
        for onnx_file in onnx_files:
            model_name = os.path.basename(onnx_file).replace('.onnx', '')
            result = self.evaluate_model(onnx_file)
            all_results[model_name] = result
            
        results_file = os.path.join(self.results_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
    def evaluate_model(self, onnx_path):
        cmd = f"python {self.evaluate_script} --onnx {onnx_path} --query {self.query_dir} --gallery {self.gallery_dir}"
        self.exec(cmd)
        
        result_file = f"{onnx_path.replace('.onnx', '_results.json')}"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                return json.load(f)
        return {}

def create_reid_pipeline():
    pipeline = Pipeline("reid_ccvid")
    
    train_pipe = TrainPipe("/home/ika/yzlm/TwinProject/CCVID/train_with_val.yaml")
    pipeline.add_pipe(train_pipe)
    
    export_pipe = ExportPipe(
        export_config_path="/home/ika/yzlm/TwinProject/CCVID/export.yaml",
        checkpoint_dir="/home/ika/yzlm/TwinProject/CCVID/results/train",
        export_dir="/home/ika/yzlm/TwinProject/CCVID/results/exported",
        num_classes=75
    )
    pipeline.add_pipe(export_pipe)
    
    evaluate_pipe = EvaluatePipe(
        evaluate_script="/home/ika/yzlm/TwinProject/CCVID/evaluate.py",
        onnx_dir="/home/ika/yzlm/TwinProject/CCVID/results/exported",
        query_dir="/home/ika/yzlm/TwinProject/CCVID/data/query",
        gallery_dir="/home/ika/yzlm/TwinProject/CCVID/data/bounding_box_test",
        results_dir="/home/ika/yzlm/TwinProject/CCVID/results"
    )
    pipeline.add_pipe(evaluate_pipe)
    
    return pipeline

if __name__ == "__main__":
    pipeline = create_reid_pipeline()
    pipeline.run()
    pipeline.save_results()
    