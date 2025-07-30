import os, sys, glob, re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIPELINE.pipeline import Pipe

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