import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIPELINE.pipeline import Pipe

class TrainPipe(Pipe):
    def __init__(self, config_path):
        super().__init__("train")
        self.config_path = config_path
        
    def run(self):
        cmd = f"tao model re_identification train -e {self.config_path}"
        self.exec(cmd)