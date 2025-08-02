from ReID_Pipeline.Pipes import __pipe_structure__ as p

class TrainPipe(p.Pipe):
    def __init__(self, config_path):
        super().__init__("train")
        self.config_path = config_path
        
    def run(self):
        cmd = f"tao model re_identification train -e {self.config_path}"
        self.exec(cmd)