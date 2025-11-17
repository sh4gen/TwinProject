import os

class Pipe:
    def __init__(self, name):
        self.name = name
        
    def run(self):
        raise NotImplementedError
        
    def exec(self, script):
        return os.system(script)