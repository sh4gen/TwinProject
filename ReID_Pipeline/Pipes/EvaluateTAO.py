import os
import sys
import glob
import json
import subprocess
import re
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ReID_Pipeline.Pipes import __pipe_structure__ as p


class EvaluatePipeTAO(p.Pipe):
    def __init__(self, config_file, checkpoint_dir, query_dir, gallery_dir, 
                 results_dir, use_rerank=True):
        super().__init__("evaluate_tao")
        self.config_file = config_file
        self.checkpoint_dir = checkpoint_dir
        self.query_dir = query_dir
        self.gallery_dir = gallery_dir
        self.results_dir = results_dir
        self.use_rerank = use_rerank
        self.temp_output_dir = os.path.join(self.results_dir, "tao_temp_output")

    def run(self):
        os.makedirs(self.results_dir, exist_ok=True)
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "*.pth"))
        if not checkpoint_files:
            print(f"Error: No '.pth' files found in {self.checkpoint_dir}")
            return

        all_results = {}

        for pth_file in sorted(checkpoint_files):
            model_name = os.path.basename(pth_file).replace('.pth', '')
            print("-" * 50)
            print(f"Processing checkpoint: {model_name}")
            
            try:
                result = self.evaluate_model(pth_file, model_name)
                if result:
                    all_results[model_name] = result
                else:
                    print(f"Skipping results for {model_name} due to evaluation or parsing failure.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {model_name}. Error: {e}")
                # We add a sudo cleanup here too, in case of failure, to prevent stopping the whole loop
                self.cleanup_temp_dir()

        # Save all collected results into a single JSON file at the end
        final_results_file = os.path.join(self.results_dir, "evaluation_summary.json")
        with open(final_results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        # Final cleanup at the end
        self.cleanup_temp_dir()
            
        print("\n" + "=" * 50)
        print(f"All evaluation results have been saved to: {final_results_file}")
        print("=" * 50)

    def cleanup_temp_dir(self):
        """Uses sudo to forcefully remove the temporary directory and its contents."""
        if os.path.exists(self.temp_output_dir):
            print(f"Cleaning up temporary directory with sudo: {self.temp_output_dir}")
            # --- KEY CHANGE: Use sudo to remove root-owned files ---
            self.exec(f"sudo rm -rf {self.temp_output_dir}")

    def _parse_tao_output(self, log_file_path):
        """Parses the log file to extract mAP and CMC scores."""
        if not os.path.exists(log_file_path):
            return None
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Updated regex to find mAP and Rank scores in the table format
        map_match = re.search(r"mAP\s+│\s+([\d.]+)%", content)
        rank1_match = re.search(r"Rank-1\s+│\s+([\d.]+)%", content)
        rank5_match = re.search(r"Rank-5\s+│\s+([\d.]+)%", content)
        rank10_match = re.search(r"Rank-10\s+│\s+([\d.]+)%", content)

        if not all([map_match, rank1_match, rank5_match, rank10_match]):
            print(f"Warning: Could not parse mAP/Rank scores from {log_file_path}. Check the log.")
            return None

        mAP = float(map_match.group(1)) / 100.0
        rank1 = float(rank1_match.group(1)) / 100.0
        rank5 = float(rank5_match.group(1)) / 100.0
        rank10 = float(rank10_match.group(1)) / 100.0
        
        return {"mAP": mAP, "cmc": {"Rank-1": rank1, "Rank-5": rank5, "Rank-10": rank10}}

    def evaluate_model(self, pth_path, model_name):
        # Clean and create the temp directory before each run
        self.cleanup_temp_dir()
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        log_file = os.path.join(self.results_dir, f"{model_name}_evaluation.log")
        
        base_command = (
            f"tao model re_identification evaluate "
            f"-e {self.config_file} "
            f"evaluate.checkpoint={pth_path} "
            f"evaluate.query_dataset={self.query_dir} "
            f"evaluate.test_dataset={self.gallery_dir} "
            f"evaluate.results_dir={self.temp_output_dir} "
            f"re_ranking.re_ranking={self.use_rerank}"
        )
        
        full_command_with_redirect = f"{base_command} > {log_file} 2>&1"
        exit_code = self.exec(full_command_with_redirect)
        
        if exit_code != 0:
            print(f"Error: Evaluation command for {model_name} failed with exit code {exit_code}.")
            print(f"Check the log for details: {log_file}")
            return None
        
        print(f"Evaluation for {model_name} completed. Log saved to {log_file}")
        return self._parse_tao_output(log_file)


if __name__ == '__main__':
    CONFIG_FILE = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTTC+PRCC+ULIRI/combined.yaml"
    CHECKPOINT_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTTC+PRCC+ULIRI/results_0.0.1/train"
    QUERY_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTTC+PRCC+ULIRI/data/query"
    GALLERY_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTTC+PRCC+ULIRI/data/bounding_box_test"
    RESULTS_DIR = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTTC+PRCC+ULIRI/evaluation_results_0.0.1"
    USE_RERANK = False
    
    if 'DOCKER_HOST' in os.environ:
        os.environ.pop('DOCKER_HOST')
        
    eval_pipe = EvaluatePipeTAO(
        config_file=CONFIG_FILE,
        checkpoint_dir=CHECKPOINT_DIR,
        query_dir=QUERY_DIR,
        gallery_dir=GALLERY_DIR,
        results_dir=RESULTS_DIR,
        use_rerank=USE_RERANK
    )
    
    eval_pipe.run()

