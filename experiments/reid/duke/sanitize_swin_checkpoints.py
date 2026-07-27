#!/usr/bin/env python3
import glob
import os

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


CHECKPOINT_DIR = "/mnt/2tb_ssd/TwinProject/experiments/reid/duke/results_swin_plain/train"


def clean_object(obj):
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: clean_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_object(v) for v in obj]
    return obj


def main():
    checkpoints = sorted(
        p for p in glob.glob(os.path.join(CHECKPOINT_DIR, "model_epoch_*.pth"))
        if not os.path.basename(p).startswith("sanitized_")
    )
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {CHECKPOINT_DIR}")

    for checkpoint_path in checkpoints:
        dirname, basename = os.path.split(checkpoint_path)
        output_path = os.path.join(dirname, f"sanitized_{basename}")
        if os.path.exists(output_path):
            print(f"exists {output_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        torch.save(clean_object(checkpoint), output_path)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
