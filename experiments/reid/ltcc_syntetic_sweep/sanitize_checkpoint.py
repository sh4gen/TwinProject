#!/usr/bin/env python3
import os
import sys

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def clean_object(obj):
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: clean_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_object(v) for v in obj]
    return obj


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"Usage: {sys.argv[0]} INPUT_CHECKPOINT [OUTPUT_CHECKPOINT]")

    input_path = sys.argv[1]
    dirname, basename = os.path.split(input_path)
    output_path = sys.argv[2] if len(sys.argv) == 3 else os.path.join(dirname, f"sanitized_{basename}")

    if os.path.exists(output_path):
        print(output_path)
        return

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    torch.save(clean_object(checkpoint), output_path)
    print(output_path)


if __name__ == "__main__":
    main()
