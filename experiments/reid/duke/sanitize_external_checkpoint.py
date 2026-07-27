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
    if len(sys.argv) != 3:
        raise SystemExit("Usage: sanitize_external_checkpoint.py INPUT.pth OUTPUT.pth")

    input_path, output_path = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    torch.save(clean_object(checkpoint), output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
