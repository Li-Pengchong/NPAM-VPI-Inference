#!/usr/bin/env python3
"""Run five-fold ensemble inference with the nodule–pleural anatomy model (NPAM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import ResNet


CHANNELS = ("ct", "nodule_mask", "pleural_region_mask")
EXPECTED_SHAPE = (3, 96, 96, 96)


def build_model(device: torch.device) -> torch.nn.Module:
    """Construct the architecture used for all five NPAM folds."""
    return ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        n_input_channels=3,
        num_classes=2,
    ).to(device)


def load_input(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected input shape {EXPECTED_SHAPE}, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Input must contain only finite numeric values.")

    array = np.asarray(array, dtype=np.float32)
    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError("All input channels must be scaled to the range [0, 1].")
    for index, name in enumerate(CHANNELS[1:], start=1):
        values = np.unique(array[index])
        if not np.all(np.isin(values, [0.0, 1.0])):
            raise ValueError(f"{name} must be binary; observed values include {values[:10]}.")
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def find_weights(weights_dir: Path) -> list[Path]:
    expected = [weights_dir / f"fold_{fold}_best_auc.pth" for fold in range(5)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fold weights:\n" + "\n".join(missing))
    return expected


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dictionary in {path}.")
    return {key.removeprefix("module."): value for key, value in state.items()}


def predict(input_tensor: torch.Tensor, weights: list[Path], device: torch.device) -> dict:
    input_tensor = input_tensor.to(device)
    fold_scores: list[float] = []
    with torch.inference_mode():
        for path in weights:
            model = build_model(device)
            model.load_state_dict(load_state_dict(path, device), strict=True)
            model.eval()
            score = torch.softmax(model(input_tensor), dim=1)[0, 1].item()
            fold_scores.append(float(score))
            del model

    ensemble_score = float(np.mean(fold_scores))
    return {
        "channel_order": list(CHANNELS),
        "fold_scores": fold_scores,
        "ensemble_score": ensemble_score,
        "default_cutoff": 0.5,
        "classification_at_default_cutoff": int(ensemble_score >= 0.5),
        "note": "The ensemble score is a model score and should not be interpreted as a calibrated probability.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Three-channel .npy input.")
    parser.add_argument("--weights-dir", type=Path, required=True, help="Directory containing five fold weights.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu if needed.")
    device = torch.device(args.device)
    result = predict(load_input(args.input), find_weights(args.weights_dir), device)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
