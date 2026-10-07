"""Export a version-2 POS checkpoint, including its exact deployment contract."""

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn


def export_checkpoint(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos") or {}
    contract = infos.get("yaw_pos_contract")
    if not isinstance(contract, dict) or contract.get("version") != 2:
        raise ValueError("Expected a fresh POS version-2 checkpoint with yaw_pos_contract.")
    if contract.get("empirical_normalization") or contract.get("activation") != "elu":
        raise ValueError("Exporter requires the unnormalized ELU POS actor.")
    dimensions = [contract["actor_observations"], *contract["actor_hidden_dims"], contract["actions"]]
    layers = []
    for index, (inputs, outputs) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(inputs, outputs))
        if index < len(dimensions) - 2:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in checkpoint["model_state_dict"].items()
                           if key.startswith("actor.")}, strict=True)
    actor.eval()
    curriculum = infos.get("yaw_curriculum") or {}
    stage = curriculum.get("values", {}).get("_yaw_task_curriculum_yaw_stage")
    levels = curriculum.get("yaw_rate_levels")
    if type(stage) is not int or not isinstance(levels, list) or not 0 <= stage < len(levels):
        raise ValueError("Checkpoint has no valid trained yaw stage; export would lose the trained command limit.")
    metadata = dict(contract, trained_yaw_limit=float(levels[stage]), checkpoint_iteration=checkpoint.get("iter"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(actor, torch.zeros(1, dimensions[0]), str(output), opset_version=17, dynamo=False,
                      input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}})
    model = onnx.load(str(output))
    onnx.helper.set_model_props(model, {"yaw_pos_contract": json.dumps(metadata, separators=(",", ":"))})
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    output.with_suffix(".contract.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata = export_checkpoint(args.checkpoint, args.output)
    print(f"Exported {args.output}; trained yaw limit {metadata['trained_yaw_limit']} rad/s, contract v2.")


if __name__ == "__main__":
    main()
