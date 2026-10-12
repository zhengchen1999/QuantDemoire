"""Bind a checked-in execution template to newly reproduced weights."""
import hashlib
import json
from pathlib import Path

from quantdemoire.deploy import sha256
from quantdemoire.profiles import profile_environment
from script._internal.common import ROOT, read_json, write_json


def prepare_runtime(folder, bits):
    folder = Path(folder)
    if (folder / "runtime.json").exists():
        profile_environment(folder)
        return
    template_path = ROOT / f"config/release/w{bits}a{bits}.json"
    template = read_json(template_path)
    config = read_json(folder / "config.json")
    geometry = {}
    for name, spec in config["layers"].items():
        keys = ("kind", "shape", "stride", "padding", "dilation")
        if spec["kind"] == "integer":
            keys += ("bits", "padded_channels", "main_output")
        geometry[name] = {key: spec.get(key) for key in keys}
    geometry_hash = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if geometry_hash != template["geometry_sha256"]:
        raise ValueError("Model geometry differs from the ESDNet execution template")
    weight_hash = sha256(folder / "model.safetensors")
    plan = dict(template["joint_plan"], format="quantdemoire-joint-plan-v1",
                model_sha256=weight_hash, status="unmeasured_proposal")
    write_json(folder / "joint_plan.json", plan)
    write_json(folder / "runtime.json", dict(
        format="quantdemoire-runtime-v1", status="unmeasured_proposal",
        model_sha256=weight_hash, config_sha256=sha256(folder / "config.json"),
        environment=template["environment"], joint_plan="joint_plan.json",
        joint_plan_sha256=sha256(folder / "joint_plan.json"),
        template_sha256=sha256(template_path),
        note="Execution settings only. Use this model's new quality, validation, speed and memory reports."))
    profile_environment(folder)
