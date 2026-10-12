#!/usr/bin/env python3
"""Evaluate an ESDNet checkpoint with W4A4 or W8A8 fake quantization."""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import math
import os
import random
from pathlib import Path
import hashlib
import json
import time
import sys

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from tqdm import tqdm

from quantdemoire.data import create_dataset
from quantdemoire.esdnet import ESDNet
from quantdemoire.quantization.calibration import run_two_pass_calibration
from quantdemoire.quantization.frequency_loss import (
    build_frequency_loss,
    build_reconstruction_loss,
)
from quantdemoire.quantization.pipeline import (
    collect_trainable_quant_params,
    enable_frequency_calibration,
    load_quant_state,
    project_quantization_bounds,
    replace_model_convs,
    resolve_frequency_config,
    save_quant_state,
)


PROJECT_ROOT = Path.cwd()
PATH_FIELDS = {
    "SAVE_PREFIX",
    "TRAIN_DATASET",
    "TEST_DATASET",
    "CALI_DATASET",
    "LOAD_PATH",
    "QUANT_STATE_PATH",
}


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML experiment config")
    parser.add_argument("--calibrate-only", action="store_true", help="save calibrated state without running the legacy evaluator")
    return parser.parse_args()


def load_config(path: Path) -> argparse.Namespace:
    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    flattened = {}
    for value in payload.values():
        if isinstance(value, dict):
            flattened.update(value)
    for field in PATH_FIELDS:
        value = flattened.get(field)
        if value:
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            flattened[field] = str(resolved.resolve())
    flattened["CONFIG_PATH"] = str(config_path)
    required = {
        "GPU_ID",
        "SEED",
        "WORKER",
        "SAVE_PREFIX",
        "EXP_NAME",
        "DATA_TYPE",
        "TEST_DATASET",
        "LOAD_PATH",
    }
    missing = sorted(required - flattened.keys())
    if missing:
        raise ValueError(f"missing required config fields: {', '.join(missing)}")
    return argparse.Namespace(**flattened)


def initialize(args) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
    args.TEST_RESULT_DIR = str(
        Path(args.SAVE_PREFIX) / args.EXP_NAME / "test_result"
    )
    args.NETS_DIR = str(Path(args.SAVE_PREFIX) / args.EXP_NAME / "net_checkpoints")
    Path(args.TEST_RESULT_DIR).mkdir(parents=True, exist_ok=True)

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.SEED)
    torch.backends.cudnn.deterministic = args.SEED == 0
    torch.backends.cudnn.benchmark = args.SEED != 0
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(args, model):
    load_path = Path(args.LOAD_PATH)
    if not load_path.is_file():
        raise FileNotFoundError(f"model checkpoint was not found: {load_path}")
    save_path = Path(args.TEST_RESULT_DIR) / "customer"
    save_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.TEST_RESULT_DIR) / "customer_result.log"
    checkpoint = torch.load(load_path, map_location="cpu")
    state_dict = checkpoint if load_path.suffix == ".pth" else checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    return save_path, log_path


def configure_logging(log_path: Path) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32):
    height, width = tensor.shape[-2:]
    target_height = math.ceil(height / multiple) * multiple
    target_width = math.ceil(width / multiple) * multiple
    top = (target_height - height) // 2
    bottom = target_height - height - top
    left = (target_width - width) // 2
    right = target_width - width - left
    if not any((left, right, top, bottom)):
        return tensor, (left, right, top, bottom)
    channels = [
        F.pad(tensor[:, index : index + 1], (left, right, top, bottom), value=mean)
        for index, mean in enumerate((0.3827, 0.4141, 0.3912))
    ]
    return torch.cat(channels, dim=1), (left, right, top, bottom)


def forward_batch(args, batch, model, device, save_path=None, metrics=None):
    input_image = batch["in_img"].to(device, non_blocking=True)
    target = batch["label"].to(device, non_blocking=True)
    input_image, (left, right, top, bottom) = pad_to_multiple(input_image)
    with torch.no_grad():
        prediction, _, _ = model(input_image)
    height_end = prediction.shape[-2] - bottom if bottom else prediction.shape[-2]
    width_end = prediction.shape[-1] - right if right else prediction.shape[-1]
    prediction = prediction[:, :, top:height_end, left:width_end]

    lpips_value = psnr = ssim = 0.0
    if metrics is not None:
        lpips_value, psnr, ssim = metrics.compute(prediction, target)
    if bool(getattr(args, "SAVE_IMG", False)) and save_path is not None:
        extension = args.SAVE_IMG if isinstance(args.SAVE_IMG, str) else "png"
        array = (
            prediction[0].detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        )
        Image.fromarray(array).save(Path(save_path) / f"test_{batch['number'][0]}.{extension}")
    return lpips_value, psnr, ssim


def save_current_quant_state(model, path) -> None:
    if not path:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    save_quant_state(model, state_path)
    logging.warning("save quantized parameters to %s", state_path)


def run_frequency_calibration(args, model, device, teacher_model=None):
    config = resolve_frequency_config(args)
    if (
        not config["enable"]
        and not config["finetune_enable"]
        and not config["alternate_finetune"]
    ) or config["epochs"] <= 0:
        return

    quant_params = collect_trainable_quant_params(
        model, train_smooth=config["train_smooth"]
    )
    if not quant_params:
        logging.warning("skip frequency calibration: no trainable quant parameters")
        return
    def fixed_digest():
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        for name, module in model.named_modules():
            for field in ('outlier_mask','original_outliers','quantized_weight','weight_scale','weight_zero_point'):
                if hasattr(module, field):
                    digest.update((name+'.'+field).encode())
                    digest.update(getattr(module,field).detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    frozen_before = fixed_digest()
    history = dict(config=config, trainable_parameters=[name for name,p in model.named_parameters() if p.requires_grad],
                   fixed_parameters_and_masks_sha256=frozen_before, epochs=[], complete=False)
    loader = create_dataset(
        args,
        data_path=getattr(args, "CALI_DATASET", args.TRAIN_DATASET),
        mode="train",
        sample_num=getattr(args, "FREQUENCY_SAMPLE_NUM", getattr(args, "CALI_NUM", 32)),
        batch_size=getattr(args, "FREQUENCY_BATCH_SIZE", 1),
        shuffle=True,
        drop_last=False,
    )
    frequency_criterion = build_frequency_loss(config).to(device)
    reconstruction_criterion = build_reconstruction_loss(config).to(device)
    weight_params = [
        parameter
        for module in model.modules()
        for parameter in (getattr(module, "lb_w", None), getattr(module, "ub_w", None))
        if parameter is not None and parameter.requires_grad
    ]
    weight_ids = {id(parameter) for parameter in weight_params}
    activation_params = [parameter for parameter in quant_params if id(parameter) not in weight_ids]
    groups = []
    if activation_params:
        groups.append({"params": activation_params, "lr": config["lr"]})
    if weight_params:
        weight_lr = config["weight_lr"] if config["weight_lr"] > 0 else config["lr"]
        groups.append({"params": weight_params, "lr": weight_lr})
    optimizer = optim.Adam(groups, betas=(0.9, 0.999))
    scheduler = None
    if config["scheduler"] == "cosine_restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, round(config["scheduler_t0_epochs"] * max(1, len(loader)))),
            T_mult=config["scheduler_t_mult"],
            eta_min=config["scheduler_eta_min"],
        )
    elif config["scheduler"] != "none":
        raise ValueError(f"unsupported frequency scheduler: {config['scheduler']}")

    if config["alternate_finetune"]:
        mode = "alternate"
    elif config["enable"]:
        mode = "frequency"
    else:
        mode = "reconstruction"
    logging.warning(
        "start calibration finetune: mode=%s epochs=%d lr=%.6f",
        mode,
        config["epochs"],
        config["lr"],
    )
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    for epoch in range(config["epochs"]):
        started = time.monotonic()
        learning_rates = []
        use_frequency = (
            epoch % 2 == 0 if config["alternate_finetune"] else config["enable"]
        )
        criterion = frequency_criterion if use_frequency else reconstruction_criterion
        loss_name = config["loss_type"] if use_frequency else "reconstruction"
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Finetune {epoch + 1:02d} [{loss_name}]", disable=not sys.stderr.isatty())
        for batch_index, batch in enumerate(progress):
            learning_rates.append(optimizer.param_groups[0]['lr'])
            input_image = batch["in_img"].to(device)
            target = batch["label"].to(device)
            outputs = model(input_image)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_image)
                loss = sum(
                    F.l1_loss(output, teacher_output)
                    for output, teacher_output in zip(outputs, teacher_outputs)
                )
                loss_name = "teacher_reconstruction"
            else:
                loss = criterion(*outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            project_quantization_bounds(model)
            if scheduler is not None:
                scheduler.step(epoch * len(loader) + batch_index + 1)
            total_loss += loss.item()
            if not sys.stderr.isatty() and ((batch_index+1)%20==0 or batch_index+1==len(loader)):
                logging.warning('epoch %d batch %d/%d loss %.6f lr %.8f',epoch+1,batch_index+1,
                                len(loader),total_loss/(batch_index+1),learning_rates[-1])
            progress.set_description(
                f"Finetune {epoch + 1:02d} [{loss_name}]: Loss {total_loss / (batch_index + 1):.6f}"
            )
        if fixed_digest() != frozen_before:
            raise RuntimeError('calibration changed a frozen parameter or outlier mask')
        history['epochs'].append(dict(epoch=epoch+1,loss=total_loss/max(len(loader),1),
            batches=len(loader),seconds=time.monotonic()-started,
            lr_first=learning_rates[0],lr_last=learning_rates[-1],fixed_parameters_unchanged=True))
        if getattr(args,'QUANT_STATE_PATH',None) and bool(getattr(args,'SAVE_QUANT_STATE',True)):
            path=Path(args.QUANT_STATE_PATH)
            save_current_quant_state(model,path.with_name(f'quant_state_epoch{epoch+1}.pth'))
            history['complete']=epoch+1==config['epochs']
            from quantdemoire.experiment import write_json
            write_json(path.with_name('calibration_history.json'),history)
    enable_frequency_calibration(
        model, trainable=False, train_smooth=config["train_smooth"]
    )
    model.eval()


def evaluate(args, loader, model, device, save_path, metrics) -> None:
    totals = {"lpips": 0.0, "psnr": 0.0, "ssim": 0.0}
    progress = tqdm(loader)
    count = 0
    for count, batch in enumerate(progress, start=1):
        model.eval()
        lpips_value, psnr, ssim = forward_batch(
            args, batch, model, device, save_path=save_path, metrics=metrics
        )
        if metrics is not None:
            logging.info(
                "%s: LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f",
                batch["number"][0],
                lpips_value,
                psnr,
                ssim,
            )
            totals["lpips"] += lpips_value
            totals["psnr"] += psnr
            totals["ssim"] += ssim
            progress.set_description(
                "Test: Avg. LPIPS = %.4f, Avg. PSNR = %.4f and SSIM = %.4f"
                % tuple(totals[key] / count for key in ("lpips", "psnr", "ssim"))
            )
    if count == 0:
        raise RuntimeError("test dataset is empty")
    if metrics is not None:
        logging.warning(
            "Avg. LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f",
            *(totals[key] / count for key in ("lpips", "psnr", "ssim")),
        )


def main() -> None:
    command_args = parse_command_line()
    args = load_config(command_args.config)
    device = initialize(args)
    from quantdemoire.precision import configure
    configure()
    from quantdemoire.deploy import sha256
    from quantdemoire.experiment import write_json
    if str(getattr(args,'QUANT_MODE','online')).lower()=='online':
        root=Path(getattr(args,'CALI_DATASET',args.TRAIN_DATASET))
        pairs=sorted(root.rglob('*_gt.jpg'))[:int(getattr(args,'CALI_NUM',200))]
        if len(pairs)!=int(getattr(args,'CALI_NUM',200)):
            raise ValueError('calibration dataset has fewer pairs than CALI_NUM')
        provenance=dict(config=str(command_args.config),config_sha256=sha256(command_args.config),
            checkpoint_sha256=sha256(args.LOAD_PATH),seed=args.SEED,torch=torch.__version__,
            tf32=False,calibration_pairs=[dict(image=str(p.relative_to(root)),gt_sha256=sha256(p),
                input_sha256=sha256(p.with_name(p.name.replace('_gt.jpg','_moire.jpg')))) for p in pairs],
            source_sha256={str(Path('quantdemoire') / p.relative_to(Path(__file__).parent)):sha256(p) for p in
                [Path(__file__), *sorted((Path(__file__).parent/'quantization').glob('*.py'))]})
        if getattr(args,'QUANT_STATE_PATH',None):
            write_json(Path(args.QUANT_STATE_PATH).with_name('provenance.json'),provenance)
    model = ESDNet(
        en_feature_num=args.EN_FEATURE_NUM,
        en_inter_num=args.EN_INTER_NUM,
        de_feature_num=args.DE_FEATURE_NUM,
        de_inter_num=args.DE_INTER_NUM,
        sam_number=args.SAM_NUMBER,
    ).to(device)
    save_path, log_path = load_checkpoint(args, model)
    configure_logging(log_path)
    logging.warning(datetime.datetime.now())

    teacher_model = None
    if bool(getattr(args, "CALIBRATION_TEACHER_ENABLE", False)):
        teacher_model = copy.deepcopy(model).to(device).eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    replaced = replace_model_convs(model, args)
    model = model.to(device)
    logging.warning("load model from %s", args.LOAD_PATH)
    logging.warning("save image results to %s", save_path)
    logging.warning("save logger to %s", log_path)
    logging.warning("replaced %d convolution layers with QuantizedConv2d", len(replaced))

    quant_mode = str(getattr(args, "QUANT_MODE", "online")).lower()
    quant_state_path = getattr(args, "QUANT_STATE_PATH", None)
    if quant_mode == "offline":
        if not quant_state_path:
            raise ValueError("QUANT_STATE_PATH must be set when QUANT_MODE=offline")
        if not Path(quant_state_path).is_file():
            raise FileNotFoundError(f"quantization state was not found: {quant_state_path}")
        load_quant_state(model, quant_state_path, strict=False)
        logging.warning("load quantized parameters from %s", quant_state_path)
    elif quant_mode == "online":
        calibration_forward = lambda config, batch, network, save_path, compute_metrics: forward_batch(
            config, batch, network, device
        )
        run_two_pass_calibration(args, model, calibration_forward)
        logging.warning("finished two-pass calibration")
        run_frequency_calibration(args, model, device, teacher_model=teacher_model)
        if bool(getattr(args, "SAVE_QUANT_STATE", True)):
            save_current_quant_state(model, quant_state_path)
    else:
        raise ValueError(f"unsupported QUANT_MODE: {quant_mode}")

    if command_args.calibrate_only:
        return

    metrics = None
    if bool(getattr(args, "EVALUATION_METRIC", True)):
        from quantdemoire.metrics import QualityMetrics

        metrics = QualityMetrics(device)
    args.BATCH_SIZE = 1
    loader = create_dataset(
        args,
        data_path=args.TEST_DATASET,
        mode="test",
        sample_num=getattr(args, "TEST_NUM", None),
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    evaluate(args, loader, model, device, save_path, metrics)


if __name__ == "__main__":
    main()
