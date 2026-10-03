from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pair_paths(gt_path: Path) -> tuple[Path, Path]:
    suffix = "_gt.jpg"
    if not gt_path.name.endswith(suffix):
        raise ValueError(f"UHDM ground truth must end with {suffix}: {gt_path}")
    moire_path = gt_path.with_name(gt_path.name[: -len(suffix)] + "_moire.jpg")
    if not gt_path.is_file() or not moire_path.is_file():
        raise FileNotFoundError(f"incomplete UHDM pair: {gt_path}")
    return gt_path, moire_path


def _discover_pairs(data_root: Path) -> list[Path]:
    selected = sorted(data_root.rglob("*_gt.jpg"))
    for gt_path in selected:
        _pair_paths(gt_path)
    if not selected:
        raise FileNotFoundError(f"no UHDM image pairs found under {data_root}")
    return selected


class UHDMDataset(Dataset):
    def __init__(self, args, image_list: list[Path], mode: str) -> None:
        self.args = args
        self.image_list = image_list
        self.mode = mode

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> dict[str, object]:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        gt_path, moire_path = _pair_paths(self.image_list[index])
        with Image.open(gt_path) as gt_image, Image.open(moire_path) as moire_image:
            gt_image = gt_image.convert("RGB")
            moire_image = moire_image.convert("RGB")
            if gt_image.size != moire_image.size:
                raise ValueError(f"UHDM pair has mismatched dimensions: {gt_path}")

            loader = str(getattr(self.args, "LOADER", "default")).lower()
            if self.mode == "train" and loader == "crop":
                crop_size = int(self.args.CROP_SIZE)
                width, height = gt_image.size
                if crop_size > min(width, height):
                    raise ValueError(
                        f"crop size {crop_size} exceeds image dimensions {width}x{height}"
                    )
                left = random.randint(0, width - crop_size)
                top = random.randint(0, height - crop_size)
                box = (left, top, left + crop_size, top + crop_size)
                gt_image = gt_image.crop(box)
                moire_image = moire_image.crop(box)
            elif loader == "resize":
                resize_size = int(self.args.RESIZE_SIZE)
                gt_image = gt_image.resize((resize_size, resize_size), Image.Resampling.BICUBIC)
                moire_image = moire_image.resize(
                    (resize_size, resize_size), Image.Resampling.BICUBIC
                )

            return {
                "in_img": _image_to_tensor(moire_image),
                "label": _image_to_tensor(gt_image),
                "number": gt_path.name.removesuffix("_gt.jpg"),
            }


def create_dataset(
    args,
    data_path,
    mode="train",
    sample_num=None,
    batch_size=None,
    shuffle=None,
    drop_last=None,
):
    if str(args.DATA_TYPE).upper() != "UHDM":
        raise ValueError(f"this standalone repository supports DATA_TYPE=UHDM, got {args.DATA_TYPE}")

    data_root = Path(data_path).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"UHDM dataset directory was not found: {data_root}")
    image_list = _discover_pairs(data_root)
    dataset: Dataset = UHDMDataset(args, image_list, mode=mode)
    if sample_num is not None and int(sample_num) < len(dataset):
        dataset = Subset(dataset, range(int(sample_num)))

    return DataLoader(
        dataset,
        batch_size=int(batch_size if batch_size is not None else args.BATCH_SIZE),
        shuffle=bool(mode == "train" if shuffle is None else shuffle),
        num_workers=int(args.WORKER),
        drop_last=bool(mode == "train" if drop_last is None else drop_last),
        pin_memory=torch.cuda.is_available(),
    )
