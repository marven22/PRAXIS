"""Dataset and transform utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class MITIndoorFolder(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        size: Optional[int] = None,
        num_classes: int = 12,
        seed: int = 0,
    ) -> None:
        super().__init__()
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self.root = root
        self.transform = transform
        self.size = size
        self.num_classes = num_classes

        all_classes = [d.name for d in root_path.iterdir() if d.is_dir()]
        all_classes = sorted(all_classes)
        if len(all_classes) < num_classes:
            raise ValueError(
                f"Found only {len(all_classes)} classes in {root}, "
                f"need num_classes={num_classes}"
            )

        self.classes = all_classes[:num_classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        items = []
        for c in self.classes:
            cdir = root_path / c
            for fp in cdir.iterdir():
                if not fp.is_file():
                    continue
                if not fp.name.lower().endswith(DEFAULT_IMAGE_EXTS):
                    continue
                items.append((str(fp), self.class_to_idx[c]))

        rng = np.random.RandomState(seed)
        rng.shuffle(items)

        if size is not None and size < len(items):
            items = items[:size]

        if not items:
            raise ValueError(f"No images found under {root}")

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fp, y = self.items[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def make_base_transforms() -> Tuple[Callable, Callable]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, test_tf


def corruption_transform(kind: str, severity: float) -> Callable[[Image.Image], Image.Image]:
    if kind == "none" or severity <= 0:
        return lambda img: img

    if kind == "blur":
        radius = 0.5 + 2.5 * severity
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius))

    if kind == "jpeg":
        quality = int(100 - 60 * severity)

        def f(img: Image.Image) -> Image.Image:
            from io import BytesIO

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")

        return f

    if kind == "noise":

        def f(img: Image.Image) -> Image.Image:
            arr = np.asarray(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.15 * severity, arr.shape)
            arr = np.clip(arr + noise, 0, 1)
            return Image.fromarray((arr * 255).astype(np.uint8))

        return f

    raise ValueError(f"Unknown corruption type: {kind}")


def make_augmented_train_transform(
    aug_strength: float,
    corruption: str,
    severity: float,
) -> Callable[[Image.Image], torch.Tensor]:
    b = 0.1 + 0.3 * aug_strength
    c = 0.1 + 0.3 * aug_strength
    s = 0.1 + 0.3 * aug_strength
    h = 0.02 + 0.08 * aug_strength

    corr = corruption_transform(corruption, severity)

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h),
            transforms.Lambda(corr),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_train_loader(
    train_dir: str,
    size: int,
    aug_strength: float,
    corruption: str,
    severity: float,
    batch_size: int,
    num_classes: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    tf = make_augmented_train_transform(aug_strength, corruption, severity)
    ds = MITIndoorFolder(train_dir, transform=tf, size=size, num_classes=num_classes, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def build_test_loader(
    test_dir: str,
    size_test: int,
    batch_size: int,
    num_classes: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    _, tf = make_base_transforms()
    ds = MITIndoorFolder(test_dir, transform=tf, size=size_test, num_classes=num_classes, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
