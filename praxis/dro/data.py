from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MITIndoorFolder(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        size: Optional[int] = None,
        num_classes: int = 12,
        seed: int = 0,
    ) -> None:
        self.transform = transform
        classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )[:num_classes]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        items = []
        for c in classes:
            cdir = os.path.join(root, c)
            for fn in os.listdir(cdir):
                if fn.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    items.append((os.path.join(cdir, fn), self.class_to_idx[c]))

        rng = random.Random(seed)
        rng.shuffle(items)
        if size is not None:
            items = items[:size]

        if not items:
            raise ValueError(f"No images found under {root}")

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        fp, y = self.items[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def corruption_transform(kind: str, severity: float) -> Callable[[Image.Image], Image.Image]:
    if kind == "none" or severity <= 0:
        return lambda img: img

    if kind == "blur":
        return lambda img: img.filter(ImageFilter.GaussianBlur(0.5 + 2.5 * severity))

    if kind == "jpeg":
        q = int(100 - 60 * float(severity))

        def apply_jpeg(img: Image.Image) -> Image.Image:
            from io import BytesIO

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")

        return apply_jpeg

    if kind == "noise":

        def apply_noise(img: Image.Image) -> Image.Image:
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr += np.random.normal(0, 0.15 * severity, arr.shape)
            arr = np.clip(arr, 0, 1)
            return Image.fromarray((arr * 255).astype(np.uint8))

        return apply_noise

    raise ValueError(kind)


def make_train_transform(aug: float, corruption: str, severity: float) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3 * aug, 0.3 * aug, 0.3 * aug),
            transforms.Lambda(corruption_transform(corruption, severity)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def make_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_train_loader(
    train_dir: str,
    size: int,
    aug: float,
    corruption: str,
    severity: float,
    batch_size: int,
    num_classes: int,
    seed: int,
) -> DataLoader:
    tf = make_train_transform(aug, corruption, severity)
    ds = MITIndoorFolder(train_dir, transform=tf, size=size, num_classes=num_classes, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def build_test_loader(
    test_dir: str,
    size: int,
    batch_size: int,
    num_classes: int,
    seed: int,
) -> DataLoader:
    tf = make_test_transform()
    ds = MITIndoorFolder(test_dir, transform=tf, size=size, num_classes=num_classes, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
