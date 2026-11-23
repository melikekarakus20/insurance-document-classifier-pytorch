"""Demo data and user-supplied image/metadata CSV dataset helpers."""
from pathlib import Path
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset


def make_demo_dataset(n=320, seed=42, num_types=5, num_classes=4):
    """Artificial geometric marks; NOT actual insurance documents or OCR labels."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 1, 64, 64), dtype=np.float32)
    categories = rng.integers(0, num_classes, n)
    type_ids = rng.integers(0, num_types, n)
    for i, cls in enumerate(categories):
        image = rng.normal(0.04, 0.03, (64, 64)).astype(np.float32)
        # Each label has a visually distinct vertical-bar position.
        x = 8 + int(cls) * 13 + int(rng.integers(-2, 3))
        image[10:54, x:x + 5] += 0.85
        image[29:33, 8:56] += 0.25
        images[i, 0] = np.clip(image, 0, 1)
    type_onehot = np.eye(num_types, dtype=np.float32)[type_ids]
    return TensorDataset(torch.from_numpy(images), torch.from_numpy(type_onehot),
                         torch.from_numpy(categories.astype(np.int64)))


class InsuranceImageDataset(Dataset):
    """CSV columns: image_path,type_id,label. CSV path is relative root for images."""

    def __init__(self, csv_path, num_types, num_classes):
        self.root = Path(csv_path).resolve().parent
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError('CSV dataset is empty')
        for row in self.rows:
            if not {'image_path', 'type_id', 'label'} <= row.keys():
                raise ValueError('CSV must have columns image_path,type_id,label')
            t, y = int(row['type_id']), int(row['label'])
            if not (0 <= t < num_types and 0 <= y < num_classes):
                raise ValueError('type_id / label must be zero-based and in configured range')
        self.num_types = num_types

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = (self.root / row['image_path']).resolve()
        with Image.open(image_path) as im:
            im = im.convert('L').resize((64, 64))
            image = np.asarray(im, dtype=np.float32)[None] / 255
        type_id = int(row['type_id'])
        onehot = torch.zeros(self.num_types, dtype=torch.float32)
        onehot[type_id] = 1
        return torch.from_numpy(image.copy()), onehot, torch.tensor(int(row['label']), dtype=torch.long)
