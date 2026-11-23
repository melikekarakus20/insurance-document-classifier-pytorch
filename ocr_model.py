"""Multi-input *image classifier* prototype (not a text transcription OCR engine)."""
import torch
from torch import nn


class OCRModel(nn.Module):
    """Classifies 64x64 grayscale images using auxiliary one-hot document type data."""

    def __init__(self, num_types: int = 5, num_classes: int = 4):
        super().__init__()
        self.num_types = num_types
        self.num_classes = num_classes
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
        )
        self.type_layer = nn.Sequential(nn.Linear(num_types, 32), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, image: torch.Tensor, type_info: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (1, 64, 64):
            raise ValueError('Expected images shaped [batch, 1, 64, 64]')
        if type_info.ndim != 2 or type_info.shape[1] != self.num_types:
            raise ValueError(f'Expected type_info shaped [batch, {self.num_types}]')
        a = self.image_layer(image)
        b = self.type_layer(type_info)
        return self.classifier(torch.cat((a, b), dim=1))
