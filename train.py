"""Train a multi-input image classifier; use --demo for generated geometric examples."""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    from .ocr_model import OCRModel
    from .data import make_demo_dataset, InsuranceImageDataset
except ImportError:
    from ocr_model import OCRModel
    from data import make_demo_dataset, InsuranceImageDataset


def run_training(dataset, output_dir='outputs', epochs=10, batch_size=32,
                 lr=0.001, num_types=5, num_classes=4, seed=42):
    """An 80/20 random holdout. For real documents, split by document/customer to avoid leakage."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if len(dataset) < 10:
        raise ValueError('At least 10 samples required for this demo split')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_count = int(len(dataset) * 0.8)
    training, validation = random_split(dataset, [training_count, len(dataset) - training_count],
                                        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OCRModel(num_types=num_types, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        model.train()
        total_loss, total_count = 0.0, 0
        for images, type_info, labels in train_loader:
            images, type_info, labels = images.to(device), type_info.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images, type_info)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            total_count += labels.numel()
        model.eval()
        val_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for images, type_info, labels in valid_loader:
                images, type_info, labels = images.to(device), type_info.to(device), labels.to(device)
                logits = model(images, type_info)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.numel()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                count += labels.numel()
        history['train_loss'].append(total_loss / total_count)
        history['val_loss'].append(val_loss / count)
        history['val_accuracy'].append(correct / count)
        print(f"Epoch {epoch+1:02d}/{epochs} | train loss {history['train_loss'][-1]:.4f} | "
              f"val loss {history['val_loss'][-1]:.4f} | val accuracy {history['val_accuracy'][-1]:.3f}")

    torch.save({'model_state_dict': model.cpu().state_dict(), 'num_types': num_types,
                'num_classes': num_classes}, output_dir / 'model.pt')
    (output_dir / 'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, epochs + 1), history['train_loss'], label='Training loss')
    ax.plot(range(1, epochs + 1), history['val_loss'], label='Validation loss')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training history')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / 'loss_curve.png', dpi=160)
    plt.close(fig)
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Multi-input image classification prototype')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='Run on artificial geometric marks')
    source.add_argument('--csv', type=str, help='CSV: image_path,type_id,label')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-types', type=int, default=5)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()
    if args.demo:
        dataset = make_demo_dataset(num_types=args.num_types, num_classes=args.num_classes)
    else:
        dataset = InsuranceImageDataset(args.csv, args.num_types, args.num_classes)
    run_training(dataset, output_dir=args.output_dir, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, num_types=args.num_types,
                 num_classes=args.num_classes)


if __name__ == '__main__':
    main()
