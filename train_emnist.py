from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


EMNIST_BALANCED_LABELS: Dict[int, str] = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "a",
    37: "b",
    38: "d",
    39: "e",
    40: "f",
    41: "g",
    42: "h",
    43: "n",
    44: "q",
    45: "r",
    46: "t",
}

NUM_CLASSES = 47


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("data")
    checkpoint_path: Path = Path("emnist_balanced_cnn.pth")
    confusion_matrix_path: Path = Path("confusion_matrix.png")
    batch_size: int = 256
    num_workers: int = 4
    learning_rate: float = 1e-3
    max_epochs: int = 30
    val_split: float = 0.15
    early_stopping_patience: int = 5
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    seed: int = 42


class FixEMNISTOrientation:
    """Fixes EMNIST image orientation before tensor conversion.

    EMNIST samples are rotated and mirrored in their raw representation.
    Matrix transpose restores the expected upright character orientation.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        if hasattr(Image, "Transpose"):
            return img.transpose(Image.Transpose.TRANSPOSE)
        return img.transpose(Image.TRANSPOSE)


class EMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            FixEMNISTOrientation(),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            FixEMNISTOrientation(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(config: TrainConfig, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms()

    train_full = datasets.EMNIST(
        root=str(config.data_dir),
        split="balanced",
        train=True,
        transform=train_transform,
        download=True,
    )
    val_full = datasets.EMNIST(
        root=str(config.data_dir),
        split="balanced",
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.EMNIST(
        root=str(config.data_dir),
        split="balanced",
        train=False,
        transform=eval_transform,
        download=True,
    )

    val_size = int(len(train_full) * config.val_split)
    train_size = len(train_full) - val_size
    split_generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train_full), generator=split_generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    common_loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader, test_loader


def create_grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    except TypeError:
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    collect_predictions: bool = False,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

        if collect_predictions:
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, y_true, y_pred


def save_confusion_matrix(y_true: List[int], y_pred: List[int], output_path: Path) -> None:
    labels = list(range(NUM_CLASSES))
    class_names = [EMNIST_BALANCED_LABELS[idx] for idx in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("EMNIST Balanced - Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_training(config: TrainConfig) -> None:
    assert len(EMNIST_BALANCED_LABELS) == NUM_CLASSES, "Expected exactly 47 class labels."

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"Mixed precision enabled: {amp_enabled}")
    print(f"Batch size: {config.batch_size}, num_workers: {config.num_workers}")

    train_loader, val_loader, test_loader = build_dataloaders(config, device)

    model = EMNISTCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
    )
    scaler = create_grad_scaler(device, enabled=amp_enabled)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )
        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            collect_predictions=False,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{config.max_epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}% | "
            f"lr={current_lr:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"Saved new best model to: {config.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Early stopping triggered "
                    f"(no val_loss improvement for {config.early_stopping_patience} epochs)."
                )
                break

    if not config.checkpoint_path.exists():
        torch.save(model.state_dict(), config.checkpoint_path)

    best_state = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        collect_predictions=True,
    )

    save_confusion_matrix(y_true, y_pred, config.confusion_matrix_path)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Final test Accuracy: {test_acc * 100:.2f}%")
    print(f"Confusion matrix saved to: {config.confusion_matrix_path}")


if __name__ == "__main__":
    run_training(TrainConfig())