"""
Pneumonia classification (NORMAL vs PNEUMONIA) with ResNet-18.

Fixes over the original script:
  - Windows-safe: everything runs under `if __name__ == "__main__"` so
    num_workers > 0 does not recursively re-run the script.
  - Real validation set: a stratified 15% split carved out of train/,
    instead of the 16-image official val/ folder.
  - Class-weighted loss to handle the ~1:3 NORMAL:PNEUMONIA imbalance.
  - Checkpoints on balanced accuracy, not raw accuracy.
  - Reports confusion matrix, sensitivity, specificity and AUC on test.
  - No horizontal flip (chest X-rays have fixed anatomical orientation).
  - Seeded for reproducibility; torch.load(weights_only=True).
"""

import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = r"D:\Pneumonia_FL\pneumonia_dataset"

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.15
NUM_WORKERS = 4          # set to 0 if you hit any multiprocessing trouble
SEED = 42
MAX_TRAIN_IMAGES = 5000    # use only ~5000 images from the full train folder

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_pneumonia_resnet18.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================================
# 3. HELPERS
# ============================================================

class TransformedSubset(Dataset):
    """A Subset that applies its own transform.

    Needed because train and val come from the same ImageFolder but must
    be augmented differently.
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.dataset[self.indices[i]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_sample_indices(targets, max_images, seed):
    """Select at most max_images while approximately preserving class proportions."""
    targets = np.asarray(targets)
    n_total = len(targets)

    if max_images is None or max_images >= n_total:
        return np.arange(n_total)

    rng = np.random.RandomState(seed)
    classes, counts = np.unique(targets, return_counts=True)

    # Allocate samples proportionally, then distribute any rounding remainder.
    desired = counts / counts.sum() * max_images
    per_class = np.floor(desired).astype(int)
    remainder = max_images - int(per_class.sum())

    if remainder > 0:
        fractional = desired - per_class
        for idx in np.argsort(-fractional)[:remainder]:
            per_class[idx] += 1

    selected = []
    for cls, n_select in zip(classes, per_class):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        selected.extend(cls_idx[:n_select])

    rng.shuffle(selected)
    return np.asarray(selected, dtype=int)


def stratified_split(targets, val_fraction, seed):
    """Return (train_indices, val_indices), class proportions preserved."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []

    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def roc_auc(y_true, scores):
    """AUC via the rank (Mann-Whitney U) formula, with tie correction."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)

    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def evaluate(model, loader, criterion, device, positive_index):
    """Run the model over a loader; return loss and prediction arrays."""
    model.eval()

    total_loss = 0.0
    n = 0
    all_labels, all_preds, all_scores = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        n += labels.size(0)

        probs = torch.softmax(outputs, dim=1)[:, positive_index]
        preds = outputs.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    return (
        total_loss / n,
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_scores),
    )


def binary_metrics(y_true, y_pred, positive_index):
    """Confusion counts plus sensitivity, specificity, balanced accuracy."""
    pos = positive_index
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fn = int(((y_pred != pos) & (y_true == pos)).sum())
    fp = int(((y_pred == pos) & (y_true != pos)).sum())
    tn = int(((y_pred != pos) & (y_true != pos)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else 0.0)

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- datasets -------------------------------------------------
    train_dir = os.path.join(DATASET_PATH, "train")
    test_dir = os.path.join(DATASET_PATH, "test")

    for path in (train_dir, test_dir):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing folder: {path}")

    base_train = datasets.ImageFolder(train_dir, transform=None)
    classes = base_train.classes
    positive_index = classes.index("PNEUMONIA") if "PNEUMONIA" in classes else 1

    # Select a reproducible, approximately class-proportional subset from the
    # full training folder. The test folder is never sampled or modified.
    selected_idx = stratified_sample_indices(
        base_train.targets, MAX_TRAIN_IMAGES, SEED
    )
    selected_targets = np.asarray(base_train.targets)[selected_idx]

    train_rel_idx, val_rel_idx = stratified_split(
        selected_targets, VAL_FRACTION, SEED
    )

    train_idx = selected_idx[train_rel_idx]
    val_idx = selected_idx[val_rel_idx]

    train_dataset = TransformedSubset(base_train, train_idx, train_transform)
    val_dataset = TransformedSubset(base_train, val_idx, eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    print("\nClasses:", classes, "| positive =", classes[positive_index])
    print("Selected training-pool images:", len(selected_idx))
    print("Train images:", len(train_dataset))
    print("Val images:  ", len(val_dataset))
    print("Test images: ", len(test_dataset))

    selected_class_counts = Counter(selected_targets)
    print("Selected class counts:",
          {classes[c]: n for c, n in sorted(selected_class_counts.items())})

    train_targets = np.asarray(base_train.targets)[train_idx]
    print("Train class counts:",
          {classes[c]: n for c, n in sorted(Counter(train_targets).items())})

    # ---- loaders --------------------------------------------------
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # ---- model ----------------------------------------------------
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    # ---- weighted loss --------------------------------------------
    counts = np.bincount(train_targets, minlength=len(classes)).astype(float)
    counts[counts == 0] = 1.0
    class_weights = counts.sum() / (len(classes) * counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    print("Class weights:",
          {classes[i]: round(float(w), 3) for i, w in enumerate(class_weights)})

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # ---- training -------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_balanced_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        val_loss, y_true, y_pred, _ = evaluate(
            model, val_loader, criterion, device, positive_index
        )
        m = binary_metrics(y_true, y_pred, positive_index)
        scheduler.step(m["balanced_accuracy"])

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.2f}% | "
            f"val_loss {val_loss:.4f} val_acc {100 * m['accuracy']:.2f}% "
            f"sens {100 * m['sensitivity']:.2f}% "
            f"spec {100 * m['specificity']:.2f}% "
            f"bal_acc {100 * m['balanced_accuracy']:.2f}%"
        )

        if m["balanced_accuracy"] > best_val_balanced_acc:
            best_val_balanced_acc = m["balanced_accuracy"]
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> best model saved")

    # ---- test -----------------------------------------------------
    print("\nLoading best model...")
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    _, y_true, y_pred, y_score = evaluate(
        model, test_loader, criterion, device, positive_index
    )
    m = binary_metrics(y_true, y_pred, positive_index)
    auc = roc_auc((y_true == positive_index).astype(int), y_score)

    neg_name = classes[1 - positive_index]
    pos_name = classes[positive_index]

    print("\n" + "=" * 46)
    print("TEST RESULTS")
    print("=" * 46)
    print(f"Best val balanced accuracy: {100 * best_val_balanced_acc:.2f}%")
    print()
    print("Confusion matrix (rows = true, cols = predicted)")
    print(f"{'':>12}{neg_name:>12}{pos_name:>12}")
    print(f"{neg_name:>12}{m['tn']:>12}{m['fp']:>12}")
    print(f"{pos_name:>12}{m['fn']:>12}{m['tp']:>12}")
    print()
    print(f"Accuracy         : {100 * m['accuracy']:.2f}%")
    print(f"Balanced accuracy: {100 * m['balanced_accuracy']:.2f}%")
    print(f"Sensitivity      : {100 * m['sensitivity']:.2f}%  (pneumonia found)")
    print(f"Specificity      : {100 * m['specificity']:.2f}%  (normal correct)")
    print(f"Precision        : {100 * m['precision']:.2f}%")
    print(f"F1 score         : {100 * m['f1']:.2f}%")
    print(f"ROC AUC          : {auc:.4f}")
    print("=" * 46)


if __name__ == "__main__":
    main()
