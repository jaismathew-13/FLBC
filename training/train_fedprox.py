"""
Federated Learning (FedProx + FedBN) for Chest X-Ray Pneumonia Classification.
Model Architecture: ResNet-18 (ImageNet Pretrained)
Clients: 3 Simulated Non-IID Nodes (Dirichlet Partitioner)

Dataset Folder Structure:
VIVEK/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
"""

import os
import random
import csv
import copy
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ============================================================
# 1. PATHS & CONFIGURATION
# ============================================================

# Point this to your dataset folder on your workstation (e.g. r"D:\VIVEK")
DATASET_DIR = r"D:\VIVEK"

# Output directory for saving model weights and evaluation logs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_fedprox")

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_fedprox_resnet18.pth")
HISTORY_CSV_PATH = os.path.join(OUTPUT_DIR, "fedprox_training_history.csv")
RESULTS_TXT_PATH = os.path.join(OUTPUT_DIR, "fedprox_final_results.txt")
PARTITION_CSV_PATH = os.path.join(OUTPUT_DIR, "client_partitions.csv")


# ============================================================
# 2. FEDERATED LEARNING HYPERPARAMETERS
# ============================================================

NUM_CLIENTS = 3            # 3 simulated clients
NUM_ROUNDS = 20            # Total global aggregation rounds
LOCAL_EPOCHS = 3           # Local training epochs per client per round
CLIENT_FRACTION = 1.0      # 1.0 = all 3 clients participate every round

# FedProx proximal coefficient (tuned for fine-tuning ResNet-18)
MU = 1e-3                  

# Non-IID Dirichlet concentration parameter (lower = more Non-IID skew)
DIRICHLET_ALPHA = 0.5      


# ============================================================
# 3. OPTIMIZATION & TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3       # Standard for SGD fine-tuning
MOMENTUM = 0.9             # SGD momentum
WEIGHT_DECAY = 1e-4        # L2 regularization
NUM_WORKERS = 2            # Adjust based on workstation CPU cores
SEED = 42


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 5. DATA AUGMENTATION & TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# ============================================================
# 6. NON-IID DIRICHLET DATASET PARTITIONING
# ============================================================

def create_non_iid_dirichlet_partitions(dataset, num_clients=3, alpha=0.5, seed=42):
    """
    Partitions the training dataset across clients using a Dirichlet distribution
    over class labels to simulate realistic Non-IID hospital data distributions.
    """
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Prevent zero-sample partitions
        proportions = np.array([p * (len(idx_j) < len(targets) / num_clients) for p, idx_j in zip(proportions, client_indices)])
        if proportions.sum() == 0:
            proportions = np.repeat(1.0 / num_clients, num_clients)
        else:
            proportions = proportions / proportions.sum()
            
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
            
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    return client_indices


def save_and_print_partitions(dataset, client_indices):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    
    print("\n" + "=" * 70)
    print("FEDERATED CLIENT NON-IID PARTITION SUMMARY")
    print("=" * 70)
    
    for client_id, indices in enumerate(client_indices, start=1):
        client_targets = [dataset.targets[i] for i in indices]
        counts = Counter(client_targets)
        total = len(indices)
        
        normal_count = counts.get(0, 0)
        pneumonia_count = counts.get(1, 0)
        p_rate = (pneumonia_count / total * 100) if total > 0 else 0
        
        print(f"Client {client_id}: Total={total:4d} images | "
              f"NORMAL (0)={normal_count:4d} | "
              f"PNEUMONIA (1)={pneumonia_count:4d} | "
              f"Pneumonia Rate={p_rate:5.2f}%")
        
        for idx in indices:
            img_path, label = dataset.samples[idx]
            rows.append({
                "client_id": client_id,
                "image_path": img_path,
                "label": label,
                "class_name": dataset.classes[label]
            })
            
    print("=" * 70)
    pd.DataFrame(rows).to_csv(PARTITION_CSV_PATH, index=False)
    print(f"Partition details saved to: {PARTITION_CSV_PATH}\n")


# ============================================================
# 7. MODEL INITIALIZATION (ResNet-18)
# ============================================================

def create_model():
    """Initializes ResNet-18 with ImageNet weights and a 2-class binary head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


# ============================================================
# 8. FEDPROX LOCAL TRAINING (WITH FedBN & SGD & TQDM)
# ============================================================

def train_client_fedprox(
    client_id,
    round_num,
    client_model,
    global_model,
    train_loader,
    device,
    mu=MU,
    epochs=LOCAL_EPOCHS,
    lr=LEARNING_RATE
):
    """
    Executes local client training using FedProx proximal regularization + SGD with momentum.
    Displays live batch progress and metrics with tqdm.
    """
    client_model.train()
    client_model.to(device)

    optimizer = optim.SGD(
        client_model.parameters(),
        lr=lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()

    # Cache trainable global parameters (excluding BN layers) for proximal loss calculation
    global_params = {
        name: param.detach().clone().to(device)
        for name, param in global_model.named_parameters()
        if "bn" not in name and param.requires_grad
    }

    total_samples = 0
    total_loss = 0.0
    total_correct = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        # TQDM progress bar per client epoch
        pbar = tqdm(
            train_loader,
            desc=f"  [R{round_num:02d}|C{client_id}] Ep {epoch+1}/{epochs}",
            leave=False,
            dynamic_ncols=True
        )

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)

            outputs = client_model(images)
            task_loss = criterion(outputs, labels)

            # FedProx Proximal Regularization Term: (mu / 2) * ||w - w_global||^2
            proximal_penalty = 0.0
            for name, param in client_model.named_parameters():
                if name in global_params:
                    proximal_penalty += torch.sum((param - global_params[name]) ** 2)

            total_step_loss = task_loss + (mu / 2.0) * proximal_penalty
            total_step_loss.backward()

            optimizer.step()

            # Tracking statistics
            preds = outputs.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            
            epoch_correct += batch_correct
            epoch_loss += total_step_loss.item() * batch_size
            epoch_samples += batch_size

            # Update live metrics on progress bar
            pbar.set_postfix({
                "Loss": f"{total_step_loss.item():.4f}",
                "Acc": f"{100.0 * epoch_correct / max(1, epoch_samples):.1f}%"
            })

        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = (total_correct / max(1, total_samples)) * 100.0

    print(f"  -> Client {client_id} Done | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")
    return client_model.state_dict()


# ============================================================
# 9. FEDERATED AGGREGATION (FedBN Strategy)
# ============================================================

def federated_aggregate_fedbn(client_states, client_sizes, global_state):
    """
    Performs Federated Averaging on Conv and Linear layers ONLY.
    Batch Normalization statistics and affine weights remain untouched (FedBN).
    """
    total_samples = sum(client_sizes)
    new_global_state = copy.deepcopy(global_state)

    for key in global_state.keys():
        # Exclude BatchNorm layers and tracking buffers from aggregation
        if "bn" in key or "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue

        weighted_param = torch.zeros_like(global_state[key], dtype=torch.float)
        for state, size in zip(client_states, client_sizes):
            weight = size / total_samples
            weighted_param += state[key].float() * weight

        new_global_state[key] = weighted_param

    return new_global_state


# ============================================================
# 10. EVALUATION & METRICS CALCULATION (WITH TQDM)
# ============================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device, desc="Evaluating"):
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_samples += labels.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    avg_loss = total_loss / max(1, total_samples)

    # Confusion matrix calculations
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = (tp / max(1, tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


# ============================================================
# 11. MAIN TRAINING PIPELINE
# ============================================================

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("FEDPROX + FedBN RESNET-18 FEDERATED LEARNING PIPELINE")
    print("=" * 70)
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset Root         : {DATASET_DIR}")
    print(f"Clients              : {NUM_CLIENTS}")
    print(f"Global Rounds        : {NUM_ROUNDS}")
    print(f"Local Epochs / Round : {LOCAL_EPOCHS}")
    print(f"FedProx Mu (μ)       : {MU}")
    print(f"Dirichlet Alpha (α)  : {DIRICHLET_ALPHA}")
    print("=" * 70 + "\n")

    # --------------------------------------------------------
    # 1. Load Datasets via torchvision.datasets.ImageFolder
    # --------------------------------------------------------
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    test_dir = os.path.join(DATASET_DIR, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    print(f"Total Training Images   : {len(full_train_dataset)}")
    print(f"Total Validation Images : {len(val_dataset)}")
    print(f"Total Test Images       : {len(test_dataset)}")
    print(f"Classes                 : {full_train_dataset.classes} (0=NORMAL, 1=PNEUMONIA)")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --------------------------------------------------------
    # 2. Partition Data Across 3 Clients
    # --------------------------------------------------------
    client_indices = create_non_iid_dirichlet_partitions(
        full_train_dataset,
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA,
        seed=SEED
    )
    save_and_print_partitions(full_train_dataset, client_indices)

    client_loaders = [
        DataLoader(
            Subset(full_train_dataset, idxs),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        for idxs in client_indices
    ]
    client_sizes = [len(idxs) for idxs in client_indices]

    # --------------------------------------------------------
    # 3. Initialize Models
    # --------------------------------------------------------
    global_model = create_model().to(device)
    
    # Client models to retain private BatchNorm layers (FedBN)
    client_models = [create_model().to(device) for _ in range(NUM_CLIENTS)]
    for cm in client_models:
        cm.load_state_dict(global_model.state_dict())

    # Initialize CSV Logging
    with open(HISTORY_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "val_loss", "val_acc", "val_balanced_acc", "val_precision", "val_recall", "val_f1"])

    best_balanced_acc = 0.0
    best_round = 0

    # --------------------------------------------------------
    # 4. Federated Training Loop
    # --------------------------------------------------------
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*30} Global Round {round_num}/{NUM_ROUNDS} {'='*30}")
        client_states = []

        for client_id in range(NUM_CLIENTS):
            # Synchronize global Conv/Linear weights to local model
            client_state = client_models[client_id].state_dict()
            global_state = global_model.state_dict()
            for k in global_state.keys():
                if "bn" not in k and "running_" not in k and "num_batches_tracked" not in k:
                    client_state[k] = global_state[k].clone()
            client_models[client_id].load_state_dict(client_state)

            # Local FedProx Training with live tqdm bar
            updated_state = train_client_fedprox(
                client_id=client_id + 1,
                round_num=round_num,
                client_model=client_models[client_id],
                global_model=global_model,
                train_loader=client_loaders[client_id],
                device=device
            )
            client_states.append(updated_state)

        # FedBN Aggregation on Global Model
        new_global_state = federated_aggregate_fedbn(
            client_states=client_states,
            client_sizes=client_sizes,
            global_state=global_model.state_dict()
        )
        global_model.load_state_dict(new_global_state)

        # Validation Evaluation with tqdm bar
        metrics = evaluate_model(global_model, val_loader, device, desc=f"Val (Round {round_num})")

        print(f"Round {round_num} Validation Summary:")
        print(f"  Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']*100:.2f}% | "
              f"Balanced Acc: {metrics['balanced_accuracy']*100:.2f}% | "
              f"Prec: {metrics['precision']*100:.2f}% | Recall: {metrics['sensitivity']*100:.2f}% | "
              f"F1: {metrics['f1']*100:.2f}%")

        # Log History
        with open(HISTORY_CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                f"{metrics['loss']:.5f}",
                f"{metrics['accuracy']:.5f}",
                f"{metrics['balanced_accuracy']:.5f}",
                f"{metrics['precision']:.5f}",
                f"{metrics['sensitivity']:.5f}",
                f"{metrics['f1']:.5f}"
            ])

        # Save Best Model Checkpoint
        if metrics["balanced_accuracy"] > best_balanced_acc:
            best_balanced_acc = metrics["balanced_accuracy"]
            best_round = round_num
            torch.save(global_model.state_dict(), MODEL_SAVE_PATH)
            print(f"  🌟 [Checkpoint] New best model saved (Balanced Acc: {best_balanced_acc*100:.2f}%)")

    # --------------------------------------------------------
    # 5. Final Evaluation on Independent Test Set
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("FEDPROX TRAINING COMPLETE - EVALUATING ON TEST SET")
    print("=" * 70)

    global_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_metrics = evaluate_model(global_model, test_loader, device, desc="Final Test Evaluation")

    print(f"Best Model from Round : {best_round}")
    print(f"Test Loss             : {test_metrics['loss']:.4f}")
    print(f"Test Accuracy         : {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Sensitivity (Recall)  : {test_metrics['sensitivity']*100:.2f}%")
    print(f"Specificity           : {test_metrics['specificity']*100:.2f}%")
    print(f"Precision             : {test_metrics['precision']*100:.2f}%")
    print(f"F1 Score              : {test_metrics['f1']*100:.2f}%")
    print("\nConfusion Matrix (Test Set):")
    print(f"{'':>15}{'NORMAL (0)':>15}{'PNEUMONIA (1)':>15}")
    print(f"{'NORMAL (0)':>15}{test_metrics['tn']:>15}{test_metrics['fp']:>15}")
    print(f"{'PNEUMONIA (1)':>15}{test_metrics['fn']:>15}{test_metrics['tp']:>15}")
    print("=" * 70)

    # Save summary report
    with open(RESULTS_TXT_PATH, "w") as f:
        f.write("FEDPROX + FedBN CHEST X-RAY CLASSIFICATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset Path: {DATASET_DIR}\n")
        f.write(f"Number of Clients: {NUM_CLIENTS}\n")
        f.write(f"Dirichlet Alpha: {DIRICHLET_ALPHA}\n")
        f.write(f"Global Rounds: {NUM_ROUNDS}\n")
        f.write(f"Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"FedProx Mu (μ): {MU}\n")
        f.write(f"Best Round: {best_round}\n\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%\n")
        f.write(f"Sensitivity: {test_metrics['sensitivity']*100:.2f}%\n")
        f.write(f"Specificity: {test_metrics['specificity']*100:.2f}%\n")
        f.write(f"Precision: {test_metrics['precision']*100:.2f}%\n")
        f.write(f"F1 Score: {test_metrics['f1']*100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {test_metrics['tn']} | FP: {test_metrics['fp']}\n")
        f.write(f"FN: {test_metrics['fn']} | TP: {test_metrics['tp']}\n")

    print(f"Results successfully saved to:\n  - Model: {MODEL_SAVE_PATH}\n  - CSV: {HISTORY_CSV_PATH}\n  - Report: {RESULTS_TXT_PATH}\n")


if __name__ == "__main__":
    main()