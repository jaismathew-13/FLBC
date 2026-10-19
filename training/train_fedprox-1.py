"""
Pneumonia classification using ResNet-18 for the Chest X-Ray
Pneumonia -- Balanced Dataset -- FEDPROX (proximal local training +
FedAvg aggregation).

This is the same federated simulation as train_fedavg.py -- same
"realistic_hospitals" client partition scheme, same metrics, same
aggregation rule, now run on the folder-based Chest X-Ray Pneumonia
Balanced dataset -- with exactly ONE change: each client's LOCAL training objective gets an added
proximal term that penalizes drifting too far from the current global
model (Li et al., 2020, "Federated Optimization in Heterogeneous
Networks"):

    L_client(w) = CrossEntropy(w) + (mu / 2) * ||w - w_global||^2

This is the actual, literal definition of FedProx: it only changes the
CLIENT's local loss. The SERVER-side aggregation step is untouched --
it is still plain sample-size-weighted FedAvg averaging, exactly as in
train_fedavg.py. FedProx is specifically designed to help with non-IID,
unequal-sized clients (like the CLIENT_PROFILES hospital network built
in train_fedavg.py): without the proximal term, a high-prevalence site
training for a local epoch can drift its weights far from what a
low-prevalence site would produce, and FedAvg's blind averaging of two
very different local optima can hurt the global model more than it
helps. The proximal term keeps every client's local update anchored
closer to the shared global model, which usually converges faster and
more stably under this kind of heterogeneity.

Expected local structure (labels come from the folder names):

D:/Vivek/
    train/
        NORMAL/             # label 0
        PNEUMONIA/          # label 1
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

Important:
- Labels come from the class folder names (NORMAL -> 0, PNEUMONIA -> 1).
  No metadata CSV files are used any more.
- train/ is the pool that gets partitioned across the simulated
  clients. val/ is the central held-out validation set used after every
  round to pick the best global model (balanced accuracy). test/ is
  fully labeled and is evaluated exactly ONCE at the end with the best
  global model, so test data is never used for model selection.

FEDERATED SIMULATION DESIGN
----------------------------
- NUM_CLIENTS clients each hold a private, non-overlapping shard of
  the labeled training pairs.
- PARTITION_STRATEGY = "realistic_hospitals" (default) -> named,
                                       interpretable site profiles (a
                                       big general hospital, a referral
                                       center, a community clinic, a
                                       rural clinic, ...), each with a
                                       different case volume AND a
                                       different pneumonia prevalence
                                       derived from the dataset's
                                       ACTUAL measured prevalence
                                       (not an arbitrary random skew).
                                       Every client is guaranteed to
                                       have at least MIN_IMAGES_PER_CLIENT
                                       images AND both classes present
                                       (prevalence is clamped into
                                       PREVALENCE_CLIP), which mirrors
                                       how real clinics work: even a
                                       referral center sees plenty of
                                       normal X-rays, and even a
                                       routine clinic catches some
                                       pneumonia. See CLIENT_PROFILES.
  PARTITION_STRATEGY = "noniid_unequal" -> the same size/floor
                                       guarantees but with fully random
                                       (Dirichlet/Beta) skew instead of
                                       named profiles -- useful for
                                       stress-testing with more extreme,
                                       unlabeled heterogeneity.
  PARTITION_STRATEGY = "iid"       -> stratified-random split, equal
                                       sizes, each client's class
                                       balance mirrors the global
                                       class balance. Kept for
                                       comparison experiments.
  PARTITION_STRATEGY = "dirichlet" -> label-skewed but EQUAL-size
                                       split (Dirichlet(NON_IID_ALPHA)
                                       per class only). Kept for
                                       comparison experiments.
- Each round: the server sends the current global weights to every
  participating client, each client trains locally for LOCAL_EPOCHS
  epochs on ONLY its own data using its own class-weighted loss PLUS
  the FedProx proximal term (mu / 2) * ||w - w_global||^2 (a
  client would not know the global class balance in real FL), and
  returns its updated weights + sample count.
- The server aggregates with plain FedAvg: a sample-size-weighted
  average of client state_dicts (McMahan et al., 2017) -- FedProx
  does not change this step.
- CLIENT_FRACTION controls partial participation per round (1.0 =
  every client participates every round).
- MU controls the strength of the proximal term. mu = 0 recovers
  plain FedAvg exactly. Typical useful values are small, e.g.
  0.001-0.5; the FedProx paper found gains from values as low as 0.01
  on non-IID data. Larger mu anchors clients more tightly to the
  global model (more stability, slower local adaptation); smaller mu
  behaves more like unconstrained FedAvg.

This file is intentionally structured to mirror train_baseline_logged.py
so the two are easy to diff and compare.
"""

import copy
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = r"D:\Vivek"    # <-- root folder that contains train/ val/ test/

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# Class sub-folder name (case-insensitive) -> integer label
CLASS_FOLDERS = {"NORMAL": 0, "PNEUMONIA": 1}

MAX_TRAIN_IMAGES = None       # None = use every image in train/. Set a number to
                               # use a stratified subsample of train/ instead.

# --- Federated learning settings ---
NUM_CLIENTS = 3
NUM_ROUNDS = 20
LOCAL_EPOCHS =2           # local epochs per client, per round
CLIENT_FRACTION = 1.0         # fraction of clients sampled each round

PARTITION_STRATEGY = "realistic_hospitals"  # "realistic_hospitals" (default), "noniid_unequal", "iid", or "dirichlet"
MIN_IMAGES_PER_CLIENT = None            # None = auto (50% of an equal share of train/, so it
                                         # always fits the dataset size). Or set an int floor.

# "realistic_hospitals": named, interpretable site profiles. size_weight
# controls relative case volume (renormalized to sum to 1); prevalence_multiplier
# scales the dataset's ACTUAL measured pneumonia prevalence up or down per site
# (1.0 = same as the overall dataset). Must have exactly NUM_CLIENTS entries.
# NOTE (balanced dataset): overall prevalence is ~0.5, so multipliers are kept
# <= 1.3 (0.5 * 1.3 = 0.65 = upper PREVALENCE_CLIP); larger values would all clip
# to the same 0.65 and the sites would stop being distinct.
CLIENT_PROFILES = [
    {"name": "City_General_Hospital",      "size_weight": 0.45, "prevalence_multiplier": 1.00},
    {"name": "Suburban_Community_Clinic",   "size_weight": 0.25, "prevalence_multiplier": 0.50},
    {"name": "Respiratory_Referral_Center", "size_weight": 0.30, "prevalence_multiplier": 1.30},
]
PREVALENCE_NOISE_STD = 0.03    # small per-site random variation around each target prevalence
PREVALENCE_CLIP = (0.05, 0.65) # every client's pneumonia fraction is clamped into this
                                # range, so EVERY client always contains both classes --
                                # no site ends up 0% or 100% pneumonia.

# "noniid_unequal": fully random alternative (no named profiles) -- kept for
# ablation/comparison experiments.
SIZE_ALPHA = 0.5                        # Dirichlet concentration for client SIZE skew
                                         # (lower = more unequal client sizes)
LABEL_ALPHA = 0.3                       # Beta concentration for per-client LABEL skew
                                         # (lower = more extreme pneumonia/normal skew)
NON_IID_ALPHA = 0.5            # only used when PARTITION_STRATEGY == "dirichlet"

# --- FedProx setting ---
MU = 0.01                      # proximal term strength. mu=0 -> identical to plain FedAvg.
                                # Typical useful range: 0.001-0.5 (FedProx paper).

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 0                # Windows-safe
SEED = 42                      # Seed for client partitioning, init and shuffling

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_fedprox")
MODEL_PATH = os.path.join(MODEL_DIR, "best_pneumonia_resnet18_fedprox.pth")
TEST_PREDICTIONS_PATH = os.path.join(MODEL_DIR, "test_predictions_fedprox.csv")
RESULTS_PATH = os.path.join(MODEL_DIR, "fedprox_results.txt")
HISTORY_PATH = os.path.join(MODEL_DIR, "fedprox_history.csv")
CLIENT_SUMMARY_PATH = os.path.join(MODEL_DIR, "fedprox_client_partition.csv")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# 4. FILE HELPERS
# ============================================================

def image_files(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing folder: {folder}")

    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# 5. FOLDER -> IMAGE PAIRS
# ============================================================

def load_split_pairs(split_dir):
    """
    Read one split folder (train/, val/ or test/) laid out as

        split_dir/NORMAL/*.jpg
        split_dir/PNEUMONIA/*.jpg

    Returns:
        list of (image_path, label)
        label 0 = NORMAL
        label 1 = PNEUMONIA
    """
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    # Case-insensitive lookup so "Normal", "normal" and "NORMAL" all work.
    subfolders = {
        name.lower(): name
        for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    }

    pairs = []

    for class_name, label in CLASS_FOLDERS.items():
        actual = subfolders.get(class_name.lower())

        if actual is None:
            raise FileNotFoundError(
                f"Missing '{class_name}' folder inside {split_dir} "
                f"(found: {sorted(subfolders.values())})"
            )

        for path in image_files(os.path.join(split_dir, actual)):
            pairs.append((path, label))

    counts = Counter(label for _, label in pairs)

    if not pairs:
        raise RuntimeError(f"No images found in {split_dir}")

    if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
        raise RuntimeError(
            f"Both classes must be present in {split_dir}; found {dict(counts)}"
        )

    return pairs


# ============================================================
# 6. DATASET
# ============================================================

class PneumoniaDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label = self.pairs[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not open image: {image_path}") from exc

        image = self.transform(image)

        return image, label


# ============================================================
# 7. OPTIONAL STRATIFIED SUBSAMPLING
# ============================================================

def stratified_sample(pairs, max_images, seed):
    """Select at most max_images while preserving class proportions."""
    if max_images is None or max_images >= len(pairs):
        return pairs

    if max_images < 2:
        raise ValueError("MAX_TRAIN_IMAGES must be at least 2.")

    rng = np.random.RandomState(seed)

    normal = [p for p in pairs if p[1] == 0]
    pneumonia = [p for p in pairs if p[1] == 1]

    if not normal or not pneumonia:
        raise RuntimeError(
            "Both classes must be present before stratified sampling."
        )

    total = len(normal) + len(pneumonia)

    n_normal = round(max_images * len(normal) / total)
    n_normal = max(1, min(n_normal, len(normal)))

    n_pneumonia = max_images - n_normal

    if n_pneumonia < 1:
        n_pneumonia = 1
        n_normal = max_images - 1

    if n_pneumonia > len(pneumonia):
        n_pneumonia = len(pneumonia)
        n_normal = max_images - n_pneumonia

    if n_normal > len(normal):
        n_normal = len(normal)

    rng.shuffle(normal)
    rng.shuffle(pneumonia)

    selected = normal[:n_normal] + pneumonia[:n_pneumonia]
    rng.shuffle(selected)

    return selected


# ============================================================
# 7b. FEDERATED CLIENT PARTITIONING
# ============================================================

def partition_clients_iid(pairs, num_clients, seed):
    """
    Stratified-random partition: every client's local class balance
    approximates the global class balance (the "easy" FL setting).
    """
    rng = np.random.RandomState(seed)

    normal = [p for p in pairs if p[1] == 0]
    pneumonia = [p for p in pairs if p[1] == 1]

    rng.shuffle(normal)
    rng.shuffle(pneumonia)

    client_pairs = [[] for _ in range(num_clients)]

    for i, item in enumerate(normal):
        client_pairs[i % num_clients].append(item)

    for i, item in enumerate(pneumonia):
        client_pairs[i % num_clients].append(item)

    for c in client_pairs:
        rng.shuffle(c)

    return client_pairs


def partition_clients_dirichlet(pairs, num_clients, seed, alpha):
    """
    Non-IID partition via a Dirichlet(alpha) distribution over clients,
    applied per class. Lower alpha -> more label skew per client
    (e.g. one hospital sees mostly pneumonia cases, another mostly
    normal). This is the standard non-IID simulation used in FL
    literature (Hsu et al., 2019 "Measuring the Effects of Non-Identical
    Data Distribution for Federated Visual Classification").
    """
    rng = np.random.RandomState(seed)

    by_class = defaultdict(list)
    for item in pairs:
        by_class[item[1]].append(item)

    client_pairs = [[] for _ in range(num_clients)]

    for label, items in by_class.items():
        rng.shuffle(items)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # Convert proportions to cumulative split points.
        counts = (proportions * len(items)).astype(int)

        # Make sure every item is assigned (rounding remainder -> last client)
        counts[-1] += len(items) - counts.sum()

        start = 0
        for cid in range(num_clients):
            end = start + counts[cid]
            client_pairs[cid].extend(items[start:end])
            start = end

    for c in client_pairs:
        rng.shuffle(c)

    return client_pairs


def enforce_min_sizes(raw_shares, total_pool, num_clients, min_images):
    """
    Given fractional shares (summing to ~1) for each client, convert to
    integer target sizes that sum exactly to total_pool, then guarantee
    every client has at least min_images by "water-filling": any client
    below the floor is topped up using surplus taken proportionally from
    clients above the floor. Always feasible as long as
    total_pool >= num_clients * min_images (checked by the caller).
    """
    raw_shares = np.asarray(raw_shares, dtype=float)
    raw_shares = raw_shares / raw_shares.sum()

    sizes = np.floor(raw_shares * total_pool).astype(int)

    # Distribute the rounding remainder to the largest-share clients.
    remainder = total_pool - int(sizes.sum())
    if remainder > 0:
        order = np.argsort(-raw_shares)
        for i in range(remainder):
            sizes[order[i % num_clients]] += 1

    # Water-fill: top up any client below the floor using surplus taken
    # proportionally from clients above the floor. Feasible because
    # total_pool >= num_clients * min_images guarantees
    # sum(surplus) >= sum(deficit).
    deficit = np.maximum(0, min_images - sizes)
    total_deficit = int(deficit.sum())

    if total_deficit > 0:
        surplus = np.maximum(0, sizes - min_images)
        total_surplus = int(surplus.sum())

        take = np.floor(surplus / total_surplus * total_deficit).astype(int)
        take = np.minimum(take, surplus)

        shortfall = total_deficit - int(take.sum())
        if shortfall > 0:
            givers = np.argsort(-surplus)
            for i in range(shortfall):
                g = givers[i % num_clients]
                if take[g] < surplus[g]:
                    take[g] += 1

        sizes = sizes - take
        sizes = np.where(deficit > 0, min_images, sizes)

    # Final rounding safety net: reconcile any 1-2 image drift against
    # the largest client so the total still matches total_pool exactly.
    drift = total_pool - int(sizes.sum())
    if drift != 0:
        sizes[int(np.argmax(sizes))] += drift

    return sizes


def compute_unequal_client_sizes(total_pool, num_clients, min_images, size_alpha, rng):
    """
    Decide an UNEQUAL target sample count per client such that every
    client gets at least min_images, and all target sizes sum exactly
    to total_pool. Sizes are drawn from a Dirichlet(size_alpha)
    distribution (lower size_alpha -> more imbalance), then
    floor-enforced by enforce_min_sizes.
    """
    if total_pool < num_clients * min_images:
        raise RuntimeError(
            f"Training pool has {total_pool} images, which is not enough to "
            f"give every one of {num_clients} clients at least "
            f"{min_images} images (needs >= {num_clients * min_images}). "
            f"Increase MAX_TRAIN_IMAGES (or set it to None to use the full "
            f"dataset), or lower MIN_IMAGES_PER_CLIENT / NUM_CLIENTS."
        )

    raw_shares = rng.dirichlet(alpha=[size_alpha] * num_clients)
    return enforce_min_sizes(raw_shares, total_pool, num_clients, min_images)


def compute_profile_client_sizes(total_pool, num_clients, min_images, profiles):
    """
    Same floor-guaranteed sizing as compute_unequal_client_sizes, but
    driven by fixed, named hospital-size weights instead of a random
    draw, so results are interpretable and reproducible run-to-run.
    """
    if total_pool < num_clients * min_images:
        raise RuntimeError(
            f"Training pool has {total_pool} images, which is not enough to "
            f"give every one of {num_clients} clients at least "
            f"{min_images} images (needs >= {num_clients * min_images}). "
            f"Increase MAX_TRAIN_IMAGES (or set it to None to use the full "
            f"dataset), or lower MIN_IMAGES_PER_CLIENT / NUM_CLIENTS."
        )

    raw_shares = [prof["size_weight"] for prof in profiles]
    return enforce_min_sizes(raw_shares, total_pool, num_clients, min_images)


def partition_clients_noniid_unequal(pairs, num_clients, seed, label_alpha, size_alpha, min_images):
    """
    Non-IID AND unequal client partition, with a guaranteed per-client
    floor -- the realistic "cross-silo hospital" setting.

    - Client SIZES are unequal (Dirichlet(size_alpha) over clients,
      floor-enforced by compute_unequal_client_sizes).
    - Client LABEL MIX is skewed: each client's target pneumonia
      fraction is drawn from Beta(label_alpha, label_alpha). Small
      label_alpha pushes most clients toward being mostly-pneumonia or
      mostly-normal, which is the standard way FL papers simulate
      heterogeneous hospitals.
    - Allocation is done by walking clients in random order and pulling
      from shared, pre-shuffled per-class pools; if a client's desired
      class split can't be met (a class pool ran low), the shortfall is
      filled from whichever class still has capacity so every client
      still reaches its exact target size.
    """
    rng = np.random.RandomState(seed)

    normal_pool = [p for p in pairs if p[1] == 0]
    pneumonia_pool = [p for p in pairs if p[1] == 1]

    rng.shuffle(normal_pool)
    rng.shuffle(pneumonia_pool)

    total_pool = len(pairs)

    target_sizes = compute_unequal_client_sizes(
        total_pool, num_clients, min_images, size_alpha, rng
    )

    pneumonia_fracs = rng.beta(label_alpha, label_alpha, size=num_clients)

    client_pairs = [[] for _ in range(num_clients)]
    n_ptr = 0
    p_ptr = 0

    order = rng.permutation(num_clients)

    for c in order:
        s = int(target_sizes[c])
        frac = float(pneumonia_fracs[c])

        want_pneumonia = int(round(s * frac))
        want_normal = s - want_pneumonia

        avail_pneumonia = len(pneumonia_pool) - p_ptr
        avail_normal = len(normal_pool) - n_ptr

        take_pneumonia = min(want_pneumonia, avail_pneumonia)
        take_normal = min(want_normal, avail_normal)

        shortfall = s - (take_pneumonia + take_normal)
        if shortfall > 0:
            extra_normal = min(shortfall, avail_normal - take_normal)
            take_normal += extra_normal
            shortfall -= extra_normal

            extra_pneumonia = min(shortfall, avail_pneumonia - take_pneumonia)
            take_pneumonia += extra_pneumonia
            shortfall -= extra_pneumonia
            # Any remaining shortfall here would mean the whole pool is
            # exhausted, which cannot happen since sizes sum to total_pool.

        client_pairs[c].extend(pneumonia_pool[p_ptr:p_ptr + take_pneumonia])
        p_ptr += take_pneumonia

        client_pairs[c].extend(normal_pool[n_ptr:n_ptr + take_normal])
        n_ptr += take_normal

        rng.shuffle(client_pairs[c])

    return client_pairs


def partition_clients_realistic_hospitals(pairs, num_clients, seed, profiles, min_images,
                                           prevalence_noise_std, prevalence_clip):
    """
    Named, interpretable "hospital network" partition: fixed site
    profiles (a big general hospital, a referral center, a community
    clinic, etc.) each with a different total case volume and a
    different pneumonia prevalence, grounded in the ACTUAL measured
    prevalence of the dataset rather than an arbitrary random skew.

    Every client is guaranteed to contain BOTH classes -- prevalence is
    clipped to prevalence_clip (e.g. (0.05, 0.65)) so no site is ever
    100% or 0% pneumonia, which mirrors how real clinics work: even a
    respiratory referral center still sees plenty of normal chest
    X-rays, and even a routine-checkup clinic still catches some
    pneumonia cases.

    Allocation order is largest-pneumonia-appetite-first, so specialty/
    referral sites actually get a concentrated share of positive cases
    rather than whatever happens to be left over once other clients
    have been served.
    """
    if len(profiles) != num_clients:
        raise ValueError(
            f"CLIENT_PROFILES has {len(profiles)} entries but "
            f"NUM_CLIENTS={num_clients}; they must match."
        )

    rng = np.random.RandomState(seed)

    normal_pool = [p for p in pairs if p[1] == 0]
    pneumonia_pool = [p for p in pairs if p[1] == 1]

    rng.shuffle(normal_pool)
    rng.shuffle(pneumonia_pool)

    total_pool = len(pairs)
    global_prevalence = len(pneumonia_pool) / max(1, total_pool)

    target_sizes = compute_profile_client_sizes(total_pool, num_clients, min_images, profiles)

    low, high = prevalence_clip
    pneumonia_fracs = np.zeros(num_clients)

    for i, prof in enumerate(profiles):
        noise = rng.normal(0.0, prevalence_noise_std)
        frac = global_prevalence * prof["prevalence_multiplier"] + noise
        pneumonia_fracs[i] = float(np.clip(frac, low, high))

    desired_pneumonia = np.round(target_sizes * pneumonia_fracs).astype(int)
    # Guarantee at least a couple of positive AND negative cases per
    # client even after rounding, so "both classes present" holds
    # exactly, not just approximately.
    desired_pneumonia = np.clip(desired_pneumonia, 2, target_sizes - 2)

    # Proportionally rescale demand so it never exceeds total supply for
    # either class. Without this, a high-prevalence site (e.g. the
    # referral center) can exhaust the pneumonia pool before a
    # low-prevalence site is served, silently zeroing that site's
    # minority class despite the floor above. Scaling keeps every
    # client's *relative* share of each class intact while guaranteeing
    # total demand <= total supply, so the allocation below can always
    # honor every client's floor.
    avail_pneumonia_total = len(pneumonia_pool)
    total_desired_pneumonia = int(desired_pneumonia.sum())
    if total_desired_pneumonia > avail_pneumonia_total:
        scale = avail_pneumonia_total / total_desired_pneumonia
        scaled = np.floor(desired_pneumonia * scale).astype(int)
        scaled = np.maximum(scaled, np.minimum(2, target_sizes - 2))
        while scaled.sum() > avail_pneumonia_total:
            scaled[int(np.argmax(scaled))] -= 1
        desired_pneumonia = scaled

    desired_normal = target_sizes - desired_pneumonia
    avail_normal_total = len(normal_pool)
    total_desired_normal = int(desired_normal.sum())
    if total_desired_normal > avail_normal_total:
        scale = avail_normal_total / total_desired_normal
        scaled_normal = np.floor(desired_normal * scale).astype(int)
        scaled_normal = np.maximum(scaled_normal, np.minimum(2, target_sizes - 2))
        while scaled_normal.sum() > avail_normal_total:
            scaled_normal[int(np.argmax(scaled_normal))] -= 1
        desired_pneumonia = target_sizes - scaled_normal

    # Serve the sites with the biggest pneumonia appetite first so
    # specialty/referral sites get concentrated positive cases rather
    # than whatever is left over.
    order = np.argsort(-desired_pneumonia)

    client_pairs = [[] for _ in range(num_clients)]
    n_ptr = 0
    p_ptr = 0

    for c in order:
        s = int(target_sizes[c])
        want_pneumonia = int(desired_pneumonia[c])
        want_normal = s - want_pneumonia

        avail_pneumonia = len(pneumonia_pool) - p_ptr
        avail_normal = len(normal_pool) - n_ptr

        take_pneumonia = min(want_pneumonia, avail_pneumonia)
        take_normal = min(want_normal, avail_normal)

        shortfall = s - (take_pneumonia + take_normal)
        if shortfall > 0:
            extra_normal = min(shortfall, avail_normal - take_normal)
            take_normal += extra_normal
            shortfall -= extra_normal

            extra_pneumonia = min(shortfall, avail_pneumonia - take_pneumonia)
            take_pneumonia += extra_pneumonia
            shortfall -= extra_pneumonia

        client_pairs[c].extend(pneumonia_pool[p_ptr:p_ptr + take_pneumonia])
        p_ptr += take_pneumonia

        client_pairs[c].extend(normal_pool[n_ptr:n_ptr + take_normal])
        n_ptr += take_normal

        rng.shuffle(client_pairs[c])

    return client_pairs


def partition_clients(pairs, num_clients, seed, strategy, alpha, label_alpha=None, size_alpha=None,
                       min_images=None, profiles=None, prevalence_noise_std=None, prevalence_clip=None):
    if strategy == "iid":
        return partition_clients_iid(pairs, num_clients, seed)
    elif strategy == "dirichlet":
        return partition_clients_dirichlet(pairs, num_clients, seed, alpha)
    elif strategy == "noniid_unequal":
        return partition_clients_noniid_unequal(
            pairs, num_clients, seed, label_alpha, size_alpha, min_images
        )
    elif strategy == "realistic_hospitals":
        return partition_clients_realistic_hospitals(
            pairs, num_clients, seed, profiles, min_images, prevalence_noise_std, prevalence_clip
        )
    else:
        raise ValueError(f"Unknown PARTITION_STRATEGY: {strategy}")


def select_clients(num_clients, fraction, seed):
    """Randomly sample the subset of clients that participate this round."""
    rng = np.random.RandomState(seed)

    n_selected = max(1, int(round(num_clients * fraction)))
    selected = rng.choice(num_clients, size=n_selected, replace=False)

    return sorted(int(c) for c in selected)


# ============================================================
# 8. METRICS (identical to baseline)
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    return total_loss / max(1, total), y_true, y_pred


def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


@torch.no_grad()
def predict_with_probs(model, loader, device):
    """Return (predicted labels, P(pneumonia)) in loader order."""
    model.eval()

    predictions = []
    probabilities = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        predictions.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        probabilities.extend(probs[:, 1].cpu().numpy().tolist())

    return np.asarray(predictions), probabilities


# ============================================================
# 9. MODEL HELPERS
# ============================================================

def build_model(pretrained):
    """Build a ResNet-18 with a 2-class head."""
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model


def class_weights_for(pairs, device):
    counts = Counter(label for _, label in pairs)

    normal_count = max(1, counts.get(0, 0))
    pneumonia_count = max(1, counts.get(1, 0))
    total = normal_count + pneumonia_count

    return torch.tensor(
        [total / (2 * normal_count), total / (2 * pneumonia_count)],
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# 9b. FEDPROX: LOCAL TRAINING (proximal term) + FEDAVG AGGREGATION
# ============================================================

def client_update(client_model, global_state, dataset, device, local_epochs, lr, class_weights, mu, client_label="client"):
    """
    One client's local training round -- FedProx: local objective is
    the usual class-weighted cross-entropy PLUS a proximal term
    (mu / 2) * ||w - w_global||^2 that penalizes the local weights for
    drifting away from the current global model. Setting mu=0 makes
    this identical to plain FedAvg local training.

    client_model is a reusable ResNet-18 instance (architecture only);
    its weights are overwritten with global_state at the start of every
    call so we avoid rebuilding the network each round.

    Returns (state_dict on CPU, number of local samples).
    """
    client_model.load_state_dict(global_state)
    client_model.to(device)
    client_model.train()

    # Snapshot of the global weights this client started from, used as
    # the anchor point for the proximal term. Only trainable parameters
    # are regularized (not BatchNorm running-stat buffers).
    global_params = [p.detach().clone() for p in client_model.parameters()]

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(client_model.parameters(), lr=lr)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    n_batches = len(loader)

    for epoch_idx in range(local_epochs):
        epoch_label = f"{client_label}" if local_epochs == 1 else f"{client_label} (local ep {epoch_idx + 1}/{local_epochs})"

        batch_progress = tqdm(
            loader,
            desc=epoch_label,
            unit="batch",
            total=n_batches,
            leave=False,
        )

        for images, labels in batch_progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = client_model(images)
            loss = criterion(outputs, labels)

            if mu > 0:
                proximal_term = torch.zeros((), device=device)
                for local_p, global_p in zip(client_model.parameters(), global_params):
                    proximal_term = proximal_term + (local_p - global_p).pow(2).sum()
                loss = loss + (mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()

            batch_progress.set_postfix(loss=f"{loss.item():.4f}")

    state = {k: v.detach().cpu().clone() for k, v in client_model.state_dict().items()}

    return state, len(dataset)


def fedavg_aggregate(client_states, client_sizes):
    """
    Sample-size-weighted average of client state_dicts (McMahan et al.,
    2017, Algorithm 1). Integer buffers (e.g. BatchNorm's
    num_batches_tracked) are averaged as floats and rounded back.
    """
    total = sum(client_sizes)
    weights = [size / total for size in client_sizes]

    avg_state = {}

    for key in client_states[0].keys():
        stacked = torch.stack(
            [client_states[i][key].float() * weights[i] for i in range(len(client_states))],
            dim=0,
        )
        summed = stacked.sum(dim=0)

        original_dtype = client_states[0][key].dtype
        if original_dtype in (torch.int64, torch.int32, torch.int16):
            avg_state[key] = summed.round().to(original_dtype)
        else:
            avg_state[key] = summed.to(original_dtype)

    return avg_state


# ============================================================
# 10. MAIN
# ============================================================

def main():
    set_seed(SEED)

    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA (BALANCED) CLASSIFICATION - RESNET-18 - FEDPROX")
    print("=" * 60)

    print("\nChecking dataset...")

    for folder in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Required folder not found: {folder}")

    print(f"Train folder      : {TRAIN_DIR}")
    print(f"Validation folder : {VAL_DIR}")
    print(f"Test folder       : {TEST_DIR}")

    # --------------------------------------------------------
    # Load labeled images from the class sub-folders of each split
    # --------------------------------------------------------
    train_pairs_all = load_split_pairs(TRAIN_DIR)
    val_pairs = load_split_pairs(VAL_DIR)
    test_pairs = load_split_pairs(TEST_DIR)

    # Optional stratified subsample of train/ (None = use everything).
    train_pairs = stratified_sample(train_pairs_all, MAX_TRAIN_IMAGES, SEED)

    def label_counts(pairs):
        counts = Counter(label for _, label in pairs)
        return {"NORMAL": counts.get(0, 0), "PNEUMONIA": counts.get(1, 0)}

    print()
    print("Training pool (to be split across clients):", len(train_pairs), label_counts(train_pairs))
    print("Validation images (central holdout)       :", len(val_pairs), label_counts(val_pairs))
    print("Test images (final evaluation only)       :", len(test_pairs), label_counts(test_pairs))

    # Per-client minimum size: auto-scaled to the dataset unless set explicitly.
    if MIN_IMAGES_PER_CLIENT is None:
        min_images_per_client = int(0.5 * len(train_pairs) / NUM_CLIENTS)
    else:
        min_images_per_client = MIN_IMAGES_PER_CLIENT

    print(f"Min images per client: {min_images_per_client}")

    # --------------------------------------------------------
    # Partition the training pool across simulated clients
    # --------------------------------------------------------
    if PARTITION_STRATEGY == "realistic_hospitals" and len(CLIENT_PROFILES) != NUM_CLIENTS:
        raise ValueError(
            f"CLIENT_PROFILES has {len(CLIENT_PROFILES)} entries but "
            f"NUM_CLIENTS={NUM_CLIENTS}; they must match."
        )

    if PARTITION_STRATEGY in ("noniid_unequal", "realistic_hospitals") and \
            len(train_pairs) < NUM_CLIENTS * min_images_per_client:
        raise RuntimeError(
            f"Training pool has {len(train_pairs)} images after the "
            f"validation split, but {NUM_CLIENTS} clients x "
            f"{min_images_per_client} MIN_IMAGES_PER_CLIENT requires at "
            f"least {NUM_CLIENTS * min_images_per_client}. Set "
            f"MAX_TRAIN_IMAGES = None (or a larger number) to use more of "
            f"the dataset, or lower MIN_IMAGES_PER_CLIENT / NUM_CLIENTS."
        )

    client_pairs = partition_clients(
        train_pairs,
        NUM_CLIENTS,
        SEED,
        PARTITION_STRATEGY,
        NON_IID_ALPHA,
        label_alpha=LABEL_ALPHA,
        size_alpha=SIZE_ALPHA,
        min_images=min_images_per_client,
        profiles=CLIENT_PROFILES,
        prevalence_noise_std=PREVALENCE_NOISE_STD,
        prevalence_clip=PREVALENCE_CLIP,
    )

    client_names = (
        [prof["name"] for prof in CLIENT_PROFILES]
        if PARTITION_STRATEGY == "realistic_hospitals"
        else [f"Client_{i}" for i in range(NUM_CLIENTS)]
    )

    print(f"\nPartition strategy: {PARTITION_STRATEGY}")
    if PARTITION_STRATEGY == "dirichlet":
        print(f"Non-IID alpha: {NON_IID_ALPHA}")
    if PARTITION_STRATEGY == "noniid_unequal":
        print(f"Size alpha: {SIZE_ALPHA}  |  Label alpha: {LABEL_ALPHA}  |  Min images/client: {min_images_per_client}")
    print(f"FedProx mu: {MU}")
    if PARTITION_STRATEGY == "realistic_hospitals":
        print(f"Min images/client: {min_images_per_client}  |  Prevalence clip: {PREVALENCE_CLIP} (both classes guaranteed)")

    client_summary_rows = []

    print("\nClient data summary:")
    for cid, pairs in enumerate(client_pairs):
        counts = Counter(label for _, label in pairs)
        n_normal = counts.get(0, 0)
        n_pneumonia = counts.get(1, 0)
        pneumonia_fraction = n_pneumonia / max(1, len(pairs))

        print(
            f"  Client {cid} ({client_names[cid]}): {len(pairs)} images "
            f"(NORMAL={n_normal}, PNEUMONIA={n_pneumonia}, "
            f"pneumonia_fraction={pneumonia_fraction:.2f})"
        )

        client_summary_rows.append(
            {
                "client_id": cid,
                "client_name": client_names[cid],
                "num_images": len(pairs),
                "non_pneumonia": n_normal,
                "pneumonia": n_pneumonia,
                "pneumonia_fraction": round(pneumonia_fraction, 4),
            }
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Datasets / loaders
    # --------------------------------------------------------
    client_datasets = [
        PneumoniaDataset(pairs, train_transform) for pairs in client_pairs
    ]

    val_dataset = PneumoniaDataset(val_pairs, eval_transform)
    test_dataset = PneumoniaDataset(test_pairs, eval_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Each client's own class-weighted loss (a client only knows its
    # own local label distribution in real FL).
    client_class_weights = [
        class_weights_for(pairs, device) for pairs in client_pairs
    ]

    # --------------------------------------------------------
    # Global model (pretrained ImageNet init, shared starting point)
    # --------------------------------------------------------
    print("\nCreating global ResNet-18...")

    try:
        global_model = build_model(pretrained=True)
    except Exception as exc:
        print("Could not load pretrained ResNet-18 weights.")
        print("If this is the first run, check your internet connection.")
        raise exc

    global_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}

    # Reusable model instance for both client-side local training and
    # server-side evaluation, to avoid rebuilding the architecture
    # every round.
    client_model = build_model(pretrained=False).to(device)
    eval_model = build_model(pretrained=False).to(device)
    eval_criterion = nn.CrossEntropyLoss()  # unweighted; monitoring only

    # --------------------------------------------------------
    # Federated training loop
    # --------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_balanced_accuracy = -1.0
    best_round = 0
    history = []

    print("\nStarting federated training (FedProx local + FedAvg aggregation)...\n")

    for rnd in range(NUM_ROUNDS):
        participating = select_clients(NUM_CLIENTS, CLIENT_FRACTION, seed=SEED + rnd)

        client_states = []
        client_sizes = []

        progress = tqdm(
            participating,
            desc=f"Round {rnd + 1}/{NUM_ROUNDS} - clients",
            unit="client",
            leave=True,
        )

        for cid in progress:
            state, n_samples = client_update(
                client_model,
                global_state,
                client_datasets[cid],
                device,
                LOCAL_EPOCHS,
                LEARNING_RATE,
                client_class_weights[cid],
                MU,
                client_label=client_names[cid],
            )

            client_states.append(state)
            client_sizes.append(n_samples)

            progress.set_postfix(client=cid, n=n_samples)

        # Server-side aggregation
        global_state = fedavg_aggregate(client_states, client_sizes)

        # Evaluate the new global model on the central validation set
        eval_model.load_state_dict(global_state)

        val_loss, y_true, y_pred = evaluate(eval_model, val_loader, eval_criterion, device)
        m = metrics(y_true, y_pred)

        history.append(
            {
                "round": rnd + 1,
                "participating_clients": len(participating),
                "val_loss": val_loss,
                "val_accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "f1": m["f1"],
            }
        )

        print(
            f"Round [{rnd + 1}/{NUM_ROUNDS}] "
            f"clients={len(participating)} "
            f"val_loss {val_loss:.4f} "
            f"val_acc {100 * m['accuracy']:.2f}% "
            f"sens {100 * m['sensitivity']:.2f}% "
            f"spec {100 * m['specificity']:.2f}% "
            f"bal_acc {100 * m['balanced_accuracy']:.2f}%"
        )

        if m["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = m["balanced_accuracy"]
            best_round = rnd + 1

            torch.save(
                {
                    "model_state_dict": global_state,
                    "best_val_balanced_accuracy": best_balanced_accuracy,
                    "seed": SEED,
                    "num_clients": NUM_CLIENTS,
                    "partition_strategy": PARTITION_STRATEGY,
                    "class_names": ["NORMAL", "PNEUMONIA"],
                },
                MODEL_PATH,
            )

            print("  -> best global model saved")

    # --------------------------------------------------------
    # Final validation evaluation (best global round)
    # --------------------------------------------------------
    print("\nLoading best global model...")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    eval_model.load_state_dict(checkpoint["model_state_dict"])

    _, y_true, y_pred = evaluate(eval_model, val_loader, eval_criterion, device)
    m = metrics(y_true, y_pred)

    print("\n" + "=" * 52)
    print("FINAL FEDPROX VALIDATION RESULTS")
    print("=" * 52)

    print(f"Best round: {best_round}")
    print(f"Best val balanced accuracy: {100 * best_balanced_accuracy:.2f}%")

    print()
    print("Confusion matrix (rows = true, cols = predicted)")
    print(f"{'':>18}{'NORMAL':>16}{'PNEUMONIA':>14}")
    print(f"{'NORMAL':>18}{m['tn']:>16}{m['fp']:>14}")
    print(f"{'PNEUMONIA':>18}{m['fn']:>16}{m['tp']:>14}")

    print()
    print(f"Accuracy          : {100 * m['accuracy']:.2f}%")
    print(f"Balanced accuracy : {100 * m['balanced_accuracy']:.2f}%")
    print(f"Sensitivity       : {100 * m['sensitivity']:.2f}%")
    print(f"Specificity       : {100 * m['specificity']:.2f}%")
    print(f"Precision         : {100 * m['precision']:.2f}%")
    print(f"F1 score          : {100 * m['f1']:.2f}%")
    print("=" * 52)

    # --------------------------------------------------------
    # Final TEST evaluation (best global model, evaluated once)
    # --------------------------------------------------------
    print("\nEvaluating best global model on the held-out test set...")

    y_pred_test, test_probs = predict_with_probs(eval_model, test_loader, device)
    y_true_test = np.asarray([label for _, label in test_pairs])
    tm = metrics(y_true_test, y_pred_test)

    print("\n" + "=" * 52)
    print("FINAL FEDPROX TEST RESULTS")
    print("=" * 52)

    print("Confusion matrix (rows = true, cols = predicted)")
    print(f"{'':>18}{'NORMAL':>16}{'PNEUMONIA':>14}")
    print(f"{'NORMAL':>18}{tm['tn']:>16}{tm['fp']:>14}")
    print(f"{'PNEUMONIA':>18}{tm['fn']:>16}{tm['tp']:>14}")

    print()
    print(f"Accuracy          : {100 * tm['accuracy']:.2f}%")
    print(f"Balanced accuracy : {100 * tm['balanced_accuracy']:.2f}%")
    print(f"Sensitivity       : {100 * tm['sensitivity']:.2f}%")
    print(f"Specificity       : {100 * tm['specificity']:.2f}%")
    print(f"Precision         : {100 * tm['precision']:.2f}%")
    print(f"F1 score          : {100 * tm['f1']:.2f}%")
    print("=" * 52)

    pd.DataFrame(
        {
            "image_path": [path for path, _ in test_pairs],
            "TrueTarget": y_true_test,
            "PredictedTarget": y_pred_test,
            "PneumoniaProbability": test_probs,
        }
    ).to_csv(TEST_PREDICTIONS_PATH, index=False)

    print(f"Test predictions saved to: {TEST_PREDICTIONS_PATH}")

    # --------------------------------------------------------
    # Save history, client partition summary, and results
    # --------------------------------------------------------
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)

    pd.DataFrame(client_summary_rows).to_csv(CLIENT_SUMMARY_PATH, index=False)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("CHEST X-RAY PNEUMONIA (BALANCED) CLASSIFICATION - FEDPROX (ResNet-18)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset path: {DATASET_PATH}\n")
        f.write(f"Max training images (pool): {MAX_TRAIN_IMAGES}\n")
        f.write(f"Labeled training images available (train/): {len(train_pairs_all)}\n")
        f.write(f"Training pool (across clients): {len(train_pairs)}\n")
        f.write(f"Validation images (central holdout, val/): {len(val_pairs)}\n")
        f.write(f"Test images (final evaluation, test/): {len(test_pairs)}\n")
        f.write(f"Number of clients: {NUM_CLIENTS}\n")
        f.write(f"Partition strategy: {PARTITION_STRATEGY}\n")
        if PARTITION_STRATEGY == "dirichlet":
            f.write(f"Non-IID alpha: {NON_IID_ALPHA}\n")
        if PARTITION_STRATEGY == "noniid_unequal":
            f.write(f"Size alpha: {SIZE_ALPHA}\n")
            f.write(f"Label alpha: {LABEL_ALPHA}\n")
            f.write(f"Min images per client: {min_images_per_client}\n")
        if PARTITION_STRATEGY == "realistic_hospitals":
            f.write(f"Min images per client: {min_images_per_client}\n")
            f.write(f"Prevalence clip: {PREVALENCE_CLIP} (both classes guaranteed for every client)\n")
            f.write("Client profiles:\n")
            for prof in CLIENT_PROFILES:
                f.write(
                    f"  - {prof['name']}: size_weight={prof['size_weight']}, "
                    f"prevalence_multiplier={prof['prevalence_multiplier']}\n"
                )
        f.write(f"Rounds: {NUM_ROUNDS}\n")
        f.write(f"Local epochs per round: {LOCAL_EPOCHS}\n")
        f.write(f"FedProx mu: {MU}\n")
        f.write(f"Client fraction per round: {CLIENT_FRACTION}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Device: {device}\n\n")

        f.write(f"Best round: {best_round}\n")
        f.write(f"Best validation balanced accuracy: {100 * best_balanced_accuracy:.2f}%\n\n")

        f.write("CONFUSION MATRIX (rows=true, cols=predicted)\n")
        f.write("                    NORMAL    PNEUMONIA\n")
        f.write(f"NORMAL       {m['tn']:>10}    {m['fp']:>10}\n")
        f.write(f"PNEUMONIA          {m['fn']:>10}    {m['tp']:>10}\n\n")

        f.write("FINAL VALIDATION METRICS\n")
        f.write(f"Accuracy          : {100 * m['accuracy']:.2f}%\n")
        f.write(f"Balanced accuracy : {100 * m['balanced_accuracy']:.2f}%\n")
        f.write(f"Sensitivity       : {100 * m['sensitivity']:.2f}%\n")
        f.write(f"Specificity       : {100 * m['specificity']:.2f}%\n")
        f.write(f"Precision         : {100 * m['precision']:.2f}%\n")
        f.write(f"F1 score          : {100 * m['f1']:.2f}%\n\n")

        f.write("FINAL TEST METRICS (best global model, test/ folder)\n")
        f.write("CONFUSION MATRIX (rows=true, cols=predicted)\n")
        f.write("                    NORMAL    PNEUMONIA\n")
        f.write(f"NORMAL       {tm['tn']:>10}    {tm['fp']:>10}\n")
        f.write(f"PNEUMONIA          {tm['fn']:>10}    {tm['tp']:>10}\n")
        f.write(f"Accuracy          : {100 * tm['accuracy']:.2f}%\n")
        f.write(f"Balanced accuracy : {100 * tm['balanced_accuracy']:.2f}%\n")
        f.write(f"Sensitivity       : {100 * tm['sensitivity']:.2f}%\n")
        f.write(f"Specificity       : {100 * tm['specificity']:.2f}%\n")
        f.write(f"Precision         : {100 * tm['precision']:.2f}%\n")
        f.write(f"F1 score          : {100 * tm['f1']:.2f}%\n\n")

        f.write("OUTPUT FILES\n")
        f.write(f"Best model: {MODEL_PATH}\n")
        f.write(f"Test predictions: {TEST_PREDICTIONS_PATH}\n")
        f.write(f"Round-by-round history: {HISTORY_PATH}\n")
        f.write(f"Client partition summary: {CLIENT_SUMMARY_PATH}\n")
        f.write(f"Results: {RESULTS_PATH}\n")

    print(f"Round-by-round history saved to: {HISTORY_PATH}")
    print(f"Client partition summary saved to: {CLIENT_SUMMARY_PATH}")
    print(f"Final results saved to: {RESULTS_PATH}")

    print(f"\nBest global model saved to: {MODEL_PATH}")
    print("\nFederated training (FedProx) completed successfully.")


if __name__ == "__main__":
    main()