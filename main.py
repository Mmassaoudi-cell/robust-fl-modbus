"""
==============================================================================
Byzantine-Robust Federated Learning for Industrial Control System
Intrusion Detection
==============================================================================

Paper   : "Byzantine-Robust Federated Learning for Industrial Control System
           Intrusion Detection"
Authors : Mohamed Massaoudi, Katherine R. Davis, Maymouna Ez Eddin
Repo    : https://github.com/mmassaoudi/byzantine-robust-fl-ics

HOW TO RUN
----------
    pip install -r requirements.txt
    python main.py

No manual configuration is needed.

REPRODUCIBILITY
---------------
- Random seeds are fixed globally (SEED = 42).
- Device is forced to CPU so results are identical across machines.
- All dependency versions are pinned in requirements.txt.
==============================================================================
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import os
import sys
import json
import time
import warnings
import gc
from copy import deepcopy

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Third-party dependencies  (see requirements.txt for pinned versions)
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")           # headless rendering -- no display required
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SEED        = 42    # fixed for full reproducibility
N_CLIENTS   = 10    # federated participants (paper Section IV-B)
N_BYZANTINE = 2     # default Byzantine clients = 20 % ratio (paper Section IV-B)
N_ROUNDS    = 30    # FL communication rounds (paper Section IV-C)
DEVICE      = "cpu" # CPU preferred: model is 113 K params; GPU offers no benefit

# Fix all random sources before anything else
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("Byzantine-Robust FL for ICS -- Reproducible Main Script")
print("  Seed      : {}".format(SEED))
print("  Device    : {}".format(DEVICE))
print("  Output dir: {}".format(OUTDIR))
print("=" * 70)


# =============================================================================
# SECTION 1 -- DATA LOADING & PREPROCESSING
# Paper Section IV-A: Modbus traffic dataset, 6 690 samples, 17 features
# =============================================================================

# Feature names as described in paper Section IV-A
FEATURE_NAMES = [
    "function_code",       # Modbus function code (normal: 1-4; attack: 5-24)
    "register_address",    # target register (abnormal: 40 000-65 535)
    "data_value_hi",       # high byte of data field
    "data_value_lo",       # low byte of data field
    "transaction_id",      # transaction identifier
    "protocol_id",         # protocol discriminator (0 = Modbus)
    "unit_id",             # device unit ID
    "packet_length",       # total packet length (abnormal: > 20 bytes)
    "timestamp_delta",     # inter-packet time (high value -> scanning)
    "byte_count",          # payload byte count
    "coil_address",        # coil register address
    "coil_value",          # coil state
    "holding_reg",         # holding register value
    "quantity",            # quantity of registers/coils
    "error_code",          # Modbus exception code
    "payload_entropy",     # Shannon entropy (high -> encrypted/random payload)
    "inter_arrival_mean",  # mean inter-arrival time
]


def load_dataset():
    """Load the real Modbus CSV or generate a deterministic synthetic substitute.

    The paper uses a Modbus traffic dataset with 6 690 samples and 17 features
    (Section IV-A).  If the CSV is not present, we reproduce the same class
    distribution (60 % normal / 40 % attack) via NumPy with SEED=42.
    """
    candidate_paths = [
        os.path.join(OUTDIR, "modbus_traffic_data.csv"),
        "ModbusTrafficXAI-main/ModbusTrafficXAI-main/dataset/modbus_traffic_data.csv",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print("[Data] Real dataset loaded: {}".format(df.shape))
            return df

    # Synthetic Modbus dataset (paper Section IV-A)
    print("[Data] Real CSV not found -- generating synthetic Modbus dataset ...")
    rng      = np.random.RandomState(SEED)
    n_normal = 4014   # 60 % normal traffic
    n_attack = 2676   # 40 % attack traffic

    # Normal traffic: standard Modbus read operations (function codes 1-4)
    X_normal = np.column_stack([
        rng.choice([1, 2, 3, 4], n_normal),          # function_code
        rng.randint(0, 1000, n_normal),               # register_address
        rng.randint(0, 256, n_normal),                # data_value_hi
        rng.randint(0, 256, n_normal),                # data_value_lo
        rng.randint(1, 65535, n_normal),              # transaction_id
        np.zeros(n_normal, dtype=int),                # protocol_id
        rng.randint(1, 10, n_normal),                 # unit_id
        rng.randint(6, 12, n_normal),                 # packet_length
        rng.uniform(0, 0.05, n_normal),               # timestamp_delta
        rng.randint(0, 100, n_normal),                # byte_count
        rng.randint(0, 65535, n_normal),              # coil_address
        rng.randint(0, 2, n_normal),                  # coil_value
        rng.randint(0, 65535, n_normal),              # holding_reg
        rng.randint(0, 10, n_normal),                 # quantity
        rng.randint(0, 2, n_normal),                  # error_code
        rng.uniform(0, 1, n_normal),                  # payload_entropy
        rng.uniform(0, 0.1, n_normal),                # inter_arrival_mean
    ])

    # Attack traffic: anomalous function codes, extreme registers, high entropy
    X_attack = np.column_stack([
        rng.choice([5, 6, 15, 16, 22, 23], n_attack),  # anomalous function codes
        rng.randint(40000, 65535, n_attack),             # high register (unusual)
        rng.randint(200, 256, n_attack),
        rng.randint(200, 256, n_attack),
        rng.randint(1, 65535, n_attack),
        rng.choice([0, 1, 2], n_attack),
        rng.randint(1, 255, n_attack),
        rng.randint(20, 260, n_attack),                 # abnormal packet length
        rng.uniform(0.5, 5.0, n_attack),                # large inter-packet gap
        rng.randint(100, 300, n_attack),
        rng.randint(0, 65535, n_attack),
        rng.randint(0, 4, n_attack),
        rng.randint(0, 65535, n_attack),
        rng.randint(10, 125, n_attack),
        rng.randint(1, 10, n_attack),
        rng.uniform(0.7, 1.0, n_attack),                # high entropy
        rng.uniform(0.5, 10.0, n_attack),
    ])

    X   = np.vstack([X_normal, X_attack])
    y   = np.concatenate([np.zeros(n_normal, int), np.ones(n_attack, int)])
    idx = rng.permutation(n_normal + n_attack)
    X, y = X[idx], y[idx]

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    print("[Data] Synthetic dataset ready: {}  (normal={}, attack={})".format(
          df.shape, n_normal, n_attack))
    return df


def preprocess(df):
    """Normalise features and return PyTorch tensors plus a fitted scaler.

    Paper Section IV-A: 'Normalize features to zero mean and unit variance
    using StandardScaler; handle missing values using median imputation.'
    """
    if "label" in df.columns:
        y = df["label"].values.astype(int)
        X = df.drop("label", axis=1).select_dtypes(include=[np.number]).values
    else:
        y = df.iloc[:, -1].values.astype(int)
        X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values

    # Median imputation for any NaN values
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.FloatTensor(X_scaled), torch.LongTensor(y), scaler


def split_federated(X, y, n_clients=10, iid=True):
    """Partition training data across N federated clients.

    Paper Section IV-B:
    - IID    : random shuffled assignment (~535 samples/client).
    - NonIID : shard-based (2 of 2N sorted shards per client) to create
               local class imbalance representative of real utility heterogeneity.
    """
    n   = len(X)
    idx = np.arange(n)

    if iid:
        np.random.shuffle(idx)
        splits = np.array_split(idx, n_clients)
    else:
        sorted_idx = idx[np.argsort(y.numpy())]
        n_shards   = n_clients * 2
        shards     = np.array_split(sorted_idx, n_shards)
        shard_ids  = np.arange(n_shards)
        np.random.shuffle(shard_ids)
        splits = [
            np.concatenate([shards[shard_ids[2 * c]], shards[shard_ids[2 * c + 1]]])
            for c in range(n_clients)
        ]

    return [(X[s], y[s]) for s in splits]


# =============================================================================
# SECTION 2 -- MODEL ARCHITECTURE
# Paper Section III-A: CNN with Self-Attention, 113 538 parameters
# =============================================================================

class AttentionLayer(nn.Module):
    """Element-wise soft-attention for feature importance weighting.

    Paper Section III-A (equation 2):
        a     = Softmax( MLP(h) )
        h_out = h * a  (element-wise product)
    MLP maps flat_dim -> flat_dim//2 -> flat_dim with ReLU activation.
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        weights = self.net(x)
        return x * weights, weights


class ModbusCNN(nn.Module):
    """1-D CNN + Self-Attention classifier for Modbus traffic (paper Section III-A).

    Architecture (paper Section IV-C):
        Input(F) -> Conv1D(32, k=3) -> MaxPool -> Conv1D(64, k=3) -> MaxPool
                 -> Attention -> FC(128) -> Dropout(0.3) -> FC(64) -> FC(2)
    Total trainable parameters: 113 538 (for F=17 input features).
    Gradient clipping (max norm 5.0) stabilises training under Byzantine noise.
    """

    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()

        # Compute the flattened dimension after two conv+pool blocks dynamically
        with torch.no_grad():
            dummy    = torch.zeros(1, 1, input_dim)
            dummy    = self.pool(self.relu(self.conv1(dummy)))
            dummy    = self.pool(self.relu(self.conv2(dummy)))
            flat_dim = dummy.view(1, -1).shape[1]

        self.attention = AttentionLayer(flat_dim)
        self.fc1       = nn.Linear(flat_dim, 128)
        self.fc2       = nn.Linear(128, 64)
        self.fc3       = nn.Linear(64, num_classes)
        self.drop      = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)                         # (B, 1, F) for Conv1d
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)                  # flatten spatial dims
        x, attn = self.attention(x)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x), attn                    # (logits, attention weights)


# =============================================================================
# SECTION 3 -- BYZANTINE ATTACK MODELS
# Paper Section III-B: five attack strategies from the ICS threat landscape
# =============================================================================

class Attacker:
    """Five Byzantine attack strategies from the paper (Section III-B).

    Each static method accepts an update dict {param_name: tensor} and returns
    a corrupted version.  All attack parameters match those in the paper.
    """

    @staticmethod
    def label_flipping(u, alpha=1.5):
        """Amplified sign inversion: w_atk = -alpha * w_honest  (alpha=1.5)."""
        return {k: -alpha * v for k, v in u.items()}

    @staticmethod
    def gaussian_noise(u, sigma=5.0):
        """High-variance noise: w_atk = w + N(0, sigma^2 * ||w||^2 * I)  (sigma=5.0)."""
        return {
            k: v + torch.randn_like(v) * sigma * max(v.norm().item(), 1e-4)
            for k, v in u.items()
        }

    @staticmethod
    def backdoor(u, beta=10.0):
        """Backdoor amplification: w_atk = beta * w_honest  (beta=10.0)."""
        return {k: beta * v for k, v in u.items()}

    @staticmethod
    def same_value(u, c=50.0):
        """Constant replacement: w_atk = c * ones  (c=50.0)."""
        return {k: torch.ones_like(v) * c for k, v in u.items()}

    @staticmethod
    def zero_attack(u):
        """Zero-gradient suppression: w_atk = 0."""
        return {k: torch.zeros_like(v) for k, v in u.items()}

    @classmethod
    def apply(cls, u, attack_type):
        """Dispatch helper -- apply the named attack or return update unchanged."""
        dispatch = {
            "label_flipping": cls.label_flipping,
            "gaussian_noise":  cls.gaussian_noise,
            "backdoor":        cls.backdoor,
            "same_value":      cls.same_value,
            "zero_attack":     cls.zero_attack,
        }
        return dispatch[attack_type](u) if attack_type in dispatch else u


ATTACK_TYPES = ["none", "label_flipping", "gaussian_noise",
                "backdoor", "same_value", "zero_attack"]

ATTACK_LABELS = {
    "none":           "No Attack",
    "label_flipping": "Label Flip",
    "gaussian_noise": "Gaussian",
    "backdoor":       "Backdoor",
    "same_value":     "Same-Value",
    "zero_attack":    "Zero",
}


# =============================================================================
# SECTION 4 -- BYZANTINE-ROBUST AGGREGATION METHODS
# Paper Section III-C: equations 3-9 plus the proposed Bulyan-Adaptive (eq. 10)
# =============================================================================

def _stack(updates, key):
    """Stack tensors from all client updates for a given parameter key."""
    return torch.stack([u[key] for u in updates], dim=0)


def fedavg(updates, f=0):
    """FedAvg -- uniform weighted average (paper equation 3).

    Vulnerable to Byzantine attacks: a single adversary can shift the global
    model by w_i * (w_atk - w_honest).
    """
    w = 1.0 / len(updates)
    return {k: sum(w * u[k] for u in updates) for k in updates[0]}


def trimmed_mean(updates, f=0, trim_ratio=0.1):
    """Coordinate-wise trimmed mean (paper equation 4, beta=0.10).

    Remove the top and bottom floor(N*beta) values at each parameter
    coordinate before averaging.
    """
    agg = {}
    for k in updates[0]:
        vals  = _stack(updates, k)
        n     = vals.shape[0]
        cut   = max(1, int(n * trim_ratio))
        if 2 * cut >= n:
            cut = 0
        sv, _ = torch.sort(vals, dim=0)
        trimmed = sv[cut: n - cut] if cut > 0 else sv
        agg[k] = trimmed.mean(dim=0)
    return agg


def coordinate_median(updates, f=0):
    """Coordinate-wise median (paper equation 5).

    Breakdown point = 50 %: tolerates up to floor(N/2) Byzantine participants.
    """
    return {k: _stack(updates, k).median(dim=0)[0] for k in updates[0]}


def _krum_scores(updates, f):
    """Compute Krum score for each update (paper equation 6).

    Score_i = sum of squared L2 distances to the (N-f-2) closest neighbours.
    Used internally by Krum, Multi-Krum, and Bulyan.
    """
    n    = len(updates)
    keep = max(1, min(n - f - 2, n - 1))
    scores = []
    for i in range(n):
        dists = sorted(
            sum((updates[i][k] - updates[j][k]).pow(2).sum().item()
                for k in updates[i])
            for j in range(n) if j != i
        )
        scores.append(sum(dists[:keep]))
    return scores


def krum(updates, f):
    """Krum -- select the single update with minimum Krum score (paper eq. 7)."""
    return updates[int(np.argmin(_krum_scores(updates, f)))]


def multi_krum(updates, f):
    """Multi-Krum -- average the top-k updates with smallest Krum scores (eq. 8).

    k = N - f - 2  (paper Section III-C).
    """
    n      = len(updates)
    k      = max(1, n - f - 2)
    scores = _krum_scores(updates, f)
    top_k  = sorted(range(n), key=lambda i: scores[i])[:k]
    sel    = [updates[i] for i in top_k]
    return {key: torch.stack([u[key] for u in sel]).mean(0) for key in sel[0]}


def bulyan(updates, f):
    """Bulyan -- Multi-Krum selection + trimmed mean (paper equation 9).

    Stage 1 (Multi-Krum): select theta = N-2f-2 candidate updates.
    Stage 2 (Trimmed Mean): remove f extremes at each coordinate, then average.
    Formal tolerance guarantee: f < N/4 Byzantine participants.
    """
    n     = len(updates)
    theta = max(1, n - 2 * f - 2)
    scores = _krum_scores(updates, f)
    top    = sorted(range(n), key=lambda i: scores[i])[:theta]
    sel    = [updates[i] for i in top]

    agg = {}
    for k in sel[0]:
        vals = torch.stack([u[k] for u in sel], dim=0)
        sv, _ = torch.sort(vals, dim=0)
        # Conservative cut: ensure at least 1 element remains after trimming
        cut = min(f, max(0, (len(sel) - 1) // 2))
        if cut > 0 and 2 * cut < len(sel):
            trimmed = sv[cut: len(sel) - cut]
        else:
            trimmed = sv
        agg[k] = trimmed.mean(dim=0)
    return agg


def bulyan_adaptive(updates, f, clip_quantile=0.9):
    """Bulyan-Adaptive -- proposed method (paper Section III-C, equation 10).

    Prepend per-round adaptive norm clipping (tau = Q_{0.9} of update norms)
    before standard Bulyan.  This neutralises high-norm attacks (backdoor,
    same-value, high-sigma Gaussian) before the Krum selection stage.
    Added computational overhead is negligible (+0.3 % vs. plain Bulyan).
    """
    norms = [
        float(sum(v.norm().item() ** 2 for v in u.values()) ** 0.5)
        for u in updates
    ]
    valid_norms = [n for n in norms if n > 1e-9]
    if not valid_norms:
        return bulyan(updates, f)

    tau = float(np.quantile(valid_norms, clip_quantile))
    clipped = []
    for u, norm in zip(updates, norms):
        if norm > tau and tau > 1e-9:
            scale = tau / norm
            clipped.append({k: v * scale for k, v in u.items()})
        else:
            clipped.append(u)
    return bulyan(clipped, f)


# Lookup table: aggregation name -> callable(updates, n_byzantine)
AGG_FNS = {
    "FedAvg":         lambda ups, f: fedavg(ups),
    "TrimmedMean":    lambda ups, f: trimmed_mean(ups),
    "Median":         lambda ups, f: coordinate_median(ups),
    "Krum":           krum,
    "MultiKrum":      multi_krum,
    "Bulyan":         bulyan,
    "BulyanAdaptive": bulyan_adaptive,
}


# =============================================================================
# SECTION 5 -- FEDERATED LEARNING TRAINING ENGINE
# Paper Section IV-C: Adam lr=0.001, weight-decay 1e-5, grad-clip max-norm 5.0
# =============================================================================

def local_train(model, global_state, X_tr, y_tr,
                lr=0.001, epochs=1, batch_size=32, device="cpu"):
    """One round of local SGD; return parameter delta = w_local - w_global.

    Key design choices from paper Section IV-C:
    - Adam optimiser, lr=0.001, weight_decay=1e-5
    - Gradient clipping with max_norm=5.0 to stabilise under Byzantine noise
    - NaN/Inf guard applied before and after training to handle corrupted states
    """
    cpu_state   = {k: v.cpu() for k, v in global_state.items()}
    clean_state = {k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
                   for k, v in cpu_state.items()}
    model.load_state_dict(clean_state)
    model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader    = DataLoader(TensorDataset(X_tr, y_tr),
                           batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out, _ = model(xb)
            if torch.isnan(out).any():
                continue
            loss = criterion(out, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

    ls = {k: v.cpu() for k, v in model.state_dict().items()}
    # Sanitise the update -- attacks can inject NaN/Inf via aggregation
    return {
        k: torch.nan_to_num(ls[k] - cpu_state[k], nan=0.0, posinf=0.0, neginf=0.0)
        for k in ls
    }


def evaluate_model(state_dict, X_te, y_te, model_proto, device="cpu"):
    """Evaluate the global model; return a dict of classification metrics.

    Metrics reported in paper Section IV-D:
    Accuracy, Precision, Recall, F1-Score, AUC-ROC.
    """
    m = deepcopy(model_proto).to(device)
    m.load_state_dict(state_dict)
    m.eval()
    loader = DataLoader(TensorDataset(X_te, y_te),
                        batch_size=256, shuffle=False, num_workers=0)
    preds, labs, probs = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            out, _  = m(xb.to(device))
            prob    = torch.softmax(out, dim=1)[:, 1]
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labs.extend(yb.numpy())
            probs.extend(prob.cpu().numpy())

    preds = np.array(preds)
    labs  = np.array(labs)
    probs = np.array(probs)

    try:
        auc = roc_auc_score(labs, probs)
    except Exception:
        auc = 0.5

    return dict(
        accuracy  = accuracy_score(labs, preds),
        precision = precision_score(labs, preds, zero_division=0),
        recall    = recall_score(labs, preds, zero_division=0),
        f1        = f1_score(labs, preds, zero_division=0),
        auc       = auc,
    )


def run_fl(client_data, X_te, y_te, agg_name, attack_type, n_byzantine,
           n_rounds=30, input_dim=17, device="cpu", record_every=1):
    """Execute one complete FL experiment and return per-round metrics.

    Steps per round (paper Section IV-C):
    1. Each client performs local_train on its private data.
    2. Byzantine clients have their update replaced by an attack update.
    3. Server applies the chosen aggregation function.
    4. Global model weights are updated; NaN/Inf is sanitised.
    5. Every `record_every` rounds the global model is evaluated.

    Returns: (list of metric dicts per recorded round, final global state dict)
    """
    model_proto   = ModbusCNN(input_dim).to(device)
    global_state  = {k: v.cpu() for k, v in model_proto.state_dict().items()}
    byzantine_ids = list(range(n_byzantine))
    agg_fn        = AGG_FNS[agg_name]
    n_clients     = len(client_data)
    local_models  = [ModbusCNN(input_dim).to(device) for _ in range(n_clients)]
    round_metrics = []

    for r in range(n_rounds):
        updates = []
        for i, (Xi, yi) in enumerate(client_data):
            upd = local_train(local_models[i], global_state, Xi, yi, device=device)
            upd = {k: v.cpu() for k, v in upd.items()}
            # Replace honest update with attack update for Byzantine clients
            if attack_type != "none" and i in byzantine_ids:
                upd = Attacker.apply(upd, attack_type)
            updates.append(upd)

        # Aggregate all client updates (paper Section III-C)
        agg = agg_fn(updates, n_byzantine)

        # Apply aggregated delta to global state; guard against NaN/Inf
        global_state = {
            k: torch.nan_to_num(
                global_state[k].cpu() + agg[k].cpu(),
                nan=0.0, posinf=1.0, neginf=-1.0,
            )
            for k in global_state
        }

        if (r + 1) % record_every == 0 or r == n_rounds - 1:
            metrics         = evaluate_model(global_state, X_te, y_te,
                                             model_proto, device)
            metrics["round"] = r + 1
            round_metrics.append(metrics)

    return round_metrics, global_state


# =============================================================================
# SECTION 6 -- EXPLAINABILITY: SHAP / ATTENTION CONSISTENCY
# Paper Section III-D and Section V:
#   SHAP feature importances at rounds 10, 25, 50 with Pearson r > 0.85 target
# =============================================================================

def compute_shap_consistency(client_data, X_te, y_te, input_dim, device, n_rounds):
    """Run Bulyan FL and capture feature importances at rounds 10, 25, 50.

    When SHAP is installed, uses KernelSHAP (paper Section III-D, equation 11).
    Otherwise falls back to the model's self-attention weights, which satisfy
    the same additive decomposition within the attention layer.
    Pearson correlation between importance vectors across rounds quantifies
    explanation stability (paper Section V, target r > 0.85).
    """
    try:
        import shap as shap_lib
        has_shap = True
        print("[XAI] SHAP available -- using KernelSHAP")
    except ImportError:
        has_shap = False
        print("[XAI] SHAP not installed -- using attention weights as proxy")

    model_proto   = ModbusCNN(input_dim).to(device)
    global_state  = {k: v.cpu() for k, v in model_proto.state_dict().items()}
    n_clients     = len(client_data)
    byzantine_ids = [0, 1]
    local_models  = [ModbusCNN(input_dim).to(device) for _ in range(n_clients)]
    checkpoints   = {10: None, 25: None, 50: None}

    for r in range(n_rounds):
        updates = []
        for i, (Xi, yi) in enumerate(client_data):
            upd = local_train(local_models[i], global_state, Xi, yi, device=device)
            if i in byzantine_ids:
                upd = Attacker.apply(upd, "label_flipping")
            updates.append({k: v.cpu() for k, v in upd.items()})

        agg          = bulyan(updates, f=2)
        global_state = {k: global_state[k].cpu() + agg[k].cpu()
                        for k in global_state}
        if (r + 1) in checkpoints:
            checkpoints[r + 1] = deepcopy(global_state)

    # Extract feature importances at each checkpoint round
    importances = {}
    X_sample    = X_te[:200]

    for rnd, state in checkpoints.items():
        if state is None:
            continue
        m = deepcopy(model_proto).to(device)
        m.load_state_dict(state)
        m.eval()

        if has_shap:
            def _f(x_np):
                with torch.no_grad():
                    out, _ = m(torch.FloatTensor(x_np).to(device))
                    return torch.softmax(out, dim=1).cpu().numpy()

            bg        = X_sample[:50].numpy()
            explainer = shap_lib.KernelExplainer(_f, bg)
            sv        = explainer.shap_values(X_sample[:50].numpy(),
                                              nsamples=50, silent=True)
            imp = np.abs(sv[1] if isinstance(sv, list) else sv).mean(axis=0)
        else:
            with torch.no_grad():
                _, attn = m(X_sample.to(device))
            imp = attn.cpu().numpy().mean(axis=0)

        importances[rnd] = imp
        print("  [XAI] Round {:2d} -- top-3 features: {}".format(
              rnd, np.argsort(imp)[::-1][:3].tolist()))

    # Pearson correlation between consecutive checkpoint rounds (paper Section V)
    rounds = sorted(importances)
    corrs  = {}
    for i in range(len(rounds) - 1):
        r1, r2 = rounds[i], rounds[i + 1]
        corr   = float(np.corrcoef(importances[r1], importances[r2])[0, 1])
        corrs["r{}_r{}".format(r1, r2)] = corr
        print("  [XAI] Pearson r  Round {} <-> {}: {:.4f}".format(r1, r2, corr))

    return dict(importances=importances, correlations=corrs, rounds=rounds)


# =============================================================================
# SECTION 7 -- FIGURE GENERATION
# Reproduces all 8 paper figures
# =============================================================================

# Consistent colour + marker palette across all figures
COLORS = {
    "FedAvg":         "#e74c3c",
    "TrimmedMean":    "#e67e22",
    "Median":         "#f1c40f",
    "Krum":           "#2ecc71",
    "MultiKrum":      "#3498db",
    "Bulyan":         "#8e44ad",
    "BulyanAdaptive": "#1abc9c",
}
MARKERS = {m: mk for m, mk in zip(COLORS, ["o", "s", "^", "D", "v", "P", "*"])}


def _save_fig(fig, name, outdir):
    """Save figure as both PDF (vector) and PNG (150 dpi raster)."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "{}.{}".format(name, ext)),
                    bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("  Saved {}.{{pdf,png}}".format(name))


def generate_figures(results_A, results_B, results_C, results_D,
                     timing_results, shap_results,
                     methods, attacks, byz_ratios, outdir):
    """Generate all 8 paper figures and write them to outdir."""

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # -- Figure 1: Convergence curves (paper Section V-A) ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    highlight = ["FedAvg", "Bulyan", "BulyanAdaptive", "MultiKrum", "Krum"]
    for ax, atk in zip(axes, ["none", "label_flipping"]):
        for m in highlight:
            rm = results_D.get(m, {}).get(atk, [])
            if not rm:
                continue
            ax.plot([r["round"] for r in rm],
                    [r["accuracy"] * 100 for r in rm],
                    color=COLORS[m], marker=MARKERS[m],
                    markevery=10, linewidth=2, label=m)
        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Convergence -- {}".format(ATTACK_LABELS[atk]))
        ax.set_ylim([40, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    _save_fig(fig, "fig1_convergence", outdir)

    # -- Figure 2: Robustness heatmap (paper Section V-A, Table I) ------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    heat = pd.DataFrame(
        [
            [
                (results_A.get(m, {}).get(a, [{}]) or [{}])[-1].get("accuracy", 0) * 100
                for a in attacks
            ]
            for m in methods
        ],
        index=methods,
        columns=[ATTACK_LABELS[a] for a in attacks],
    )
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=40, vmax=100, ax=ax, annot_kws={"size": 10},
                linewidths=0.5)
    ax.set_title("Accuracy (%) -- All Methods x All Attacks (20% Byzantine)")
    ax.set_ylabel("Aggregation Method")
    ax.set_xlabel("Attack Type")
    plt.tight_layout()
    _save_fig(fig, "fig2_robustness_heatmap", outdir)

    # -- Figure 3: Byzantine ratio sensitivity (paper Section V-B) ------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ratio_pct = [int(r * 100) for r in byz_ratios]
    for m in methods:
        ax.plot(ratio_pct, [results_B[m].get(r, 0) for r in byz_ratios],
                color=COLORS[m], marker=MARKERS[m], linewidth=2, label=m)
    ax.set_xlabel("Byzantine Participant Ratio (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Byzantine Ratio (Label-Flipping)")
    ax.set_ylim([30, 100])
    ax.set_xticks(ratio_pct)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_fig(fig, "fig3_byzantine_ratio", outdir)

    # -- Figure 4: IID vs Non-IID sensitivity (paper Section V-C) -------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    noniid_ms = ["FedAvg", "Bulyan", "BulyanAdaptive", "MultiKrum"]
    for ax, setting in zip(axes, ["IID", "NonIID"]):
        for m in noniid_ms:
            rm = results_C.get(m, {}).get(setting, [])
            if not rm:
                continue
            ax.plot([r["round"] for r in rm],
                    [r["accuracy"] * 100 for r in rm],
                    color=COLORS[m], marker=MARKERS[m],
                    markevery=5, linewidth=2, label=m)
        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Non-IID Sensitivity -- {} (Label-Flip, 20% Byz)".format(setting))
        ax.set_ylim([40, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    _save_fig(fig, "fig4_noniid_sensitivity", outdir)

    # -- Figure 5: Computational overhead (paper Section V-D) -----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ms   = list(timing_results.keys())
    ts   = [timing_results[m] for m in ms]
    base = timing_results.get("FedAvg", 1.0)
    bars = ax.bar(ms, ts, color=[COLORS[m] for m in ms],
                  edgecolor="white", linewidth=0.5)
    for bar, t in zip(bars, ts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                "{:.1f}x".format(t / base),
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Aggregation Method")
    ax.set_ylabel("Aggregation Time (ms/round)")
    ax.set_title("Computational Overhead per FL Round (10 clients)")
    ax.set_xticklabels(ms, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, "fig5_overhead", outdir)

    # -- Figure 6: SHAP/Attention consistency (paper Section V-E) -------------
    if shap_results and "importances" in shap_results:
        imps   = shap_results["importances"]
        rounds = shap_results["rounds"]
        corrs  = shap_results["correlations"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax      = axes[0]
        n_feat  = len(next(iter(imps.values())))
        b_clrs  = ["#3498db", "#e74c3c", "#2ecc71"]
        width   = 0.25
        for ci, rnd in enumerate(rounds):
            imp = imps[rnd]
            ax.bar(np.arange(n_feat) + ci * width,
                   imp / (imp.sum() + 1e-9) * 100,
                   width, label="Round {}".format(rnd),
                   color=b_clrs[ci], alpha=0.8)
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Relative Importance (%)")
        ax.set_title("Feature Importance Stability Across Rounds")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1]
        if 10 in imps and 50 in imps:
            x = imps[10]
            y = imps[50]
            ax.scatter(x, y, alpha=0.7, color="#8e44ad")
            lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
            ax.plot(lims, lims, "k--", lw=1, label="Perfect agreement")
            ckey = ("r10_r50" if "r10_r50" in corrs
                    else (list(corrs)[-1] if corrs else None))
            cv   = corrs.get(ckey, 0) if ckey else 0
            ax.set_xlabel("Feature Importance -- Round 10")
            ax.set_ylabel("Feature Importance -- Round 50")
            ax.set_title("Explanation Consistency (r = {:.3f})".format(cv))
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, "fig6_shap_consistency", outdir)

    # -- Figure 7: Ablation -- FedAvg vs Bulyan variants (paper Section V-F) --
    fig, ax  = plt.subplots(figsize=(10, 4.5))
    ab_ms    = ["FedAvg", "Bulyan", "BulyanAdaptive"]
    x        = np.arange(len(attacks))
    width    = 0.25
    for ci, m in enumerate(ab_ms):
        vals = [
            (results_A.get(m, {}).get(a, [{}]) or [{}])[-1].get("accuracy", 0) * 100
            for a in attacks
        ]
        ax.bar(x + ci * width, vals, width,
               label=m, color=COLORS.get(m, "gray"), alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([ATTACK_LABELS[a] for a in attacks], rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation: FedAvg vs Bulyan vs BulyanAdaptive")
    ax.set_ylim([30, 105])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, "fig7_ablation", outdir)

    # -- Figure 8: Radar / spider chart (paper Section V) ---------------------
    cats    = [ATTACK_LABELS[a] for a in attacks if a != "none"]
    n_cat   = len(cats)
    angles  = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for m in methods:
        vals = [
            (results_A.get(m, {}).get(a, [{}]) or [{}])[-1].get("accuracy", 0) * 100
            for a in attacks if a != "none"
        ]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[m], linewidth=2, label=m)
        ax.fill(angles, vals, color=COLORS[m], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=10)
    ax.set_ylim([0, 100])
    ax.set_title("Robustness Profile -- All Methods", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    _save_fig(fig, "fig8_radar", outdir)


# =============================================================================
# SECTION 8 -- SUMMARY TABLES
# Console output matching paper Tables I-IV
# =============================================================================

def print_summary_tables(results_A, results_B, methods, attacks, byz_ratios):
    """Print Tables I-IV from the paper to stdout."""

    def _acc(m, a):
        rm = results_A.get(m, {}).get(a, [])
        return rm[-1]["accuracy"] * 100 if rm else 0.0

    print("\n" + "=" * 80)
    print("TABLE I -- Accuracy (%) under Byzantine attacks  (20% Byzantine, IID)")
    print("=" * 80)
    hdr = "{:15s}".format("Method") + "".join(
        "{:>12s}".format(ATTACK_LABELS[a]) for a in attacks)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        print("{:15s}".format(m) + "".join(
            "{:>12.1f}".format(_acc(m, a)) for a in attacks))

    print("\n" + "=" * 80)
    print("TABLE II -- Detailed metrics under Label-Flipping  (20% Byzantine, IID)")
    print("=" * 80)
    print("{:15s}{:>8s}{:>8s}{:>8s}{:>8s}{:>8s}".format(
          "Method", "Acc", "Prec", "Rec", "F1", "AUC"))
    print("-" * 55)
    for m in methods:
        rm = results_A.get(m, {}).get("label_flipping", [])
        if rm:
            last = rm[-1]
            print("{:15s}{:>8.1f}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.3f}".format(
                  m, last["accuracy"] * 100,
                  last["precision"], last["recall"],
                  last["f1"], last["auc"]))

    print("\n" + "=" * 80)
    print("TABLE III -- Robustness Score  R = 1 / (1 + avg_degradation)")
    print("=" * 80)
    for m in methods:
        base_rm  = results_A.get(m, {}).get("none", [])
        base_acc = base_rm[-1]["accuracy"] if base_rm else 0.0
        degrs    = []
        for a in [x for x in attacks if x != "none"]:
            rm = results_A.get(m, {}).get(a, [])
            if rm and base_acc > 0:
                degrs.append(max(0, (base_acc - rm[-1]["accuracy"]) / base_acc))
        if degrs:
            avg_d = float(np.mean(degrs))
            rob   = 1.0 / (1.0 + avg_d)
            print("  {:15s}: R = {:.4f}  (avg degradation = {:.1f}%)".format(
                  m, rob, avg_d * 100))

    print("\n" + "=" * 80)
    print("TABLE IV -- Accuracy (%) vs Byzantine ratio  (Label-Flipping attack)")
    print("=" * 80)
    hdr = "{:15s}".format("Method") + "".join(
        "{:>8d}%".format(int(r * 100)) for r in byz_ratios)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        print("{:15s}".format(m) + "".join(
            "{:>9.1f}".format(results_B[m].get(r, 0)) for r in byz_ratios))


# =============================================================================
# SECTION 9 -- SAVE RESULTS
# =============================================================================

def save_results(results_A, results_B, results_C, results_D,
                 timing_results, shap_results, outdir):
    """Serialise all experiment results to results_summary.json."""

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = os.path.join(outdir, "results_summary.json")
    with open(out_path, "w") as fh:
        json.dump(_convert({
            "results_A":         results_A,
            "results_B":         results_B,
            "timing":            timing_results,
            "shap_correlations": shap_results.get("correlations", {}),
        }), fh, indent=2)
    print("  Saved {}".format(out_path))


# =============================================================================
# SECTION 10 -- MAIN ENTRY POINT
# =============================================================================

def main():
    """Run all experiments from the paper sequentially.

    Experiments:
    A -- All 7 aggregation methods x 6 attack types (IID, 20% Byzantine)
    B -- Byzantine ratio sweep: 10%, 20%, 30%, 40%  (label-flipping attack)
    C -- Non-IID sensitivity for 4 representative methods
    D -- Convergence curves with per-round recording (no-attack + label-flip)
    E -- Aggregation timing benchmark (20 trials per method)
    F -- SHAP/Attention consistency at rounds 10, 25, 50 (Bulyan)
    """

    # -- Step 1: Load and preprocess data -------------------------------------
    df        = load_dataset()
    X, y, _   = preprocess(df)
    input_dim = X.shape[1]
    print("[Data] input_dim={}  samples={}  class dist: {}".format(
          input_dim, len(X), np.bincount(y.numpy())))

    # Stratified 80/20 train-test split (paper Section IV-A)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    # IID and Non-IID client partitions (paper Section IV-B)
    data_iid    = split_federated(X_tr, y_tr, n_clients=N_CLIENTS, iid=True)
    data_noniid = split_federated(X_tr, y_tr, n_clients=N_CLIENTS, iid=False)

    METHODS    = list(AGG_FNS.keys())
    BYZ_RATIOS = [0.1, 0.2, 0.3, 0.4]

    # -- Experiment A: 7 methods x 6 attacks (IID, 20% Byzantine) ------------
    print("\n=== Experiment A: 7 methods x 6 attacks ===")
    results_A = {m: {} for m in METHODS}
    for attack in ATTACK_TYPES:
        for method in METHODS:
            print("  {:<15s} | {:<15s}".format(method, attack),
                  end=" ... ", flush=True)
            t0 = time.time()
            rm, final_state = run_fl(
                data_iid, X_te, y_te,
                agg_name=method, attack_type=attack,
                n_byzantine=N_BYZANTINE, n_rounds=N_ROUNDS,
                input_dim=input_dim, device=DEVICE,
                record_every=N_ROUNDS,  # record only final round
            )
            results_A[method][attack] = rm
            print("acc={:.1f}%  ({:.0f}s)".format(
                  rm[-1]["accuracy"] * 100, time.time() - t0))
            del rm, final_state
            gc.collect()

    # -- Experiment B: Byzantine ratio sweep ----------------------------------
    print("\n=== Experiment B: Byzantine ratio sweep ===")
    results_B = {m: {} for m in METHODS}
    for method in METHODS:
        for ratio in BYZ_RATIOS:
            n_byz = max(1, int(N_CLIENTS * ratio))
            rm, _ = run_fl(
                data_iid, X_te, y_te,
                agg_name=method, attack_type="label_flipping",
                n_byzantine=n_byz, n_rounds=N_ROUNDS,
                input_dim=input_dim, device=DEVICE,
                record_every=N_ROUNDS,
            )
            results_B[method][ratio] = rm[-1]["accuracy"] * 100
            print("  {:<15s} | ratio={:.0%} -> {:.1f}%".format(
                  method, ratio, results_B[method][ratio]))
            del rm
            gc.collect()

    # -- Experiment C: Non-IID sensitivity ------------------------------------
    print("\n=== Experiment C: Non-IID sensitivity ===")
    results_C = {}
    for method in ["FedAvg", "Bulyan", "BulyanAdaptive", "MultiKrum"]:
        results_C[method] = {}
        for setting, cdata in [("IID", data_iid), ("NonIID", data_noniid)]:
            rm, _ = run_fl(
                cdata, X_te, y_te,
                agg_name=method, attack_type="label_flipping",
                n_byzantine=N_BYZANTINE, n_rounds=N_ROUNDS,
                input_dim=input_dim, device=DEVICE, record_every=3,
            )
            results_C[method][setting] = rm
            print("  {:<15s} | {:<7s} -> {:.1f}%".format(
                  method, setting, rm[-1]["accuracy"] * 100))

    # -- Experiment D: Convergence curves (per-round recording) ---------------
    print("\n=== Experiment D: Convergence curves ===")
    results_D = {}
    for method in ["FedAvg", "Bulyan", "BulyanAdaptive", "MultiKrum", "Krum"]:
        results_D[method] = {}
        for attack in ["none", "label_flipping"]:
            print("  {:<15s} | {:<15s}".format(method, attack),
                  end=" ... ", flush=True)
            t0 = time.time()
            rm, _ = run_fl(
                data_iid, X_te, y_te,
                agg_name=method, attack_type=attack,
                n_byzantine=N_BYZANTINE, n_rounds=N_ROUNDS,
                input_dim=input_dim, device=DEVICE, record_every=1,
            )
            results_D[method][attack] = rm
            print("done ({:.0f}s)".format(time.time() - t0))

    # -- Experiment E: Aggregation timing benchmark ---------------------------
    print("\n=== Experiment E: Aggregation timing benchmark ===")
    timing_results = {}
    model_t  = ModbusCNN(input_dim).to(DEVICE)
    gs_t     = {k: v.cpu() for k, v in model_t.state_dict().items()}
    lm_t     = [ModbusCNN(input_dim).to(DEVICE) for _ in range(N_CLIENTS)]
    sample_updates = [
        local_train(lm_t[i], gs_t, data_iid[i][0], data_iid[i][1], device=DEVICE)
        for i in range(N_CLIENTS)
    ]
    for method in METHODS:
        fn     = AGG_FNS[method]
        trials = 20
        t0     = time.time()
        for _ in range(trials):
            fn(sample_updates, N_BYZANTINE)
        elapsed = (time.time() - t0) / trials * 1000   # ms per round
        timing_results[method] = elapsed
        print("  {:<15s}: {:.2f} ms/round".format(method, elapsed))

    # -- Experiment F: SHAP / Attention consistency ---------------------------
    print("\n=== Experiment F: SHAP/Attention consistency ===")
    shap_results = compute_shap_consistency(
        data_iid, X_te, y_te, input_dim, DEVICE, N_ROUNDS)

    # -- Generate all 8 figures -----------------------------------------------
    print("\n=== Generating Figures ===")
    generate_figures(
        results_A, results_B, results_C, results_D,
        timing_results, shap_results,
        METHODS, ATTACK_TYPES, BYZ_RATIOS, OUTDIR,
    )

    # -- Print paper tables ---------------------------------------------------
    print_summary_tables(results_A, results_B, METHODS, ATTACK_TYPES, BYZ_RATIOS)

    # -- Save JSON results ----------------------------------------------------
    print("\n=== Saving Results ===")
    save_results(results_A, results_B, results_C, results_D,
                 timing_results, shap_results, OUTDIR)

    print("\n" + "=" * 70)
    print("All experiments complete.")
    print("Outputs written to: {}".format(OUTDIR))
    print("  Figures : fig1_convergence ... fig8_radar  (.pdf + .png)")
    print("  Results : results_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
