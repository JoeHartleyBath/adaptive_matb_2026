"""Quick RNN/Transformer probe on sequential EEG features.

Builds short sequences of consecutive feature windows and trains a
1-layer GRU or small Transformer encoder, evaluated cross-subject.

Usage (from repo root):
    .venv\\Scripts\\python.exe scripts/quick_rnn_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from ml.pretrain_loader import calibration_norm_features, load_baseline_from_cache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_NORM_CACHE = _REPO_ROOT / "results" / "test_pretrain" / "norm_w2.0_s1.0.npz"
SEQ_LEN = 8           # 8 consecutive windows = 8 s context (at 1.0 s step)
HIDDEN = 48
N_HEADS = 4
N_LAYERS = 1
LR = 1e-3
EPOCHS = 20
BATCH = 128
SEED = 42
TEST_FRAC = 0.3
N_SEEDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading (same normalisation as optimise_logreg)
# ---------------------------------------------------------------------------
def load_data():
    import yaml
    qc = yaml.safe_load((_REPO_ROOT / "config" / "pretrain_qc.yaml").read_text())
    exclude = set((qc.get("excluded_participants") or {}).keys())

    npz = np.load(_NORM_CACHE, allow_pickle=False)
    pids = [p for p in list(npz["pids"]) if p not in exclude]
    feat_names = list(npz["feat_names"])

    baseline_by = load_baseline_from_cache(_NORM_CACHE, pids)
    X_by, y_by = {}, {}
    for pid in pids:
        raw = npz[f"{pid}_task_X"]
        bl = baseline_by[pid]
        X_by[pid] = calibration_norm_features(
            raw, bl["fix_X"], bl["forest_X"], bl["forest_bidx"])
        y_by[pid] = npz[f"{pid}_task_y"]
    return X_by, y_by, feat_names, sorted(pids)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Slide a window of seq_len over temporally ordered feature rows."""
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0, dtype=y.dtype)
    seqs = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
    # seqs shape: (n - seq_len + 1, F, seq_len) -> transpose to (N, seq_len, F)
    seqs = seqs.transpose(0, 2, 1).copy()
    labels = y[seq_len - 1:]  # label = last window in sequence
    return seqs, labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)  # last token


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_eval(model_cls, X_by, y_by, pids, seed, n_feat):
    rng = np.random.default_rng(seed)
    n_test = max(1, round(len(pids) * TEST_FRAC))
    test_pids = list(rng.choice(pids, size=n_test, replace=False))
    train_pids = [p for p in pids if p not in test_pids]

    # Build sequences per participant, then concat
    train_seqs, train_y = [], []
    for pid in train_pids:
        s, l = make_sequences(X_by[pid], y_by[pid], SEQ_LEN)
        if len(s) > 0:
            train_seqs.append(s)
            train_y.append(l)
    X_train = np.concatenate(train_seqs)
    y_train = np.concatenate(train_y)

    # Fit scaler on train (per-feature across all time steps)
    orig_shape = X_train.shape
    sc = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_feat)
    sc.fit(X_train_flat)
    X_train = sc.transform(X_train_flat).reshape(orig_shape)

    # DataLoader for CPU efficiency (don't load all into GPU memory)
    from torch.utils.data import TensorDataset, DataLoader
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True,
                        pin_memory=(DEVICE != "cpu"))

    model = model_cls(n_feat).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            device=DEVICE))

    # Train
    model.train()
    for ep in range(EPOCHS):
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{EPOCHS} loss={ep_loss/len(loader):.4f}",
                  flush=True)

    # Eval per test participant
    model.eval()
    all_true, all_prob = [], []
    per_pid = {}
    with torch.no_grad():
        for pid in test_pids:
            s, l = make_sequences(X_by[pid], y_by[pid], SEQ_LEN)
            if len(s) == 0:
                continue
            s_sc = sc.transform(s.reshape(-1, n_feat)).reshape(s.shape)
            logits = model(torch.tensor(s_sc, dtype=torch.float32, device=DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            try:
                per_pid[pid] = roc_auc_score(l, probs)
            except ValueError:
                per_pid[pid] = 0.5
            all_true.append(l)
            all_prob.append(probs)

    y_all = np.concatenate(all_true)
    p_all = np.concatenate(all_prob)
    try:
        overall = roc_auc_score(y_all, p_all)
    except ValueError:
        overall = 0.5
    return overall, per_pid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    X_by, y_by, feat_names, pids = load_data()
    n_feat = len(feat_names)
    print(f"Participants: {len(pids)}, Features: {n_feat}, SeqLen: {SEQ_LEN}")
    # Show dataset size
    total = sum(max(0, len(X_by[p]) - SEQ_LEN + 1) for p in pids)
    print(f"Total sequences: ~{total}")

    seeds = list(range(SEED, SEED + N_SEEDS))

    for name, cls in [("GRU", GRUClassifier), ("Transformer", TransformerClassifier)]:
        aucs = []
        for s in seeds:
            auc, _ = train_eval(cls, X_by, y_by, pids, s, n_feat)
            aucs.append(auc)
            print(f"  {name} seed={s}: AUC={auc:.4f}")
        print(f"  {name} mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print()

    # LogReg baseline for direct comparison (same splits)
    from sklearn.linear_model import LogisticRegression
    print("LogReg baseline (same splits, no sequence):")
    lr_aucs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        n_test = max(1, round(len(pids) * TEST_FRAC))
        test_pids = list(rng.choice(pids, size=n_test, replace=False))
        train_pids = [p for p in pids if p not in test_pids]
        X_tr = np.concatenate([X_by[p] for p in train_pids])
        y_tr = np.concatenate([y_by[p] for p in train_pids])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        clf = LogisticRegression(C=0.003, max_iter=2000, class_weight="balanced",
                                 random_state=s)
        clf.fit(X_tr, y_tr)
        all_t, all_p = [], []
        for pid in test_pids:
            pr = clf.predict_proba(sc.transform(X_by[pid]))[:, 1]
            all_t.append(y_by[pid])
            all_p.append(pr)
        try:
            auc = roc_auc_score(np.concatenate(all_t), np.concatenate(all_p))
        except ValueError:
            auc = 0.5
        lr_aucs.append(auc)
        print(f"  LogReg seed={s}: AUC={auc:.4f}")
    print(f"  LogReg mean AUC: {np.mean(lr_aucs):.4f} +/- {np.std(lr_aucs):.4f}")


if __name__ == "__main__":
    main()
