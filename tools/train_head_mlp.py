# tools/train_head_mlp.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    labels = list(d["labels"]) if "labels" in d.files else ["cough", "other"]
    return X, y, labels

def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return mu.astype(np.float32), std.astype(np.float32)

def standardize_apply(X: np.ndarray, mu: np.ndarray, std: np.ndarray):
    return ((X - mu) / std).astype(np.float32)

class MLPHead(nn.Module):
    def __init__(self, in_dim=2048, hidden=256, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def eval_probs(model, X, device):
    model.eval()
    xb = torch.from_numpy(X).to(device)
    logits = model(xb)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

def eval_with_threshold(y_true, cough_prob, thr, cough_id=0, other_id=1):
    y_pred = np.where(cough_prob >= thr, cough_id, other_id).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m, y_pred

def search_best_threshold(y_true, cough_prob, cough_id=0, other_id=1):
    best_thr = 0.5
    best_f1 = -1.0
    best_acc = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        acc, f1m, _ = eval_with_threshold(y_true, cough_prob, float(thr), cough_id, other_id)
        if f1m > best_f1:
            best_f1, best_acc, best_thr = f1m, acc, float(thr)
    return best_acc, best_f1, best_thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--out", default="checkpoints/head_mlp.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--cough_id", type=int, default=0)
    ap.add_argument("--other_id", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Xtr, ytr, labels = load_npz(args.train)
    Xva, yva, labels2 = load_npz(args.val)
    if labels2:
        labels = labels2

    # 标准化（只用 train 拟合）
    mu, std = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mu, std)
    Xva = standardize_apply(Xva, mu, std)

    device = "cpu"
    model = MLPHead(in_dim=Xtr.shape[1], hidden=args.hidden, num_classes=len(labels), dropout=args.dropout).to(device)

    # class weight（防止轻微不平衡）
    counts = np.bincount(ytr, minlength=len(labels)).astype(np.float32)
    weights = counts.sum() / (len(labels) * (counts + 1e-6))
    w = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best_f1 = -1.0
    best_state = None
    best_thr = 0.5

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * xb.size(0)
            n += xb.size(0)

        # val
        probs_va = eval_probs(model, Xva, device)
        cough_prob_va = probs_va[:, args.cough_id]
        acc05, f105, _ = eval_with_threshold(yva, cough_prob_va, 0.5, args.cough_id, args.other_id)
        accb, f1b, thrb = search_best_threshold(yva, cough_prob_va, args.cough_id, args.other_id)

        print(f"Epoch {ep:02d} | train_loss={loss_sum/n:.4f} | val_acc@0.5={acc05:.4f} val_f1@0.5={f105:.4f} | "
              f"best_thr={thrb:.2f} best_acc={accb:.4f} best_f1={f1b:.4f}")

        if f1b > best_f1:
            best_f1 = f1b
            best_thr = thrb
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存 best
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "mu": mu,
        "std": std,
        "labels": labels,
        "best_thr": float(best_thr),
        "arch": {"hidden": args.hidden, "dropout": args.dropout},
    }, out)
    print(f"\n[OK] saved -> {out}  (best_thr={best_thr:.2f}, best_val_macro_f1={best_f1:.4f})")

    # 详细打印 val 报告（用 best_thr）
    model.load_state_dict(best_state)
    probs_va = eval_probs(model, Xva, device)
    cough_prob_va = probs_va[:, args.cough_id]
    accv, f1v, ypv = eval_with_threshold(yva, cough_prob_va, best_thr, args.cough_id, args.other_id)
    print("\n=== VAL (best_thr) ===")
    print(f"ACC={accv:.4f}  Macro-F1={f1v:.4f}  thr={best_thr:.2f}")
    print(confusion_matrix(yva, ypv, labels=[args.cough_id, args.other_id]))
    print(classification_report(yva, ypv, target_names=labels, digits=4))

    # 可选 test_200
    if args.test:
        Xte, yte, _ = load_npz(args.test)
        Xte = standardize_apply(Xte, mu, std)
        probs_te = eval_probs(model, Xte, device)
        cough_prob_te = probs_te[:, args.cough_id]
        acct, f1t, ypt = eval_with_threshold(yte, cough_prob_te, best_thr, args.cough_id, args.other_id)
        print("\n=== TEST (using best_thr from VAL) ===")
        print(f"ACC={acct:.4f}  Macro-F1={f1t:.4f}")
        print(confusion_matrix(yte, ypt, labels=[args.cough_id, args.other_id]))

if __name__ == "__main__":
    main()
