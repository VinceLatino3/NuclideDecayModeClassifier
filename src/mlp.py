import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, random_split
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, random_split

LABELS = ["beta_minus", "beta_plus_ec", "alpha", "stable", "other"]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
def pick_device():
    if torch.cuda.is_available():
        try:
            (torch.zeros(1, device="cuda") + 1).item()   # forces a real GPU kernel to run
            return "cuda"
        except Exception:
            print("CUDA detected but this PyTorch build can't run on the GPU; using CPU")
    return "cpu"

DEVICE = pick_device()


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def prepare(train_df, val_df):
    # the scaler learns its mean/std from the training rows only
    scaler = StandardScaler().fit(train_df[TIER1])

    def to_tensors(df):
        X = torch.tensor(scaler.transform(df[TIER1]), dtype=torch.float32)
        y = torch.tensor(df["label"].map(LABEL_TO_ID).to_numpy(), dtype=torch.long)
        return X, y

    return to_tensors(train_df), to_tensors(val_df)


def class_weights(y):
    # inverse class frequency: rare classes count more in the loss
    counts = torch.bincount(y, minlength=len(LABELS)).float()
    return counts.sum() / (len(LABELS) * counts)


def train_mlp(train_df, val_df, epochs=300, patience=30, lr=1e-3,
              weight_decay=1e-4, batch_size=64, use_class_weights=True):
    torch.manual_seed(SEED)
    (Xtr, ytr), (Xva, yva) = prepare(train_df, val_df)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    Xva_d = Xva.to(DEVICE)

    model = MLP(Xtr.shape[1], len(LABELS)).to(DEVICE)
    w = class_weights(ytr).to(DEVICE) if use_class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva_d).argmax(dim=1).cpu().numpy()
        f1 = f1_score(yva.numpy(), pred, average="macro", zero_division=0)

        if f1 > best_f1:   # keep the best epoch, stop if no improvement for `patience` epochs
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  val macro-F1 {f1:.3f}  (best {best_f1:.3f})")
        if bad >= patience:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_f1


if __name__ == "__main__":
    print("device:", DEVICE)
    df = load_main_dataset()
    train, _ = random_split(df)          # test set deliberately ignored
    tr, va = train_test_split(train, test_size=0.15,
                              stratify=train["label"], random_state=SEED)
    print("train:", len(tr), " val:", len(va))

    for weighted in (True, False):
        print(f"\n--- class weights: {weighted} ---")
        model, best_f1 = train_mlp(tr, va, use_class_weights=weighted)
        print(f"best validation macro-F1: {best_f1:.3f}")