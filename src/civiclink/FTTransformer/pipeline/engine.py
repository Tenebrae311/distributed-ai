import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score

# train function
def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    running_loss = 0.0
    for x_num, x_cat, target in trainloader:
        optimizer.zero_grad()

        logits = model(x_num.to(device), x_cat.to(device))          # (B,)
        loss = criterion(logits, target.float().to(device))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)

# validation function
def val(
    model: nn.Module,
    valloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    """Validation loop.

    Returns:
        avg_loss, roc_auc, precision, mean_prob
    """
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_num, x_cat, target in valloader:
            logits = model(x_num.to(device), x_cat.to(device))        # (B,)
            total_loss += criterion(logits, target.float().to(device)).item()
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(valloader)
    roc_auc = roc_auc_score(all_targets, all_probs)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    mean_prob = float(sum(all_probs) / len(all_probs)) if len(all_probs) > 0 else 0.0
    return avg_loss, roc_auc, precision, mean_prob
