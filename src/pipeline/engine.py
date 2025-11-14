import torch
import torch.nn as nn
import torch.optim as optim


# train function
def train(model: nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int = 1):
    criterion = nn.CrossEntropyLoss()                          
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in trainloader:
            optimizer.zero_grad()

            logits = model(data)                                    # (B, 3)
            loss = criterion(logits, target)                        # target shape: (B,)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")


# validation function (3 classes)
def val(model: nn.Module, valloader: torch.utils.data.DataLoader) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()                               
    model.eval()

    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in valloader:
            logits: torch.Tensor = model(data)                      # (B, 3)
            total_loss += criterion(logits, target).item()

            preds = logits.argmax(dim=1)                           
            correct += (preds == target).sum().item()

    avg_loss = total_loss / len(valloader)
    accuracy = correct / len(valloader.dataset)
    return avg_loss, accuracy
