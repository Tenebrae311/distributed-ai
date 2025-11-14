import torch
import torch.nn as nn
import torch.optim as optim


# train function
def train(model: nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int = 1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for data, target in trainloader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

# test function
def val(model: nn.Module, valloader: torch.utils.data.DataLoader) -> tuple[float, float]:
    model.eval()
    correct, loss = 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in valloader:
            output: torch.Tensor = model(data)
            loss += criterion(output, target).item()
            correct += (output.argmax(1) == target).sum().item()
    return loss / len(valloader), correct / len(valloader.dataset)