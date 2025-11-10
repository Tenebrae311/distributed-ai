import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Einfaches Modell
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

# Trainingsfunktion
def train(model, trainloader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for data, target in trainloader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

# Testfunktion
def test(model, testloader):
    model.eval()
    correct, loss = 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            loss += criterion(output, target).item()
            correct += (output.argmax(1) == target).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)

# Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# Client starten
def main():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(".", train=True, download=True, transform=transform)
    testset = datasets.MNIST(".", train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)

    model = Net()

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model, trainloader, testloader),
    )

if __name__ == "__main__":
    main()
