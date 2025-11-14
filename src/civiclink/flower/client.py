import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from pipeline.model import FTTransformer
from pipeline.engine import train, test

# flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader):
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

def main():
    # load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(".", train=True, download=True, transform=transform)
    testset = datasets.MNIST(".", train=False, download=True, transform=transform)
    # createe data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)
    # create model
    model = FTTransformer()
    # start client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model, trainloader, testloader),
    )

if __name__ == "__main__":
    main()
