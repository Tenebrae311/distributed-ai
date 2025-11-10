from torch import Tensor
import torch.nn as nn

# Einfaches Modell
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x.view(-1, 28 * 28))