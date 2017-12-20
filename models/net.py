import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 101, False)

    def forward(self, x):
        x = self.resnet50(x)
        x = F.softmax(x, dim=1)
        return x
