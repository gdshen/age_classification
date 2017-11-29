from torchvision.models import vgg16, resnet50
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 101, False)

    def forward(self, x):
        x = self.resnet50(x)
        x = F.softmax(x)

        x = x @ Variable(torch.arange(0, 101)).view(-1, 1).cuda()

        return x
