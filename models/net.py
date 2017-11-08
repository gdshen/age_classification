from torchvision.models import vgg16
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        # for m in self.vgg.modules():
        #     if isinstance(m, nn.Conv2d):
        #         print(m.weight)
        self.features = nn.DataParallel(self.vgg.features)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.fc = nn.Linear(4096, 101, False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc(x)
        x = F.softmax(x)

        x = x @ Variable(torch.arange(0, 101)).view(-1, 1).cuda()
        return x
