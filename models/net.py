import torch.nn as nn
from torchvision.models import resnet50


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.resnet50 = nn.DataParallel(nn.Sequential(*list(resnet50(pretrained=True).children())[:-1]))
#         self.fc = nn.Linear(2048, 101, False)
#         # self.resnet50.fc = nn.Linear(2048, 101, False)
#
#     def forward(self, x):
#         x = self.resnet50(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = F.softmax(x, dim=1)
#         return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = nn.DataParallel(nn.Sequential(*list(resnet50(pretrained=True).children())[:-1]))
        self.fc = nn.Linear(2048, 1, False)

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
