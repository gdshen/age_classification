from torchvision.models import vgg16
import torch.nn as nn
from torch.autograd import Variable
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        # for m in self.vgg.modules():
        #     if isinstance(m, nn.Conv2d):
        #         print(m.weight)
        self.features = self.vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 101)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = Net()
    img = Variable(torch.Tensor(1, 3, 224, 224))
    y = net(img)
    print(y)

