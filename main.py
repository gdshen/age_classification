from data.face import FaceDatasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from config import DefaultConfig

config = DefaultConfig()

train_loader = DataLoader(
    FaceDatasets(config.imdb_csv_path, train=True, transform=transforms.Compose([
        transforms.Scale([224, 224]),
        transforms.ToTensor()
    ])), batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
)

test_loader = DataLoader(
    FaceDatasets(config.imdb_csv_path, train=True, transform=transforms.Compose([
        transforms.Scale([224, 224]),
        transforms.ToTensor()
    ])), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
)


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    # train_loader
    print('hello')
