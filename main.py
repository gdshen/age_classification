import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from config import DefaultConfig
from data.face import IMDBWIKIDatasets, AsianFaceDatasets
from models.net import Net

config = DefaultConfig()

# train_loader = DataLoader(
#     IMDBWIKIDatasets(config.imdb_csv_path, train=True, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor()
#     ])), batch_size=config.batch_size, shuffle=True,
#     num_workers=config.num_workers
# )
train_loader = DataLoader(
    AsianFaceDatasets(config.asian_csv_path, config.asian_imgs_dir, train=True, transform=transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor()
    ])), batch_size=config.batch_size, shuffle=True,
    num_workers=config.num_workers
)

test_loader = DataLoader(
    IMDBWIKIDatasets(config.imdb_csv_path, train=True, transform=transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor()
    ])), batch_size=config.batch_size, shuffle=False,
    num_workers=config.num_workers
)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data)
        # print(type(data))
        # print(type(target))
        # print(target)
        # data, target = data.cuda(), target.cuda()
        # print(data.shape)
        # print(target.shape)
        data, target = Variable(data), Variable(target)
        # optimizer.zero_grad()
        output = model(data)

        print(output)
        # todo
        # loss = None
        # loss.backward()
        # optimizer.step()
        # if batch_idx % config.log_interval == 0:
        #     print('Train Epoch:')


def test():
    pass


if __name__ == '__main__':
    for epoch in range(1, config.epoch + 1):
        train(epoch)
