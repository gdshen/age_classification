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
import os

config = DefaultConfig()

# train_loader = DataLoader(
#     IMDBWIKIDatasets(config.imdb_csv_path, train=True, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor()
#     ])), batch_size=config.batch_size, shuffle=True,
#     num_workers=config.num_workers
# )
# test_loader = DataLoader(
#     IMDBWIKIDatasets(config.imdb_csv_path, train=True, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor()
#     ])), batch_size=config.batch_size, shuffle=False,
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
    AsianFaceDatasets(config.asian_csv_path, config.asian_imgs_dir, train=False, transform=transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor()
    ])), batch_size=config.batch_size, shuffle=False,
    num_workers=config.num_workers
)

model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.FloatTensor)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % config.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100 * batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.data[0]:.6f}')

        if epoch % config.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f'checkpoint-{epoch}.pth'))


def test():
    pass


if __name__ == '__main__':
    for epoch in range(1, config.epoch + 1):
        train(epoch)
