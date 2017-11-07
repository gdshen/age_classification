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
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

config = DefaultConfig()

train_loader = DataLoader(
    IMDBWIKIDatasets(config.imdb_csv_train, train=True, transform=transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor()
    ])), batch_size=config.batch_size, shuffle=True,
    num_workers=config.num_workers
)
test_loader = DataLoader(
    IMDBWIKIDatasets(config.imdb_csv_test, train=False, transform=transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor()
    ])), batch_size=config.batch_size, shuffle=False,
    num_workers=config.num_workers
)

# train_loader = DataLoader(
#     AsianFaceDatasets(config.asian_csv_train, config.asian_imgs_dir, train=True, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor()
#     ])), batch_size=config.batch_size, shuffle=True,
#     num_workers=config.num_workers
# )
#
# test_loader = DataLoader(
#     AsianFaceDatasets(config.asian_csv_test, config.asian_imgs_dir, train=False, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor()
#     ])), batch_size=config.batch_size, shuffle=False,
#     num_workers=config.num_workers
# )

model = Net()
model.cuda()

optimizer = optim.SGD([{'params': model.features.parameters()},
                       {'params': model.classifier.parameters()},
                       {'params': model.fc.parameters(), 'lr': config.fc_learning_rate}], lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)

scheduler = StepLR(optimizer, step_size=config.decay_epoches, gamma=config.decay_gamma)


def train(epoch, writer):
    scheduler.step()
    model.train()
    # print learning rate
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'], end='\t')
    # print('\n')

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
            writer.add_scalar('train/loss', loss.data[0],
                              (epoch - 1) * len(train_loader.dataset) + batch_idx * config.batch_size)
            print(
                f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100 * batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.data[0]:.6f}')

        if epoch % config.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f'checkpoint-{epoch}.pth'))


def test(epoch, writer):
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        target = target.type(torch.FloatTensor)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = F.l1_loss(output, target).cpu()
        test_loss += loss.data[0] * config.batch_size
    test_loss /= len(test_loader.dataset)

    writer.add_scalar('test/loss', test_loss, epoch)
    print(f'Testing Accuracy is {test_loss}')


if __name__ == '__main__':
    writer = SummaryWriter(config.logs_dir)
    for epoch in range(1, config.epoch + 1):
        train(epoch, writer)
        test(epoch, writer)
    writer.close()
