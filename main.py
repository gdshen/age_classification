import fire
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from data.face import WholeFaceDatasets
from models.net import Net


## IMDB data loader
# train_loader = DataLoader(
#     IMDBWIKIDatasets(config.imdb_csv_train, train=True, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor()
#     ])), batch_size=config.batch_size, shuffle=True,
#     num_workers=config.num_workers
# )
# test_loader = DataLoader(
#     IMDBWIKIDatasets(config.imdb_csv_test, train=False, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor()
#     ])), batch_size=config.batch_size, shuffle=False,
#     num_workers=config.num_workers
# )

## Asian data loader
# train_loader = DataLoader(
#     AsianFaceDatasets(config.asian_csv_train, config.asian_imgs_dir, train=True, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])), batch_size=config.batch_size, shuffle=True,
#     num_workers=config.num_workers
# )
#
# test_loader = DataLoader(
#     AsianFaceDatasets(config.asian_csv_test, config.asian_imgs_dir, train=False, transform=transforms.Compose([
#         transforms.Scale((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])), batch_size=config.batch_size, shuffle=False,
#     num_workers=config.num_workers
# )

def train(**kwargs):
    config.parse(kwargs)
    writer = SummaryWriter(config.logs_dir)
    train_loader = DataLoader(
        WholeFaceDatasets(config.whole_csv_train, config.whole_imgs_base_dir, train=True, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])), batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers
    )

    test_loader = DataLoader(
        WholeFaceDatasets(config.whole_csv_test, config.whole_imgs_base_dir, train=False, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])), batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )

    weights_vector = Variable(torch.arange(0, 101)).view(-1, 1).cuda()
    model = Net()
    if config.using_pretrain_model:
        model.load_state_dict(torch.load(config.pretrain_model_path))
    model.cuda()

    fc_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())

    optimizer = optim.SGD([{'params': base_params},
                           {'params': model.fc.parameters(), 'lr': config.fc_learning_rate}],
                          lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)

    scheduler = StepLR(optimizer, step_size=config.decay_epoches, gamma=config.decay_gamma)

    for epoch in range(config.epoch):
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
                # writer.add_scalar('train/loss', loss.data[0],
                #                 (epoch - 1) * len(train_loader.dataset) + batch_idx * config.batch_size)
                print(
                    f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100 * batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.data[0]:.6f}')

        # if epoch % config.checkpoint_interval == 0:
        #     torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f'checkpoint_whole3-{epoch}.pth'))

        val(epoch, writer, test_loader, model)
    writer.close()


def val(epoch, writer, test_loader, model):
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
    model.train()


def test(**kwargs):
    config.parse(kwargs)
    test_loader = DataLoader(
        WholeFaceDatasets(config.whole_csv_test, config.whole_imgs_base_dir, train=False, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])), batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )

    model = Net()
    if config.using_pretrain_model:
        model.load_state_dict(torch.load(config.pretrain_model_path))
    model.cuda()
    model.eval()

    test_loss = 0
    tables = None
    for data, target in test_loader:
        target = target.type(torch.FloatTensor)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # print(output)
        loss = F.l1_loss(output, target).cpu()

        a = output.cpu().data.numpy()
        b = np.reshape(target.cpu().data.numpy(), (-1, 1))
        pair = np.hstack((a, b))
        if tables is None:
            tables = pair
        else:
            tables = np.vstack((tables, pair))

        test_loss += loss.data[0] * config.batch_size
    test_loss /= len(test_loader.dataset)
    # print(tables)
    np.save('predicted_and_real.npy', tables)

    print(f'Testing Accuracy is {test_loss}')


if __name__ == '__main__':
    fire.Fire()
