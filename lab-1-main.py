'''Train CIFAR10 with PyTorch.'''

import torchvision
import os
import argparse
from models import *
import numpy as np
import math, torch
import torch.nn as nn
from tqdm import trange
import torch.optim as optim
from utils import progress_bar
from torchsummary import summary
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
from matplotlib import pyplot as plt 
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_validation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset_full = torchvision.datasets.CIFAR10(
    root='/opt/img/effdl-cifar10/',
    train=True,
    download=True,
    transform=transform_train
)

val_size = int(0.1 * len(trainset_full))   # 10%
train_size = len(trainset_full) - val_size

train_subset, val_subset = random_split(
    trainset_full,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

trainloader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
validationloader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)
print(f"Train size: {len(train_subset)} | Validation size: {len(val_subset)}")

validationset = torchvision.datasets.CIFAR10(
    root='/opt/img/effdl-cifar10/', train=False, download=True, transform=transform_validation)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/opt/img/effdl-cifar10/',
    train=False,
    download=True,
    transform=transform_validation
)

testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
print(f"Test size: {len(testset)}")



'''
## number of target samples for the final dataset
num_train_examples = len(train_subset)
num_samples_subset = 15000
## We set a seed manually so as to reproduce the results easily
seed  = 2147483647
## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices
c10train_subset = torch.utils.data.Subset(train_subset,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(train_subset)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")
'''



classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


### Let's do a figure for each batch
f = plt.figure(figsize=(10,10))

for i,(data,target) in enumerate(trainloader):
    data = (data.numpy())
    print(data.shape)
    plt.subplot(2,2,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))

    break

f.savefig('train_DA.png')


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    train_acc = 100. * correct / total
    train_loss /= len(trainloader)
    
    save_checkpoint(net, optimizer, epoch, best_acc= train_acc, loss=train_loss, path='./checkpoint/train_loss_epoch_{}.pth')
    
    if train_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': train_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = validation_acc
        save_checkpoint(net, optimizer, epoch, best_acc, path='./checkpoint/train_ckpt.pth')

    return train_loss, train_acc    
    
# Validation
def validation(epoch):
    global best_acc
    net.eval()
    validation_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validationloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validationloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (validation_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    validation_acc = 100.*correct/total
    #save_checkpoint(loss=validation_loss, epoch=epoch, path='./checkpoint/loss_epoch_{}.pth')
    save_checkpoint(net, optimizer, epoch, best_acc=validation_acc, loss=validation_loss, path='./checkpoint/loss_epoch_{}.pth')
    
    if validation_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': validation_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = validation_acc
        save_checkpoint(net, optimizer, epoch, best_acc, path='./checkpoint/ckpt.pth')
        
    return validation_loss, validation_acc


def test_final():
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader)
    test_acc = 100. * correct / total

    print("\n====== FINAL TEST RESULT ======")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("================================")

    return test_loss, test_acc

    
def save_checkpoint(model, optimizer, epoch, best_acc, loss=None, acc=None, path='./checkpoint/ckpt.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'loss': loss,
        'acc': acc
    }, path)
    print(f"Checkpoint salvo em {path}")



hparams = {
    "model": "ResNet18",
    "lr_init": args.lr,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "CosineAnnealingLR",
    "T_max": 200,
    "batch_size": 32,
    #"seed": seed,
}

history = {
    "epoch": [],
    "train_loss": [],
    "validation_loss": [],
    "test_loss": [],
    
    "train_acc": [],
    "validation_acc": [],
    "test_acc": [],
    "lr": [],
}


for epoch in range(start_epoch, start_epoch+5):

    train_loss, train_acc = train(epoch)
    #validation_loss, validation_acc = validation(epoch, hparams, history)
    validation_loss, validation_acc = validation(epoch)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["train_acc"].append(train_acc)
    history["validation_acc"].append(validation_acc)
    history["lr"].append(optimizer.param_groups[0]['lr'])

    if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
    torch.save({"hparams": hparams, "history": history}, "./checkpoint/training_log.pth")
    scheduler.step()

test_loss, test_acc = test_final()

'''

import torch
from matplotlib import pyplot as plt

def plot_training_results(log_path='./checkpoint/training_log.pth'):
    # Carrega os dados salvos
    checkpoint = torch.load(log_path)
    history = checkpoint['history']

    # Recupera métricas
    epochs = history['epoch']
    train_acc = history['train_acc']
    val_acc = history['validation_acc']
    test_acc = [0]*(len(epochs)-1) + [checkpoint.get('test_acc', 0)]  # se quiser adicionar test_acc no final
    train_loss = history['train_loss']
    val_loss = history['validation_loss']
    test_loss = [0]*(len(epochs)-1) + [checkpoint.get('test_loss', 0)]

    # Plot Accuracy x Epoch
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_acc, label='Train Acc', marker='o')
    plt.plot(epochs, val_acc, label='Validation Acc', marker='o')
    if any(test_acc):
        plt.plot(epochs, test_acc, label='Test Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy x Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('./checkpoint/acc_vs_epoch.png')
    plt.show()

    # Plot Loss x Epoch
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_loss, label='Train loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation loss', marker='o')
    if any(test_loss):
        plt.plot(epochs, test_loss, label='Test loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss x Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('./checkpoint/loss_vs_epoch.png')
    plt.show()

# Chama a função
plot_training_results()
'''