'''Train CIFAR10 with PyTorch.'''
import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='ResNet18', type=str, help='Model: ResNet18 ou VGG19')
parser.add_argument('--epochs', default=5, type=int, help='Número de épocas para o teste')
parser.add_argument('--dry_run', action='store_true', help='Se marcado, apenas simula o treino')
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
## We set a seed manually so as to reproduce the results easily
seed  = 2147483647
best_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


full_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

full_valset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_test)  # sem augment

# Split indices
num_train = len(full_trainset)
indices = torch.randperm(num_train, generator=torch.Generator().manual_seed(seed))
train_size = int(0.9 * num_train)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Subsets
train_subset = torch.utils.data.Subset(full_trainset, train_indices)
validation_subset = torch.utils.data.Subset(full_valset, val_indices)

# Loaders
trainloader = torch.utils.data.DataLoader(
    train_subset, batch_size=32, shuffle=True, num_workers=2)

validationloader = torch.utils.data.DataLoader(
    validation_subset, batch_size=32, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
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
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net_name = net.module.__class__.__name__ if isinstance(net, torch.nn.DataParallel) else net.__class__.__name__
checkpoint_dir = f'./checkpoint{net_name}-Validation'
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

    
if args.resume:
    assert os.path.isdir(checkpoint_dir), f'Error: checkpoint directory {checkpoint_dir} not found!'
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best.pth')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint.get('best_acc', 0)  # caso queira salvar best_acc também
    start_epoch = checkpoint.get('epoch', 0)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
T_max=200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

# Training
def train(epoch):
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

    train_loss_epoch = train_loss / len(trainloader)
    train_acc_epoch = 100. * correct / total

    return train_loss_epoch, train_acc_epoch

def validation(epoch, hparams, history):
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

    validation_loss_epoch = validation_loss / len(validationloader)
    validation_acc_epoch = 100. * correct / total
    
    # Save checkpoint.
    # acc = 100.*correct/total
        
    if validation_acc_epoch > best_acc:
        best_acc = validation_acc_epoch  # atualiza primeiro
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'hparams': hparams,
            'history': history,
            'best_acc': best_acc,
            'epoch': epoch
        }
        torch.save(state, os.path.join(checkpoint_dir, 'ckpt_best.pth'))

    return validation_loss_epoch, validation_acc_epoch


def test(epoch, hparams, history):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    test_loss_epoch = test_loss / len(testloader)
    test_acc_epoch = 100. * correct / total

    return test_loss_epoch, test_acc_epoch

hparams = {
    "model": net,
    "lr_init": args.lr,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "CosineAnnealingLR",
    "T_max": T_max,
    "batch_size": 32,
    "seed": seed,
}

history = {
    "epoch": [],
    "train_loss": [],
    "validation_loss": [],
    
    "train_acc": [],
    "validation_acc": [],
    "lr": [],
}

wandb.init(
    entity="nathalia-rezende-coelho-imt-atlantique",
    project="cifar10-comparison",
    config={
        "learning_rate": args.lr,
        "architecture": args.model,
        "dataset": "CIFAR-10",
        "epochs": args.epochs,
    },
    name=f"test-{args.model}-lr{args.lr}-DA",
    resume="allow"
)


for epoch in range(start_epoch, start_epoch+175):

    train_loss, train_acc = train(epoch)
    validation_loss, validation_acc = validation(epoch, hparams, history)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["train_acc"].append(train_acc)
    history["validation_acc"].append(validation_acc)
    history["lr"].append(optimizer.param_groups[0]['lr'])

    torch.save({"hparams": hparams, "history": history}, os.path.join(checkpoint_dir, "training_log.pth"))
    wandb.save('checkpoint/training_log.pth')
    
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "validation/loss": validation_loss,
        "train/acc": train_acc,
        "validation/acc": validation_acc,
        "lr": optimizer.param_groups[0]['lr'],
    })
    
    
    scheduler.step()

# ===== Load best validation model =====
print('\n==> Loading best validation model for testing...')
checkpoint = torch.load(os.path.join(checkpoint_dir, 'ckpt_best.pth'))
net.load_state_dict(checkpoint['net'])


# ===== Final Test =====
test_loss, test_acc = test(epoch, hparams, history)

wandb.log({
        "epoch": epoch,
        "test/loss": test_loss,
        "test/acc": test_acc,
    })
print(f'\nFinal Test Loss: {test_loss:.4f}')
print(f'Final Test Accuracy: {test_acc:.2f}%')

wandb.finish()