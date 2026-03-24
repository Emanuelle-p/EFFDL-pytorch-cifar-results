'''Train CIFAR10 with PyTorch.'''
import os
import copy
import wandb
import torch
import numpy as np
import torch.nn as nn
import torchvision
import argparse
from models import *
import torch.optim as optim
from utils import progress_bar
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from collections import OrderedDict
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

from utils import train, test, load_checkpoint
from utils import global_pruning, check_sparsity, retrain_after_pruning, prune_filters_structured, apply_thinet
from utils import apply_binaryconnect, apply_xor_quantization
from utils import NTCE_KD_Loss

from utils import cutmix_batch



# Arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='ResNet20', type=str, help='Model: ResNet18')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--T_max', default=200, type=int, help='Number of epochs for CosineAnnealingLR scheduler')
parser.add_argument('--dry_run', action='store_true', help='Se marcado, apenas simula o treino')
#parser.add_argument('--dropout', default=0, type=float, help='Dropout probability')
#parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
#parser.add_argument('--patience', default=10, type=int, help='Epochs to wait before stopping')
parser.add_argument('--cutmix', action='store_true', help='Apply CutMix data augmentation during training')
parser.add_argument('--cutmix_alpha', default=1.0, type=float, help='alpha da Beta(α,α) do CutMix (paper usa 1.0)')

parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
#parser.add_argument('--basic_train', action='store_true', help='Standard training without pruning or quantization.')
parser.add_argument('--train', action='store_true', help='Run training')
parser.add_argument('--factorization', type=str, default="standard", choices=["standard", "depthwise", "spatial", "grouped"], help="Type of convolution factorization")

parser.add_argument('--eval_fp16', action='store_true', help='Evaluate trained model using FP16 precision (half precision inference).')
parser.add_argument('--train_with_binary_connect', action='store_true', help='Train model using BinaryConnect')
parser.add_argument('--train_with_xor', action='store_true', help='Train model using XNOR-Net (binary weights and activations)')
parser.add_argument('--eval_global_pruning', action='store_true', help='Evaluate model under different global unstructured pruning ratios (no retraining).')
parser.add_argument('--retrain_global_pruning', action='store_true', help='Apply global pruning and retrain model to recover accuracy.')
parser.add_argument('--gradual_structured_prune', action='store_true', help='Apply gradual structured filter pruning with retraining for each pruning ratio.')
parser.add_argument('--ps_pu_prune_eval_fp16', action='store_true', help='Apply both unstructured and structured pruning sequentially, with retraining in between, and evaluate on fp16')
parser.add_argument('--thinet', action='store_true', help='Apply ThiNet pruning based on feature map statistics')

# NTCE-KD arguments
parser.add_argument('--use_ntce_kd', action='store_true', help='Enable NTCE-KD loss during student training.')
parser.add_argument('--teacher_checkpoint', type=str, default=None, help='Path to teacher checkpoint (.pth) used by NTCE-KD.')
parser.add_argument('--teacher_model', type=str, default='ResNet20', help='Teacher model class name available in models package.')
parser.add_argument('--ntce_temperature', type=float, default=4.0, help='Temperature used by NTCE-KD.')
parser.add_argument('--ntce_alpha', type=float, default=1.0, help='CrossEntropy weight in NTCE-KD.')
parser.add_argument('--ntce_beta', type=float, default=8.0, help='Distillation (MKL) weight in NTCE-KD.')

args = parser.parse_args()


def main():
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed  = 2147483647 #reproduce the results easily
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data augmentation
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        #RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2)

    # Classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Configurations
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    #net = ResNet18(conv_type=args.factorization, width_mult=0.75).to(device)
    net = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)
    print(f"Using convolution type: {args.factorization}")

    # NTCE-KD setup
    teacher_model = None
    if args.use_ntce_kd:
        if not args.teacher_checkpoint:
            raise ValueError("--teacher_checkpoint is required when --use_ntce_kd is enabled.")
        if not os.path.isfile(args.teacher_checkpoint):
            raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

        if args.teacher_model not in globals() or not callable(globals()[args.teacher_model]):
            raise ValueError(f"Teacher model '{args.teacher_model}' not found in models package.")

        print(f"==> Loading teacher model ({args.teacher_model}) from checkpoint: {args.teacher_checkpoint}")
        teacher_model = globals()[args.teacher_model]().to(device)
        load_checkpoint(teacher_model, args.teacher_checkpoint)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

        criterion = NTCE_KD_Loss(
            temperature=args.ntce_temperature,
            alpha=args.ntce_alpha,
            beta=args.ntce_beta,
        )
        print(
            f"Using NTCE-KD Loss (T={args.ntce_temperature}, alpha={args.ntce_alpha}, beta={args.ntce_beta})"
        )
    
    
    hparams = {
        "model": net,
        "lr_init": args.lr,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "scheduler": "CosineAnnealingLR",
        "T_max": args.T_max,
        "batch_size": 32
    }
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
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
            "use_ntce_kd": args.use_ntce_kd,
            "teacher_model": args.teacher_model if args.use_ntce_kd else None,
            "teacher_checkpoint": args.teacher_checkpoint if args.use_ntce_kd else None,
            "ntce_temperature": args.ntce_temperature if args.use_ntce_kd else None,
            "ntce_alpha": args.ntce_alpha if args.use_ntce_kd else None,
            "ntce_beta": args.ntce_beta if args.use_ntce_kd else None,
        },
        #name=f"{args.model}-basic_model_75widthscale", #-{args.factorization}",
        #name=f"{args.model}-basic_model-factorization_{args.factorization}",
        #name="ResNet18-basic_model-factorization_depthwise",
        #name="ResNet18-75widthscale-depthwise",
        name=f"{args.model}-basic_model",
        #name="checkpoint_ResNet20",
        resume="allow"
    )
        
    # Define checkpoint folder based on the W&B run's name
    checkpoint_name = f"checkpoint-{wandb.run.name}"
    if not os.path.isdir(checkpoint_name):
        os.mkdir(checkpoint_name)
        
    #checkpoint_model = os.path.join(checkpoint_name, 'training_XOR_log.pth') #trained_model.pth #/users/local/pytorch-cifar-backup/checkpoint-ResNet18-global_pruning_retrain/trained_model.pth

    if args.resume:
        checkpoint_path = os.path.join(checkpoint_name, 'ckpt.pth')
        assert os.path.isdir(os.path.dirname(checkpoint_path)), 'Error: checkpoint folder not found!'
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove "module."
            else:
                new_state_dict[k] = v

        net.load_state_dict(new_state_dict)
        start_epoch = checkpoint['epoch']
        
    best_val_loss = float('inf')
    #patience_counter = 0
    
        
    if args.train:
        # python main.py --train --factorization standard  >> BASIC_MODEL
        # python main.py --train --factorization spatial
        # python main.py --train --factorization depthwise
        # python main.py --train --factorization grouped
        start_epoch = 0
        best_acc = 0.0

        if args.cutmix:
            print("Using CutMix data augmentation")
            for epoch in range(start_epoch, start_epoch + args.epochs):
                net.train()
                total_loss, total_correct, total_samples = 0, 0, 0

                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    inputs, targets_a, targets_b, lam = cutmix_batch(
                        inputs, targets, alpha=args.cutmix_alpha
                    )

                    optimizer.zero_grad()
                    outputs = net(inputs)

                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_logits = teacher_model(inputs)
                        loss_a = criterion(outputs, teacher_logits, targets_a)
                        loss_b = criterion(outputs, teacher_logits, targets_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * targets.size(0)
                    _, predicted = outputs.max(1)
                    total_samples += targets.size(0)

                    total_correct += (
                        lam * predicted.eq(targets_a).sum().item() +
                        (1 - lam) * predicted.eq(targets_b).sum().item()
                    )

                train_loss = total_loss / total_samples
                train_acc = 100. * total_correct / total_samples

                test_loss, test_acc, best_acc = test(net, best_acc, 0, testloader, criterion, device, checkpoint_name, hparams, history)

                scheduler.step()
                
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["test_loss"].append(test_loss)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["lr"].append(optimizer.param_groups[0]['lr'])

                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'history': history,
                }
                torch.save(state, os.path.join(checkpoint_name, 'basictrain_cutmix.pth'))
                wandb.save(os.path.join(checkpoint_name, 'basictrain_cutmix.pth'))
                
                wandb.log({
                    "epoch": epoch,
                    f"train/loss": train_loss,
                    f"train/acc": train_acc,
                    f"test/loss": test_loss,
                    f"test/acc": test_acc,
                    f"lr": optimizer.param_groups[0]['lr']
                })
        
        else:
            for epoch in range(start_epoch, start_epoch + args.epochs):
                train_loss, train_acc = train(
                    net,
                    epoch,
                    trainloader,
                    optimizer,
                    criterion,
                    device,
                    teacher_model=teacher_model,
                )
                test_loss, test_acc, best_acc = test(net, best_acc, 0, testloader, criterion, device, checkpoint_name, hparams, history)
                scheduler.step()

                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["test_loss"].append(test_loss)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["lr"].append(optimizer.param_groups[0]['lr'])

                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'history': history,
                }
                torch.save(state, os.path.join(checkpoint_name, 'ckpt.pth'))
                wandb.save(os.path.join(checkpoint_name, 'ckpt.pth'))
                
                wandb.log({
                    "epoch": epoch,
                    f"train/loss": train_loss,
                    f"train/acc": train_acc,
                    f"test/loss": test_loss,
                    f"test/acc": test_acc,
                    f"lr": optimizer.param_groups[0]['lr']
                })
            
        print("\n==> Finished Training")
        #wandb.save('checkpoint/training_75widthscale.pth')
        #wandb.save('checkpoint/training_cutmix_log.pth')



    # --------------------------------------
    #            QUANTIZATION
    # --------------------------------------
    
    if args.eval_fp16:

        net = ResNet18()
        load_checkpoint(net, "checkpoint-ResNet18-global_pruning_retrain/trained_model.pth")
        net = net.to(device)

        net.eval()
        net.half() 

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device).half(), targets.to(device)

                # Forward
                outputs = net(inputs)
                # Loss
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                # Accuracy
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        # Final metrics
        avg_loss = total_loss / total_samples
        avg_acc = 100. * total_correct / total_samples
        print(f"FP16 Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.2f}%")

        # Type checking
        print("First weight type:", next(net.parameters()).dtype)  
        sample_inputs, _ = next(iter(testloader))
        print("Batch input type:", sample_inputs.to(device).half().dtype)  # torch.float16

        wandb.log({
            "fp16/test_loss": avg_loss,
            "fp16/test_acc": avg_acc
        })

    # Binary Connect 
    if args.train_with_binary_connect:
        
        net = ResNet18()
        #net = ResNet18(conv_type=args.factorization, width_mult=1.0).to(device)
        net = apply_binaryconnect(net, clip_value=1.0, first_last_fp32=True)
        print(net)
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)
            
            
        for epoch in range(start_epoch, start_epoch + args.epochs):
            net.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
            
                net.binarization()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                net.restore()
                optimizer.step()
                net.clip()
                
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
            
            train_loss = total_loss / total_samples
            train_acc = 100. * total_correct / total_samples

            # Avaliação no teste
            test_loss, test_acc, best_acc = test(
                net, best_acc, epoch, testloader, criterion, device, checkpoint_name, hparams, history
            )
            scheduler.step()
            
            # Log para W&B
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "lr": optimizer.param_groups[0]['lr']
            })
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")
            
            if epoch % 10 == 0:
                wandb.save(f'checkpoint/trainingBinary_epoch{epoch}.pth')
            
        print("\n==> Finished Training")
        wandb.save('checkpoint/trainingBinary_log.pth')
        
    # XORWrapper
    if args.train_with_xor:
        
        net = ResNet18()
        #net = ResNet18(conv_type=args.factorization, width_mult=1.0).to(device)
        net = apply_xor_quantization(net, first_last_fp32=True)
        print(net)
        net = net.to(device)
        start_epoch = 0
        best_acc = 0.0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)
        
        for epoch in range(start_epoch, start_epoch + args.epochs):
            net.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

            train_loss = total_loss / total_samples
            train_acc = 100. * total_correct / total_samples

            net.eval()
            test_loss, test_acc, best_acc = test(
                net, best_acc, epoch, testloader, criterion, device, checkpoint_name, hparams, history
            )
            scheduler.step()

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "lr": optimizer.param_groups[0]['lr']
            })
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

            if epoch % 10 == 0:
                wandb.save(f'{checkpoint_name}/trainingXOR_epoch{epoch}.pth')

        print("\n==> Finished XOR Training")
        wandb.save(f'{checkpoint_name}/trainingXOR_log.pth')
        
    # --------------------------------------
    #               PRUNING
    # --------------------------------------
    if args.eval_global_pruning:
        
        pruning_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
        #checkpoint_model = os.path.join(checkpoint_name, 'trained_model.pth')
        checkpoint_model = os.path.join(checkpoint_name, 'trainingBinary_log.pth')
        
       
        net = ResNet18()
        #net = ResNet18(conv_type=args.factorization, width_mult=1.0).to(device)
        load_checkpoint(net, "checkpoint-ResNet18-global_pruning_retrain/trained_model.pth") #"checkpoint-ResNet18-trainBinary/trainingBinary_log.pth"
        net = net.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)

        #optional
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)

        best_acc = 0

        pruning_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
        for ratio in pruning_ratios:
            net_prune = copy.deepcopy(net)
            net_prune = global_pruning(net_prune, amount=ratio)
            
           
            test_loss, test_acc, best_acc = test(
                net_prune, best_acc, 0, testloader, criterion, device, checkpoint_name, hparams, history
            )

            wandb.log({
                "global_pruning/accuracy": test_acc,
                "global_pruning/loss": test_loss,
            }, step=int(ratio*100))
            print(f"Global Pruning Ratio: {ratio*100:.0f}%, Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")


    if args.retrain_global_pruning:
        
        net = ResNet18()
        load_checkpoint(net, "checkpoint-ResNet18-global_pruning_retrain/trained_model.pth") #"checkpoint-ResNet18-trainBinary/trainingBinary_log.pth"
        net = net.to(device)
        net.eval()  
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)
        
        test_loss, test_acc, _ = test(net, best_acc=0, epoch=0, testloader=testloader, criterion=criterion, device=device, checkpoint_name=checkpoint_name)
        print(f"Acurácia final do checkpoint: {test_acc:.2f}%")

        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)

        best_acc = 0
        net = global_pruning(net, amount=0.4)

        model, history_retrain, sparsity = retrain_after_pruning(
            model=net,
            checkpoint_name=checkpoint_name,
            checkpoint_path=os.path.join(checkpoint_name, 'trainingBinary_log.pth'),
            trainloader=trainloader,
            testloader=testloader,
            hparams=hparams,
            device=device,
            amount=0.4,            # 40% pruning
            retrain_epochs=15,
            scheduler=scheduler,
            retrain_lr=0.01,
            start_epoch=199,
            wandb_log=wandb.log
        )


    if args.gradual_structured_prune:
        # Gradually prune and retrain accross layers        
        
        net = ResNet18()
        load_checkpoint(net, "checkpoint-ResNet18-trainBinary200/trainBinary200.pth")
        net = net.to(device)
        net.eval()  
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)
        
        test_loss, test_acc, _ = test(net, best_acc=0, epoch=0, testloader=testloader, criterion=criterion, device=device, checkpoint_name=checkpoint_name)
        print(f"Acurácia final do checkpoint: {test_acc:.2f}%")

        best_acc = 0

        #structured_pruning_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
        structured_pruning_ratios = [0.4, 0.7]
        retrain_epochs_per_step = 10

        for ratio in structured_pruning_ratios:    
            net_prune = copy.deepcopy(net)
            print(f"\n==> Structured Pruning: removing {int(ratio*100)}% of filters per conv layer")
            
            net_prune = prune_filters_structured(net_prune, amount=ratio)
            test_loss, test_acc, _ = test(net_prune, best_acc, 0, testloader, criterion, device, checkpoint_name)
            print(f"Before retrain - Test Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
            
            # Retrain 
            model, history_retrain, sparsity_dict = retrain_after_pruning(
                model=net_prune,
                checkpoint_name=checkpoint_name,
                checkpoint_path=checkpoint_model,
                trainloader=trainloader,
                testloader=testloader,
                hparams=hparams,
                device=device,
                amount=0.0, 
                retrain_epochs=retrain_epochs_per_step,
                scheduler=scheduler,
                retrain_lr=0.01,
                start_epoch=199,
                wandb_log=lambda metrics, step=None: wandb.log(
                    {f"structured_retrain/{k}": v for k,v in metrics.items()}
                )
            )
            
            wandb.log({
                "structured_pruning/test_acc_after_retrain": history_retrain["test_acc"][-1],
                "structured_pruning/test_loss_after_retrain": history_retrain["test_loss"][-1],
                "structured_pruning/sparsity": sparsity_dict['global']
            }, commit=False)

            torch.save(model.state_dict(), os.path.join(checkpoint_name, f'BinaryPruning_{ratio}.pth'))
            wandb.save(os.path.join(checkpoint_name, f'BinaryPruning_{ratio}.pth'))

    if args.ps_pu_prune_eval_fp16:    
        #checkpoint_model = os.path.join(checkpoint_name, 'trained_model.pth')
        checkpoint_model = os.path.join(checkpoint_name, 'ckpt.pth')
        #checkpoint_model = os.path.join(checkpoint_name, 'cifar10_resnet20-4118986f.pt')
        
    
        #net = ResNet18(conv_type="depthwise", width_mult=0.75).to(device)
        net = ResNet20().to(device)
        #load_checkpoint(net, "checkpoint-ResNet18-global_pruning_retrain/trained_model.pth") #"checkpoint-ResNet18-trainBinary/trainingBinary_log.pth"
        #load_checkpoint(net, "checkpoint-ResNet18-basic_model_75widthscale/ckpt.pth")
        #load_checkpoint(net, "checkpoint-ResNet18-basic_model-factorization_depthwise/ckpt.pth")
        #load_checkpoint(net, "checkpoint-ResNet18-75widthscale-depthwise/ckpt.pth")
        #load_checkpoint(net, "checkpoint_ResNet20/cifar10_resnet20-4118986f.pt")
        load_checkpoint(net, "checkpoint-ResNet20-basic_model/ckpt.pth")
        net.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_correct = 0
        total_samples = 0
        
        # Just to check the accuracy of the loaded checkpoint before pruning and quantization
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        accuracy = 100. * total_correct / total_samples
        print(f"Checkpoint Accuracy (ps_pu_prune_eval_fp16): {accuracy:.2f}%")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)
        
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)

        # Unstructured pruning
        best_acc = 0
        unstructured_ratio = 0.7
        net_prune = copy.deepcopy(net)
        print(f"\n==> Unstructured Pruning: removing {int(unstructured_ratio*100)}% of weights")
        net_prune = global_pruning(net_prune, amount=unstructured_ratio)
        
        test_loss, test_acc, best_acc = test(
            net_prune, best_acc, 0, testloader, criterion, device, checkpoint_name, hparams, history
        )
        
        model, history_retrain, sparsity = retrain_after_pruning(
            model=net,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_model,
            trainloader=trainloader,
            testloader=testloader,
            hparams=hparams,
            device=device,
            amount=unstructured_ratio,
            retrain_epochs=15,
            scheduler=scheduler,
            retrain_lr=0.01,
            start_epoch=199,
            wandb_log=wandb.log
        )

        
        # Structured pruning
        structured_pruning_ratio = 0.4
        retrain_epochs_per_step = 10

        net_prune = copy.deepcopy(model)
        print(f"\n==> Structured Pruning: removing {int(structured_pruning_ratio*100)}% of filters per conv layer")
        
        net_prune = prune_filters_structured(net_prune, amount=structured_pruning_ratio)
     
        test_loss, test_acc, _ = test(net_prune, best_acc, 0, testloader, criterion, device, checkpoint_name)
        print(f"Before retrain - Test Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
        
    
        model, history_retrain, sparsity_dict = retrain_after_pruning(
            model=net_prune,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_model,
            trainloader=trainloader,
            testloader=testloader,
            hparams=hparams,
            device=device,
            amount=0.0,  
            retrain_epochs=retrain_epochs_per_step,
            scheduler=scheduler,
            retrain_lr=0.01,
            start_epoch=199,
            wandb_log=lambda metrics, step=None: wandb.log(
                {f"structured_retrain/{k}": v for k,v in metrics.items()}
            )
        )
        
        # Apply quantization (FP16) after pruning
        model.eval()
        model.half()  # converte para FP16

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device).half(), targets.to(device)

                # Forward
                outputs = model(inputs)
                # Loss
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                # Acurácia
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = 100. * total_correct / total_samples
        print(f"FP16 Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.2f}%")

        print("Tipo do primeiro peso:", next(model.parameters()).dtype)  # deve ser torch.float16
        sample_inputs, _ = next(iter(testloader))
        print("Tipo de inputs do batch:", sample_inputs.to(device).half().dtype)  # torch.float16

        wandb.log({
            "fp16/test_loss": avg_loss,
            "fp16/test_acc": avg_acc
        })    
           


    if args.thinet:
        net = ResNet18()
        load_checkpoint(net, "checkpoint-ResNet18-global_pruning_retrain/trained_model.pth")
        net = net.to(device)
        net.eval()  
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max)
        
        test_loss, test_acc, _ = test(net, best_acc=0, epoch=0, testloader=testloader, criterion=criterion, device=device, checkpoint_name=checkpoint_name)
        print(f"Acurácia final do checkpoint: {test_acc:.2f}%")


        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            
        best_acc = 0

        pruning_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
        checkpoint_model = os.path.join(checkpoint_name, 'trained_model.pth')

        for ratio in pruning_ratios:
            net_prune = copy.deepcopy(net)
            print(f"\n==> ThiNet Pruning: {int(ratio*100)}% filters")

            # Apply ThiNet
            net_prune, sparsity_dict = apply_thinet(
                net_prune,
                trainloader,
                device,
                pruning_ratio=ratio,
                num_batches=10
            )

            # Evaluate before retrain
            test_loss, test_acc, _ = test(
                net_prune, best_acc, 0, testloader, criterion, device, checkpoint_name
            )

            # Retrain    
            model, history_retrain, sparsity_dict = retrain_after_pruning(
                model=net_prune,
                checkpoint_name=checkpoint_name,
                checkpoint_path=checkpoint_model,
                trainloader=trainloader,
                testloader=testloader,
                hparams=hparams,
                device=device,
                amount=0.0, 
                retrain_epochs=10,
                scheduler=scheduler,
                retrain_lr=0.01,
                start_epoch=199,
                wandb_log=lambda metrics: wandb.log(
                    {f"thinet_retrain/{k}": v for k,v in metrics.items()}
                )
            )

            wandb.log({
                "thinet/test_acc_after_retrain": history_retrain["test_acc"][-1],
                "thinet/test_loss_after_retrain": history_retrain["test_loss"][-1],
                "thinet/sparsity_after_retrain": sparsity_dict["global"]
            }, commit=False)


    wandb.finish()


'''
if args.dropout > 0:
    for name, module in reversed(list(net.named_modules())):
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            # substitui a última Linear por Dropout + Linear
            parent = net
            components = name.split('.')
            for comp in components[:-1]:
                parent = getattr(parent, comp)
                
            setattr(parent, components[-1],
                    nn.Sequential(
                        nn.Dropout(p=args.dropout),
                        nn.Linear(in_features, out_features)
                    ))
            break

# Early Stopping was desactivaded because it was observed that with patience=10 i.e. the training finished very early, even before the val courves stabilize
if args.early_stopping:
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        patience_counter = 0
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(net.state_dict(), './checkpoint/best_model_earlystop.pth')
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            net.load_state_dict(torch.load('./checkpoint/best_model_earlystop.pth'))
            break
'''

if __name__ == "__main__":
    main()