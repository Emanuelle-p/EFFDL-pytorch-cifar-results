'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import numpy
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional


class NTCE_KD_Loss(nn.Module):
    """
    Non-Target-Class-Enhanced Knowledge Distillation loss.

    This criterion combines standard CrossEntropy loss with Magnitude-Enhanced KL (MKL)
    computed from teacher and student logits. It focuses on non-target class behavior by
    applying target-logit shrinkage before KL computation.

    Args:
        temperature (float): Temperature used to soften probabilities.
        alpha (float): Weight applied to CrossEntropy loss.
        beta (float): Weight applied to MKL distillation loss.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 1.0, beta: float = 8.0):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Shrinkage coefficients used in the multi-branch MKL computation.
        self.register_buffer(
            "shrinkage_coeffs",
            torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
        )

        # Standard CE term for hard labels.
        self.ce_loss = nn.CrossEntropyLoss()

    def _apply_target_shrinkage(self, logits: torch.Tensor, targets: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply shrinkage only to the target-class logit for each sample."""
        modified_logits = logits.clone()
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        modified_logits[batch_indices, targets] = modified_logits[batch_indices, targets] - delta
        return modified_logits

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute NTCE-KD loss.

        Supports both call styles:
        - CE mode: criterion(student_logits, targets)
        - KD mode: criterion(student_logits, teacher_logits, targets)
        """
        # Backward-compatible CE style: criterion(logits, targets)
        if targets is None and teacher_logits is not None and teacher_logits.dim() == 1:
            targets = teacher_logits
            teacher_logits = None

        if targets is None:
            raise ValueError("NTCE_KD_Loss requires targets.")

        targets = targets.view(-1).long()
        ce_loss = self.ce_loss(student_logits, targets)

        # In pure evaluation/inference mode, fallback to CE only.
        if teacher_logits is None:
            return self.alpha * ce_loss

        batch_size, num_classes = student_logits.size()
        device = student_logits.device
        batch_indices = torch.arange(batch_size, device=device)
        shrinkage_coeffs = self.shrinkage_coeffs.to(device=device, dtype=student_logits.dtype)

        # Teacher is used as a fixed target distribution.
        teacher_logits = teacher_logits.detach()

        # ===== Magnitude-Enhanced KL (MKL) =====
        # Build target-logit shrinkage from teacher confidence gap.
        teacher_target_logits = teacher_logits[batch_indices, targets]
        target_mask = F.one_hot(targets, num_classes=num_classes).bool()
        teacher_non_target_max = teacher_logits.masked_fill(target_mask, float("-inf")).max(dim=1).values
        shrinkage_base = teacher_target_logits - teacher_non_target_max

        kl_losses = []
        for lambda_m in shrinkage_coeffs:
            delta = lambda_m * shrinkage_base
            student_logits_modified = self._apply_target_shrinkage(student_logits, targets, delta)
            teacher_logits_modified = self._apply_target_shrinkage(teacher_logits, targets, delta)

            log_probs_student = F.log_softmax(student_logits_modified / self.temperature, dim=1)
            probs_teacher = F.softmax(teacher_logits_modified / self.temperature, dim=1)

            kl_div = F.kl_div(log_probs_student, probs_teacher, reduction="batchmean")
            kl_losses.append(kl_div)

        mkl_loss = torch.stack(kl_losses).mean() * (self.temperature ** 2)
        total_loss = self.alpha * ce_loss + self.beta * mkl_loss
        return total_loss

# --------------------------------------
#           FROM GIT CLONE
# --------------------------------------

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
import shutil
term_width = shutil.get_terminal_size((80, 20)).columns
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def load_checkpoint(model, path):
    state_dict = torch.load(path)
    state_dict = state_dict['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

# --------------------------------------
#               TRAINING
# --------------------------------------
def train(net, epoch, trainloader, optimizer, criterion, device, teacher_model=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    if teacher_model is not None:
        teacher_model.eval()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #inputs = inputs.to(device).half()
        #targets = targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #loss = criterion(outputs.float(), targets) # converts the output back to float32
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            loss = criterion(outputs, teacher_logits, targets)
        else:
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


# --------------------------------------
#               TESTING
# --------------------------------------
def test(net, best_acc, epoch, testloader, criterion, device, checkpoint_name, hparams=None, history=None):
    net.eval()
    ##net.half()  # modelo inteiro para FP16
    ##for layer in net.modules():
    ##    if isinstance(layer, nn.BatchNorm2d):
    ##        layer.float()  # BatchNorm em FP32 para estabilidade

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            ##inputs = inputs.to(device).half()
            ##targets = targets.to(device)
            outputs = net(inputs)
            ##loss = criterion(outputs.float(), targets) # converts the output back to float32
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    test_loss_epoch = test_loss / len(testloader)
    test_acc_epoch = 100. * correct / total
        
    if test_acc_epoch > best_acc:
        print("Saving best checkpoint...")
        
        if checkpoint_name is not None:
            state_best = {
                'net': net.state_dict(),
                'hparams': hparams,
                'history': history,
            }
            torch.save(state_best, os.path.join(checkpoint_name, 'ckpt_best.pth'))
        
        best_acc = test_acc_epoch
    
    return test_loss_epoch, test_acc_epoch, best_acc



# --------------------------------------
#            QUANTIZATION
# --------------------------------------
'''
class BinaryConnect():
    def __init__(self, model):
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets += 1

        start_range = 0
        end_range = count_targets - 1
        self.bin_range = numpy.linspace(start_range, end_range, end_range - start_range + 1)\
                            .astype('int').tolist()

        self.num_of_params = len(self.bin_range)
        self.saved_params = []  # save full precision weights
        self.target_modules = []  # modules to be modified
        self.model = model

        # build initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index += 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):
        ### (1) Save current full precision parameters
        self.save_params()

        ### (2) Binarize the weights
        for w in self.target_modules:
            # BinaryConnect: Wb = +1 if w > 0, -1 if w <= 0
            w.data.copy_(w.data.sign())

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        # Clip all parameters to [-1, 1] using Hardtanh
        htan = nn.Hardtanh()
        for w in self.target_modules:
            w.data.copy_(htan(w.data))

    def forward(self, x):
        return self.model(x)
    
    
def binary_connect(net, device):
    # define your model
    netbc = binaryconnect.BC(net) ### use this to prepare your model for binarization
    netbc.model = netbc.model.to(device) # it has to be set for GPU training

    # During training (check the algorithm in the course and in the paper to see the exact sequence of operations)
    netbc.binarization() # This binarizes all weights in the model
    netbc.restore() # This reloads the full precision weights

    # After backprop
    netbc.clip() # Clip the weights
'''

class BinaryConnectWrapper(nn.Module):
    def __init__(self, model: nn.Module, clip_value: float = 1.0, first_last_fp32: bool = False) -> None:
        super().__init__()
        self.model = model
        self.clip_value = float(clip_value)

        quant_layers: list[nn.Module] = [
            module for module in self.model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        if first_last_fp32 and len(quant_layers) > 2:
            quant_layers = quant_layers[1:-1]

        self.target_modules = [module.weight for module in quant_layers]
        self.saved_params = [param.detach().clone() for param in self.target_modules]
        self._weights_binarized = False

        for module in quant_layers:
            module.weight_bit_width = 1
            module.activation_bit_width = 32
            module.quant_scheme = "binaryconnect"

    def save_params(self) -> None:
        with torch.no_grad():
            for index, param in enumerate(self.target_modules):
                saved = self.saved_params[index]
                if (
                    saved.shape != param.shape
                    or saved.device != param.device
                    or saved.dtype != param.dtype
                ):
                    self.saved_params[index] = param.detach().clone()
                else:
                    self.saved_params[index].copy_(param.detach())

    def binarization(self) -> None:
        if self._weights_binarized:
            return
        self.save_params()
        with torch.no_grad():
            for param in self.target_modules:
                param.copy_(torch.where(param >= 0, torch.ones_like(param), -torch.ones_like(param)))
        self._weights_binarized = True

    def restore(self) -> None:
        if not self._weights_binarized:
            return
        with torch.no_grad():
            for index, param in enumerate(self.target_modules):
                param.copy_(self.saved_params[index])
        self._weights_binarized = False

    def clip(self) -> None:
        with torch.no_grad():
            for param in self.target_modules:
                param.clamp_(-self.clip_value, self.clip_value)

    def post_optimizer_step(self) -> None:
        self.clip()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training flow: do not forcibly binarize weights here (trainer may control binarize/restore)
        if self.training and torch.is_grad_enabled():
            return self.model(x)

        # Eval/inference flow: binarize weights for the forward pass, then restore
        self.binarization()
        try:
            return self.model(x)
        finally:
            self.restore()


def apply_binaryconnect(model: nn.Module, clip_value: float, first_last_fp32: bool) -> BinaryConnectWrapper:
    """
    Encapsula um modelo em BinaryConnectWrapper pronto para treino e inferência binária.
    """
    return BinaryConnectWrapper(model=model, clip_value=clip_value, first_last_fp32=first_last_fp32)
        
    
# --------------------------------------
#               PRUNING
# --------------------------------------
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn

def global_pruning(model, amount):
    """
    Aplica pruning L1 global em todos os pesos (Conv2d e Linear) do modelo.
    Mantém as máscaras para retraining.
    Retorna o modelo modificado.
    """
    parameters_to_prune = []
    # Se for DataParallel, pega o modelo real
    model_to_prune = model.module if isinstance(model, torch.nn.DataParallel) else model

    for module in model_to_prune.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # Aplica pruning global L1
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # NÃO remove mask agora, para permitir retraining
    return model


def check_sparsity(model, verbose=True):
    """
    Calcula e imprime a sparsidade de cada camada (Conv2d/Linear) e global do modelo.
    Retorna um dicionário com sparsidade por camada e global.
    """
    sparsity_dict = {}
    total_global = 0
    zero_global = 0

    # Se for DataParallel, pega o modelo real
    model_to_check = model.module if isinstance(model, torch.nn.DataParallel) else model

    for name, module in model_to_check.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight
            total = weight.nelement()
            zero = torch.sum(weight == 0).item()
            sparsity = 100. * zero / total
            sparsity_dict[name] = sparsity
            if verbose:
                print(f"Sparsity in {name}.weight: {sparsity:.2f}%")
            
            total_global += total
            zero_global += zero

    global_sparsity = 100. * zero_global / total_global
    sparsity_dict['global'] = global_sparsity
    if verbose:
        print(f"Global sparsity: {global_sparsity:.2f}%")
    
    return sparsity_dict



def retrain_after_pruning(
    model,
    checkpoint_name,
    checkpoint_path,
    trainloader,
    testloader,
    hparams,
    device,
    amount,
    retrain_epochs,
    scheduler,
    retrain_lr,
    start_epoch,
    wandb_log
):
    """
   - Carrega o modelo do checkpoint
   - Aplica pruning global (L1)
   - Faz retraining para recuperar acurácia
   - Retorna modelo treinado, histórico e sparsidade
    """
    
    # Aplica pruning global (mantendo peso_orig e máscara)
    #global_pruning(model, amount=amount)
    
    history_retrain = {
    "epoch": [],
    "train_loss": [],
    "test_loss": [],
    "train_acc": [],
    "test_acc": [],
    "lr": [],
}

    # Checa sparsidade inicial pós-pruning
    sparsity_dict = check_sparsity(model, verbose=True)
    
    # Otimizador e scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=retrain_lr,
                                momentum=hparams['momentum'],
                                weight_decay=hparams['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=retrain_epochs
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Retraining / fine-tuning
    best_acc = 0
    for epoch in range(retrain_epochs):
        
        train_loss, train_acc = train(model, epoch, trainloader, optimizer, criterion, device)
        test_loss, test_acc, best_acc = test(model, best_acc, epoch, testloader, criterion, device, checkpoint_name=checkpoint_name)
        scheduler.step()

        # Atualiza histórico
        history_retrain["epoch"].append(epoch)
        history_retrain["train_loss"].append(train_loss)
        history_retrain["test_loss"].append(test_loss)
        history_retrain["train_acc"].append(train_acc)
        history_retrain["test_acc"].append(test_acc)
        history_retrain["lr"].append(optimizer.param_groups[0]['lr'])

        # Log no W&B se passado
        if wandb_log:
            wandb_log({
                "retrain/epoch": epoch,
                "retrain/train/loss": train_loss,
                "retrain/train/acc": train_acc,
                "retrain/test/loss": test_loss,
                "retrain/test/acc": test_acc,
                "retrain/lr": optimizer.param_groups[0]['lr'],
                "retrain/pruning_ratio": amount           
            }, step=epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            # opcional: salvar checkpoint do modelo pós-retrain
            torch.save(model.state_dict(), f"retrained_pruned_{int(amount*100)}.pth")

    return model, history_retrain, sparsity_dict


def prune_filters_structured(model, amount):
    """
    Task 3: Pruning Filters (Structured). 
    Prunes entire conv filters based on L1 norm.
    """
    model_to_prune = model.module if isinstance(model, torch.nn.DataParallel) else model
    for module in model_to_prune.modules():
        if isinstance(module, nn.Conv2d):
            # Prune 'amount' percentage of channels (Dimensão 0 → remove filtros inteiros em uma camada convolucional)
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0) 
    return model


# ===========================================
#              ThiNet Pruning
# ===========================================

def apply_thinet(model, trainloader, device, pruning_ratio, num_batches=10):
    """
    ThiNet-like pruning com physical removal:
    - Seleciona filtros baseando-se no impacto na próxima camada
    - Remove fisicamente filtros da camada atual
    - Reconstrói a próxima camada com menos canais de entrada
    - Retorna modelo atualizado e dicionário de sparsity
    """

    model.eval()
    model_to_prune = model.module if isinstance(model, torch.nn.DataParallel) else model
    conv_layers = [m for m in model_to_prune.modules() if isinstance(m, nn.Conv2d)]
    sparsity_dict = {}

    for i in range(len(conv_layers) - 1):
        conv = conv_layers[i]
        next_conv = conv_layers[i + 1]

        # Coletar feature maps
        X_list, Y_list = [], []
        data_iter = iter(trainloader)
        for _ in range(num_batches):
            try:
                inputs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(trainloader)
                inputs, _ = next(data_iter)

            inputs = inputs.to(device)
            with torch.no_grad():
                #feat_in = conv(inputs)
                activation = {}

                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = input[0].detach()
                    return hook

                conv.register_forward_hook(get_activation("conv"))

                # roda um batch normal
                _ = model(inputs)

                feat_in = activation["conv"]
                feat_out = next_conv(feat_in)

            B, C, H, W = feat_in.shape
            X_list.append(feat_in.permute(0, 2, 3, 1).reshape(-1, C))
            Y_list.append(feat_out.permute(0, 2, 3, 1).reshape(-1, next_conv.out_channels))

        X = torch.cat(X_list, dim=0)
        Y = torch.cat(Y_list, dim=0)

        # Seleção gulosa de canais
        num_prune = int(C * pruning_ratio)
        if num_prune == 0:
            continue

        selected = list(range(C))
        pruned = []

        for _ in range(num_prune):
            errors = []
            for c in selected:
                temp = [x for x in selected if x != c]
                X_temp = X[:, temp]
                W_ls = torch.linalg.lstsq(X_temp, Y).solution
                Y_pred = X_temp @ W_ls
                error = F.mse_loss(Y_pred, Y)
                errors.append(error.item())

            worst = selected[errors.index(max(errors))]
            selected.remove(worst)
            pruned.append(worst)

        # Physical removal da camada atual (conv)
        kept = selected
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=len(kept),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None)
        ).to(device)

        with torch.no_grad():
            new_conv.weight.copy_(conv.weight[kept, :, :, :])
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias[kept])

        # Reconstruir próxima camada com input reduzido
        new_next_conv = nn.Conv2d(
            in_channels=len(kept),
            out_channels=next_conv.out_channels,
            kernel_size=next_conv.kernel_size,
            stride=next_conv.stride,
            padding=next_conv.padding,
            bias=(next_conv.bias is not None)
        ).to(device)

        with torch.no_grad():
            new_next_conv.weight.copy_(next_conv.weight[:, kept, :, :])
            if next_conv.bias is not None:
                new_next_conv.bias.copy_(next_conv.bias)

        # Substituir módulos no modelo
        def replace_module(model, old_module, new_module):
            for name, module in model.named_modules():
                if module is old_module:
                    parent = model
                    name_parts = name.split(".")
                    for part in name_parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, name_parts[-1], new_module)
                    return

        replace_module(model_to_prune, conv, new_conv)
        replace_module(model_to_prune, next_conv, new_next_conv)

        sparsity_dict[f"conv_{i}"] = 100. * len(pruned) / C

    # Sparsity global
    total = 0
    zero = 0
    for m in [m for m in model_to_prune.modules() if isinstance(m, nn.Conv2d)]:
        w = m.weight
        total += w.numel()
        zero += (w == 0).sum().item()

    sparsity_dict["global"] = 100. * zero / total
    return model, sparsity_dict


# XOR/XNOR style quantization wrapper (weights + activations) using STE
class XORQuantizeSTE(torch.autograd.Function):
    """Straight Through Estimator quantizer that maps inputs to {-1, +1}.

    Forward: returns sign(x) with positive mapped to +1 and negative to -1.
    Backward: identity pass-through for gradients (STE).
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input >= 0, torch.ones_like(input), -torch.ones_like(input))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: pass gradients unchanged
        return grad_output


class XORWrapper(nn.Module):
    """Wrap a model to perform XOR-style binarization of weights and activations using STE.

    - Weights are binarized to {-1, +1} in-place during binarization() and restored afterward.
    - Activations are quantized to {-1, +1} using an autograd STE function via forward-pre-hooks on
      each quantized layer. This preserves gradient flow for activations.
    - If first_last_fp32=True, the first and last quantizable layers will be excluded.

    Typical usage:
      wrapper = XORWrapper(model, first_last_fp32=True)
      # During training the trainer may call wrapper.binarization() before forward and
      # wrapper.restore() after backward to emulate STE on weights. Activations are always
      # quantized via hooks (they use STE internally).
    """

    def __init__(self, model: nn.Module, clip_value: float = 1.0, first_last_fp32: bool = False) -> None:
        super().__init__()
        self.model = model
        self.clip_value = float(clip_value)

        # Find quantizable layers
        quant_layers: list[nn.Module] = [m for m in self.model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        if first_last_fp32 and len(quant_layers) > 2:
            quant_layers = quant_layers[1:-1]

        # Target parameters (weights) for binarization
        self.target_modules = [module.weight for module in quant_layers]
        self.saved_params = [param.detach().clone() for param in self.target_modules]
        self._weights_binarized = False

        # Mark metadata on modules (compatible with existing wrappers)
        for module in quant_layers:
            module.weight_bit_width = 1
            module.activation_bit_width = 1
            module.quant_scheme = "xor"

        # Register activation quantization hooks (forward-pre hooks) using STE
        self._hook_handles = []
        for module in quant_layers:
            # Define hook that quantizes the incoming activation tensor (input tuple -> new tuple)
            def make_hook():
                def hook_fn(mod, input):
                    # input is a tuple (x, ...) where x is Tensor; quantize x with STE
                    if not isinstance(input, tuple) or len(input) == 0:
                        return input
                    x = input[0]
                    # Only quantize tensors
                    if isinstance(x, torch.Tensor):
                        #x = torch.clamp(x, -1, 1)
                        #x_bin = XORQuantizeSTE.apply(x)
                        alpha_act = mod.weight.abs().mean()  # ou outro fator de escala da camada
                        x_bin = alpha_act * XORQuantizeSTE.apply(x)
                        return (x_bin,) + input[1:]
                    return input
                return hook_fn

            handle = module.register_forward_pre_hook(make_hook())
            self._hook_handles.append(handle)

    def __del__(self):
        # Remove hooks when wrapper is destroyed
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass

    def save_params(self) -> None:
        with torch.no_grad():
            for index, param in enumerate(self.target_modules):
                saved = self.saved_params[index]
                if saved.shape != param.shape or saved.device != param.device or saved.dtype != param.dtype:
                    self.saved_params[index] = param.detach().clone()
                else:
                    self.saved_params[index].copy_(param.detach())

    def binarization(self) -> None:
        """Binarize weights in-place to {-1, +1} and save originals for restoration.

        This produces the STE behaviour used by many binary network implementations: forward
        uses binarized weights, backward computes gradients, then restore() is called to put
        back the full-precision weights before optimizer.step().
        """
        if self._weights_binarized:
            return
        self.save_params()
        with torch.no_grad():
            for param in self.target_modules:
                alpha = param.abs().mean()
                param.copy_(alpha * torch.sign(param))
                #param.copy_(torch.where(param >= 0, torch.ones_like(param), -torch.ones_like(param)))
        self._weights_binarized = True

    def restore(self) -> None:
        if not self._weights_binarized:
            return
        with torch.no_grad():
            for index, param in enumerate(self.target_modules):
                param.copy_(self.saved_params[index])
        self._weights_binarized = False

    def clip(self) -> None:
        with torch.no_grad():
            for param in self.target_modules:
                param.clamp_(-self.clip_value, self.clip_value)

    def post_optimizer_step(self) -> None:
        # Clip weights after optimizer step if needed
        self.clip()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward behaviour:
        - Activations are quantized via registered forward-pre-hooks (using STE), so they are
          quantized for both training and eval.
        - Weights are binarized for inference/eval within this method to keep behaviour self-contained.
        - During training the wrapper avoids automatically binarizing weights so that training
          code can control the binarization window (binarize -> forward -> backward -> restore).
        """
        # Training flow: do not forcibly binarize weights here (trainer may control binarize/restore)
        if self.training and torch.is_grad_enabled():
            return self.model(x)

        # Eval/inference flow: binarize weights for the forward pass, then restore
        self.binarization()
        try:
            return self.model(x)
        finally:
            self.restore()


def apply_xor_quantization(model: nn.Module, clip_value: float = 1.0, first_last_fp32: bool = False) -> XORWrapper:
    """Wrap a model to enable XOR-style (binary) quantization for weights and activations.

    Args:
        model: nn.Module to wrap.
        clip_value: maximum absolute value for clipping weights after optimizer step.
        first_last_fp32: if True, skip quantization for the first and last quantizable layers.
    """
    return XORWrapper(model=model, clip_value=clip_value, first_last_fp32=first_last_fp32)



# ===========================================
#                  CUTMIX
# ===========================================

def cutmix_batch(inputs, targets, alpha=1.0):

    if alpha <= 0:
        return inputs, targets, targets, 1.0

    B, C, H, W = inputs.size()
    lam = np.random.beta(alpha, alpha)

    rw = int(W * math.sqrt(1.0 - lam))
    rh = int(H * math.sqrt(1.0 - lam))
    rx = np.random.randint(0, W)
    ry = np.random.randint(0, H)

    x1 = np.clip(rx - rw // 2, 0, W)
    x2 = np.clip(rx + rw // 2, 0, W)
    y1 = np.clip(ry - rh // 2, 0, H)
    y2 = np.clip(ry + rh // 2, 0, H)

    perm = torch.randperm(B, device=inputs.device)
    target_a = targets   # original targets
    target_b = targets[perm]  # mixed targets

    inputs[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]

    patch_area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - patch_area / float(W * H)

    return inputs, target_a, target_b, lam