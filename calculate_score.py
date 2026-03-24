from models.densenet import DenseNet169
from models.resnet import ResNet20
import torch
import torch.nn as nn
from collections import OrderedDict
from thop import profile
from models import ResNet18
import os
import matplotlib.pyplot as plt
import numpy as np

# ===== CONFIG =====
REFERENCE_PARAMS = 5.6e6
REFERENCE_OPS = 2.8e8
RESNET18_WEIGHTS = 11173962
RESNET18_MACS = 5.58e8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Functions =====
def load_model(path):
    net = ResNet18().to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    return net

def count_weights(model):
    total = sum(p.numel() for p in model.parameters())
    #nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total

def compute_macs(model):
    model = model.to(DEVICE)
    model.eval()
    input = torch.randn(1, 3, 32, 32).to(DEVICE)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(input,), verbose=False)  # it's possible to get both macs and params (weights), but we only need macs here
    return macs

def compute_score(ps, pu, qw, qa, w, f):
    param_term = ((1 - (ps + pu)) * (qw / 32) * w) / REFERENCE_PARAMS
    ops_term = ((1 - ps) * (max(qw, qa) / 32) * f) / REFERENCE_OPS
    score = param_term + ops_term
    return score
    
# ===== Main =====
def main():
    # Caminhos dos modelos
    #models_paths = {
        #"basic": "checkpoint-ResNet18-global_pruning_retrain/trained_model.pth",
        #"binary": "checkpoint-ResNet18-trainBinary200/trainBinary200.pth",
    #}

    acc = {
        "ResNet18": 94.94,
        "40unstr": 94.42,
        "70unstr": 94.61,
        "40str": 88.8,
        "70str": 87.39,
        "75widthscale": 94.41,
        "depthwise": 91.94,
        "spatial": 91.24,
        "grouped": 93.38,
        "depthwise_75widthscale": 91.82,

        "binary_40unst": 91.09,
        "binary_70unstr": 91.4,
        "binary_40str": 88.8,
        "binary_70str": 87.39,
        
        "fp16": 94.95,
        "fp16_40unst_20str": 94.27,
        "fp16_60unst_30str": 94.45,
        "fp16_70unst_40str": 94.25,
        "75widthscale_fp16_70unst_40str": 93.51,
        "75widthscale_fp16_70unst_50str": 93.39,
        "depthwise_fp16_70unst_40str": 90.73,
        "depthwise_fp16_70unst_50str": 90.27,
        "depthwise_75widthscale_fp16_70unst_40str": 90.56,
        
        "DenseNet169": 93.79,
        "ResNet20": 91.96,
        "ResNet20_fp16_70unst_40str": 90.22,
        "student_ResNet18_fp16_70unst_40str": 95.80,
    }
    
    scores = {}

    for key, path in acc.items():
        #print(f"\n=== Loading {key} model ===")
        #net = load_model(path)

        # ===== CASOS =====
        if key == "ResNet18":
            # Sem quantização, sem pruning
            scores["ResNet18"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "40unstr":
            # Sem quantização + 40% unstructured
            scores["40unstr"] = compute_score(ps=0, pu=0.4, qw=32, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "70unstr":
            # Sem quantização + 70% unstructured
            scores["70unstr"] = compute_score(ps=0, pu=0.7, qw=32, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "40str":
            # Sem quantização + 40% structured
            scores["40str"] = compute_score(ps=0.4, pu=0, qw=32, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "70str":
            # Sem quantização + 70% structured
            scores["70str"] = compute_score(ps=0.7, pu=0, qw=32, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "75widthscale":
            model_scaled = ResNet18(width_mult=0.75).to(DEVICE)
            w_width_scale = count_weights(model_scaled)
            #print(f"w scaled model - Params: {w_width_scale}")
            f_width_scale = compute_macs(model_scaled)
            #print(f"f scaled model - MACs: {f_width_scale}")
            scores["75widthscale"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_width_scale, f=f_width_scale)
            
            # Sem quantização, sem pruning + width scaling 75%
            #w_width_scale = RESNET18_WEIGHTS * 0.75
            #f_width_scale = RESNET18_MACS * 0.75
            #scores["75widthscale"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_width_scale, f=f_width_scale)
        
        if key == "depthwise":
            net = ResNet18(conv_type="depthwise", width_mult=1.0).to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["depthwise"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_real, f=f_real)
            
        if key == "spatial":
            net = ResNet18(conv_type="spatial", width_mult=1.0).to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["spatial"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_real, f=f_real)

        if key == "grouped":
            net = ResNet18(conv_type="grouped", width_mult=1.0).to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["grouped"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_real, f=f_real)
            
        if key == "depthwise_75widthscale":
            net = ResNet18(conv_type="depthwise", width_mult=0.75).to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["depthwise_75widthscale"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_real, f=f_real)

            
        if key == "binary_40unst":
            # BinaryConnect (1 bit pesos)
            # 40% unstructured
            scores["binary_40unst"] = compute_score(ps=0, pu=0.4, qw=1, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "binary_70unstr":
            # 70% unstructured
            scores["binary_70unstr"] = compute_score(ps=0, pu=0.7, qw=1, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key =="binary_40str":
            scores["binary_40str"] = compute_score(ps=0.4, pu=0, qw=1, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key =="binary_70str":
            scores["binary_70str"] = compute_score(ps=0.7, pu=0, qw=1, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)

        if key == "fp16":
            # fp16, no pruning
            scores["fp16"] = compute_score(ps=0, pu=0, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "fp16_40unst_20str":
            # 40% unstructured + 20% structured + eval fp16,
            scores["fp16_40unst_20str"] = compute_score(ps=0.2, pu=0.4, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "fp16_60unst_30str":
            scores["fp16_60unst_30str"] = compute_score(ps=0.3, pu=0.6, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        if key == "fp16_70unst_40str":
            scores["fp16_70unst_40str"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
        
        if key == "75widthscale_fp16_70unst_40str":
            model_scaled = ResNet18(width_mult=0.75).to(DEVICE)
            w_width_scale = count_weights(model_scaled)
            print(f"w scaled model - Params: {w_width_scale}")
            f_width_scale = compute_macs(model_scaled)
            print(f"f scaled model - MACs: {f_width_scale}")
            scores["75widthscale_fp16_70unst_40str"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=w_width_scale, f=f_width_scale)
        
        if key == "75widthscale_fp16_70unst_50str":
            model_scaled = ResNet18(width_mult=0.75).to(DEVICE)
            w_width_scale = count_weights(model_scaled)
            print(f"w scaled model - Params: {w_width_scale}")
            f_width_scale = compute_macs(model_scaled)
            print(f"f scaled model - MACs: {f_width_scale}")
            scores["75widthscale_fp16_70unst_50str"] = compute_score(ps=0.5, pu=0.7, qw=16, qa=16, w=w_width_scale, f=f_width_scale)
    
        if key == "depthwise_fp16_70unst_40str":
            net = ResNet18(conv_type="depthwise", width_mult=1.0).to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["depthwise_fp16_70unst_40str"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=w_real, f=f_real)
            
        if key == "depthwise_fp16_70unst_50str":
            net = ResNet18(conv_type="depthwise", width_mult=1.0).to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["depthwise_fp16_70unst_50str"] = compute_score(ps=0.5, pu=0.7, qw=16, qa=16, w=w_real, f=f_real)

        if key == "depthwise_75widthscale_fp16_70unst_40str":
            net = ResNet18(conv_type="depthwise", width_mult=0.75).to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["depthwise_75widthscale_fp16_70unst_40str"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=w_real, f=f_real)
            
        if key == "DenseNet169":
            net = DenseNet169().to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["DenseNet169"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_real, f=f_real)
        
        if key == "ResNet20":
            net = ResNet20().to(DEVICE)
            w_real = count_weights(net)
            print(f"w ResNet20 - Params: {w_real}")
            #270000
            f_real = compute_macs(net)
            print(f"f ResNet20 - MACs: {f_real}")
            #40810000
            scores["ResNet20"] = compute_score(ps=0, pu=0, qw=32, qa=32, w=w_real, f=f_real)
            
        if key == "ResNet20_fp16_70unst_40str":
            net = ResNet20().to(DEVICE)
            w_real = count_weights(net)
            f_real = compute_macs(net)
            scores["ResNet20_fp16_70unst_40str"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=w_real, f=f_real)
            
        if key == "student_ResNet18_fp16_70unst_40str":
            scores["student_ResNet18_fp16_70unst_40str"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)

    # ===== PLOT =====
    def get_color(k):
        if "ResNet18" in k:
            return "red"
        if "DenseNet169" in k:
            return "red"
        if "ResNet20" in k:
            return "red"
        elif "binary" in k:
            return "blue"
        elif "fp16" in k:
            return "orange"
        return "green"
    
    plt.figure(figsize=(9,5))
    '''
    for k in acc.keys():
        if k in scores:
            plt.scatter(scores[k], acc[k], label=k, s=40)
    '''
    for k in acc.keys():
        if k in scores:
            x = scores[k]
            y = acc[k]
            
            plt.scatter(x, y, s=40, color=get_color(k))

            
            if k == "75widthscale_fp16_70unst_50str" or k == "depthwise_75widthscale" or k == "fp16_40unst_20str" or k == "70str" or k == "depthwise" or k == "ResNet20_fp16_70unst_40str":
                plt.text(x, y, k, fontsize=7, ha='left', va='top')
            else:
                plt.text(x, y, k, fontsize=7, ha='left', va='bottom')


    # Line in acc = 90%
    plt.axhline(y=90, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Score")
    plt.ylabel("Accuracy (%)")
    plt.title("Score x Accuracy")
    #plt.legend(fontsize=6, loc='lower right')
    plt.grid(True)
    plt.savefig("score_vs_accuracy.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== PRINT SCORES =====
    print("\n===== SCORES =====")
    print(f"Basic model (no quant, no prune): {scores['ResNet18']:.6f}")
    print(f"No quant, 40% Unstructured: {scores['40unstr']:.6f}")
    print(f"No quant, 70% Unstructured: {scores['70unstr']:.6f}")
    print(f"No quant, 40% Structured: {scores['40str']:.6f}")
    print(f"No quant, 70% Structured: {scores['70str']:.6f}")
    print(f"Width Scale 75%: {scores['75widthscale']:.6f}")
    
    print(f"Depthwise: {scores['depthwise']:.6f}")
    print(f"Spatial: {scores['spatial']:.6f}")
    print(f"Grouped: {scores['grouped']:.6f}")
    print(f"Depthwise, Width Scale 75%: {scores['depthwise_75widthscale']:.6f}")

    print(f"Binary, 40% Unstructured: {scores['binary_40unst']:.6f}")
    print(f"Binary, 70% Unstructured: {scores['binary_70unstr']:.6f}")
    print(f"Binary, 40% Structured: {scores['binary_40str']:.6f}")
    print(f"Binary, 70% Structured: {scores['binary_70str']:.6f}")
    
    print(f"FP16, no pruning: {scores['fp16']:.6f}")
    print(f"FP16, 40% Unstructured + 20% Structured: {scores['fp16_40unst_20str']:.6f}")
    print(f"FP16, 60% Unstructured + 30% Structured: {scores['fp16_60unst_30str']:.6f}")
    print(f"FP16, 70% Unstructured + 40% Structured: {scores['fp16_70unst_40str']:.6f}")
    print(f"Width Scale 75% + FP16 + 70% Unstructured + 40% Structured: {scores['75widthscale_fp16_70unst_40str']:.6f}")
    print(f"Width Scale 75% + FP16 + 70% Unstructured + 50% Structured: {scores['75widthscale_fp16_70unst_50str']:.6f}")
    
    print(f"Depthwise, FP16, 70% Unstructured + 40% Structured: {scores['depthwise_fp16_70unst_40str']:.6f}")
    print(f"Depthwise, FP16, 70% Unstructured + 50% Structured: {scores['depthwise_fp16_70unst_50str']:.6f}")
    print(f"Depthwise, Width Scale 75%, FP16, 70% Unstructured + 40% Structured: {scores['depthwise_75widthscale_fp16_70unst_40str']:.6f}")

    print(f"DenseNet169: {scores['DenseNet169']:.6f}")
    print(f"ResNet20: {scores['ResNet20']:.6f}")
    print(f"ResNet20, FP16, 70% Unstructured + 40% Structured: {scores['ResNet20_fp16_70unst_40str']:.6f}")
    print(f"Student ResNet18, FP16, 70% Unstructured + 40% Structured: {scores['student_ResNet18_fp16_70unst_40str']:.6f}")
    
    
    # Possible scenarios:
    # Ps = 0.4, Pu = 0.4
    print("\n===== POSSIBLE SCENARIOS =====")
    scores["40unstructed_40structed"] = compute_score(ps=0.4, pu=0.4, qw=32, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
    print(f"Ps=0.4, Pu=0.4: {scores['40unstructed_40structed']:.6f}")
    # Ps = 0.2, Pu = 0.2
    scores["20unstructed_20structed"] = compute_score(ps=0.2, pu=0.2, qw=32, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
    print(f"Ps=0.2, Pu=0.2: {scores['20unstructed_20structed']:.6f}")
    # Ps = 0.2, Pu = 0.2, qw=1
    scores["binary_70unstr_70structed"] = compute_score(ps=0.7, pu=0.7, qw=1, qa=32, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
    print(f"Ps=0.7, Pu=0.7, qw=1: {scores['binary_70unstr_70structed']:.6f}")
    scores["binary_0unstructed_0structed"] = compute_score(ps=0, pu=0, qw=1, qa=1, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
    print(f"Ps=0, Pu=0, qw=1, qa=1: {scores['binary_0unstructed_0structed']:.6f}")
    scores["fp16"] = compute_score(ps=0, pu=0, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
    print(f"Ps=0, Pu=0, qw=16, qa=16: {scores['fp16']:.6f}")
    scores["fp16_40unst_20str"] = compute_score(ps=0.2, pu=0.4, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
    print(f"Ps=0.2, Pu=0.4, qw=16, qa=16: {scores['fp16_40unst_20str']:.6f}")
    scores["fp16_70unst_40str"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=RESNET18_WEIGHTS, f=RESNET18_MACS)
    print(f"Ps=0.4, Pu=0.7, qw=16, qa=16: {scores['fp16_70unst_40str']:.6f}")
    # As we can observe by the results, applying both Ps and Pu together reduces the score, but not as much as when the model is quantized.
    # In terms of score, implementing quantization on the activations can be promoting. Even more than only weight quantization with a strong pruning.
    # Scenarios I'll work on: 
    # Ps=0, Pu=0, qw=16, qa=16: 1.994104 
    # Ps=0.2, Pu=0.4, qw=16, qa=16: 0.996927 
    # Ps=0, Pu=0, qw=1, qa=1: 0.124631
    
    # Next approach: width scaling (reduce the number of channels) or change into ResNet20
    #w_width_scale = RESNET18_WEIGHTS * 0.75
    #f_width_scale = RESNET18_MACS * 0.75
    #scores["fp16_75width_scale"] = compute_score(ps=0, pu=0, qw=16, qa=16, w=w_width_scale, f=f_width_scale)
    #print(f"75width_scale Ps=0, Pu=0, qw=16, qa=16: {scores['fp16_75width_scale']:.6f}")
    #scores["fp16_75width_scale"] = compute_score(ps=0.4, pu=0.7, qw=16, qa=16, w=w_width_scale, f=f_width_scale)
    #print(f"75width_scale Ps=0.4, Pu=0.7, qw=16, qa=16: {scores['fp16_75width_scale']:.6f}")

if __name__ == "__main__":
    main()