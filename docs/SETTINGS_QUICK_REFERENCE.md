# Quick Reference: SAT Selective Classification Settings

## Critical Parameter Comparison

| Parameter | Paper Specification | Current run_cifar10.sh | Status | Fix Required |
|-----------|-------------------|----------------------|--------|--------------|
| **Architecture** | VGG-16 with BatchNorm | `vgg16_bn` | ✅ | No |
| **Loss Function** | SAT | `sat` | ✅ | No |
| **Dataset** | CIFAR-10 | `cifar10` | ✅ | No |
| **Total Epochs** | 300 | 300 (default) | ✅ | No |
| **Batch Size** | 128 | 128 (default) | ✅ | No |
| **Initial LR** | 0.1 | 0.1 (default) | ✅ | No |
| **SGD Momentum** | 0.9 | 0.9 (default) | ✅ | No |
| **Weight Decay** | 0.0005 | 0.0005 (default) | ✅ | No |
| **LR Schedule** | Every 25 epochs | Every 25 epochs (default) | ✅ | No |
| **LR Decay Factor (γ)** | 0.5 | 0.5 (default) | ✅ | No |
| **SAT Momentum (μ)** | **0.99** | **0.9** | ❌ | **YES** |
| **Start Epoch (ε₀)** | **0** | **150** | ❌ | **YES** |

## Issues Found

### 1. SAT Momentum (μ) - CRITICAL
- **Expected:** 0.99
- **Current:** 0.9
- **Variable:** `MOM` in run_cifar10.sh
- **Impact:** Major - affects exponential moving average behavior throughout training

### 2. Start Epoch (ε₀) - CRITICAL  
- **Expected:** 0 (SAT loss from the beginning)
- **Current:** 150 (uses CE loss for first 150 epochs)
- **Variable:** `PRETRAIN` in run_cifar10.sh
- **Impact:** Major - fundamentally changes training approach for first half

## How to Run with Correct Settings

### Option 1: Use the corrected script
```bash
cd /home/viet2005/workspace/Research/ltr_xai/SAT-selective-cls
./run_cifar10_corrected.sh <experiment_name>
```

### Option 2: Modify the original script
Change these two lines in `run_cifar10.sh`:
```bash
PRETRAIN=0    # was 150
MOM=0.99      # was 0.9
```

### Option 3: Override via command line
```bash
python -u train.py --arch vgg16_bn --gpu-id 1 \
       --pretrain 0 --sat-momentum 0.99 \
       --loss sat --dataset cifar10 \
       --save ./log/cifar10_vgg16_bn_sat_corrected
```

## Verification Checklist

Before running, verify:
- [ ] SAT momentum set to 0.99
- [ ] Pretrain epochs set to 0
- [ ] Architecture is vgg16_bn
- [ ] Loss is sat
- [ ] Dataset is cifar10
- [ ] All other defaults are used (they match the paper)

## Files Created
1. `SETTINGS_VERIFICATION.md` - Detailed analysis
2. `run_cifar10_corrected.sh` - Corrected run script
3. `SETTINGS_QUICK_REFERENCE.md` - This file
