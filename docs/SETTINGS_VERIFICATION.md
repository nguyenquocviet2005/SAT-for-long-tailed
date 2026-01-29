# SAT Selective Classification Settings Verification

## Current Configuration Analysis

### ✅ Verified Correct Settings

#### 1. Architecture
- **Required:** VGG-16 with Batch Normalization
- **Current:** `ARCH=vgg16_bn` in run script ✅
- **Code:** `train.py` line 62 default is `vgg16_bn` ✅
- **Model includes:** Batch Normalization and Dropout (built into architecture) ✅

#### 2. SGD Optimizer Settings
- **Required:** SGD with momentum=0.9, weight_decay=0.0005, initial_lr=0.1
- **Current Code (train.py lines 45-48):**
  ```python
  parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
  parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
  parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
  ```
  - Initial LR: 0.1 ✅
  - Momentum: 0.9 ✅
  - Weight Decay: 5e-4 (0.0005) ✅

#### 3. Batch Size
- **Required:** 128
- **Current Code (train.py line 36):**
  ```python
  parser.add_argument('--train-batch', default=128, type=int, metavar='N')
  ```
  - Batch Size: 128 ✅

#### 4. Total Epochs
- **Required:** 300
- **Current Code (train.py line 34):**
  ```python
  parser.add_argument('--epochs', default=300, type=int, metavar='N')
  ```
  - Total Epochs: 300 ✅

#### 5. Learning Rate Scheduler
- **Required:** Decay by 0.5 every 25 epochs
- **Current Code (train.py lines 42-44):**
  ```python
  parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75,100,125,150,175,200,225,250,275])
  parser.add_argument('--gamma', type=float, default=0.5)
  ```
  - Schedule: Every 25 epochs (25, 50, 75, ..., 275) ✅
  - Gamma: 0.5 ✅
  - **Implementation (train.py lines 385-389):**
  ```python
  def adjust_learning_rate(optimizer, epoch):
      global state
      if epoch in args.schedule:
          state['lr'] *= args.gamma
          for param_group in optimizer.param_groups:
              param_group['lr'] = state['lr']
  ```

#### 6. Dataset
- **Required:** CIFAR-10
- **Current:** `DATASET=cifar10` in run script ✅

---

### ⚠️ CRITICAL ISSUES FOUND

#### 1. SAT Momentum (μ)
- **Required:** 0.99 (for selective classification)
- **Current in run_cifar10.sh:** `MOM=0.9` ❌
- **Passed as:** `--sat-momentum ${MOM}` → results in 0.9
- **Default in train.py (line 47):** `default=0.9` ❌

**Impact:** This is a CRITICAL parameter difference. The paper specifically states that for selective classification, the momentum should be **0.99**, not 0.9.

**Fix Required:**
```bash
# In run_cifar10.sh, change:
MOM=0.9
# To:
MOM=0.99
```

#### 2. Start Epoch (ε₀)
- **Required:** 0 (moving average starts immediately)
- **Current in run_cifar10.sh:** `PRETRAIN=150` ❌
- **Expected:** `PRETRAIN=0` for SAT loss to apply from the start

**Impact:** The code uses cross-entropy loss for the first 150 epochs as "pretraining", then switches to SAT loss. This contradicts the paper's specification that ε₀ = 0.

**Code Logic (train.py lines 244-249):**
```python
if epoch >= args.pretrain:
    if args.loss == 'gambler':
        loss = criterion(outputs, targets, reward)
    elif args.loss == 'sat':
        loss = criterion(outputs, targets, indices)
```

**Fix Required:**
```bash
# In run_cifar10.sh, change:
PRETRAIN=150
# To:
PRETRAIN=0
```

---

## SAT Loss Implementation Verification

### Loss Function (loss.py)

The SAT loss implementation:
```python
class SelfAdativeTraining():
    def __init__(self, num_examples=50000, num_classes=10, mom=0.9):
        self.prob_history = torch.zeros(num_examples, num_classes)
        self.updated = torch.zeros(num_examples, dtype=torch.int)
        self.mom = mom  # This should be 0.99
        self.num_classes = num_classes
```

**Key Components:**
1. ✅ Maintains exponential moving average (`prob_history`)
2. ✅ Uses momentum parameter for EMA update
3. ✅ Tracks whether examples have been initialized (`updated`)
4. ✅ Creates soft labels based on EMA predictions

**Loss Computation:**
```python
def __call__(self, logits, y, index):
    prob = F.softmax(logits.detach()[:, :self.num_classes], dim=1)
    prob = self._update_prob(prob, index, y)
    
    soft_label = torch.zeros_like(logits)
    soft_label[torch.arange(y.shape[0]), y] = prob[torch.arange(y.shape[0]), y]
    soft_label[:, -1] = 1 - prob[torch.arange(y.shape[0]), y]
    soft_label = F.normalize(soft_label, dim=1, p=1)
    loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)
    return torch.mean(loss)
```

This matches the paper's loss formulation where:
- The soft label for correct class = p̃ᵢ (EMA prediction)
- The soft label for abstention = 1 - p̃ᵢ
- Loss is cross-entropy with these soft labels

---

## Recommended Fixes

### Update run_cifar10.sh:

```bash
ARCH=vgg16_bn
LOSS=sat
DATASET=cifar10
PRETRAIN=0          # ← CHANGED: Start SAT from epoch 0
MOM=0.99            # ← CHANGED: Use momentum 0.99 for selective classification
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_$1
GPU_ID=1

mkdir -p ./log

### train
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log

### eval
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --loss ${LOSS} --reward ${REWARD} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
```

---

## Summary

### Settings That Match Paper:
1. ✅ Architecture: VGG-16 with BatchNorm
2. ✅ Optimizer: SGD
3. ✅ Initial Learning Rate: 0.1
4. ✅ SGD Momentum: 0.9
5. ✅ Weight Decay: 0.0005
6. ✅ Batch Size: 128
7. ✅ Total Epochs: 300
8. ✅ LR Schedule: Decay by 0.5 every 25 epochs
9. ✅ Dataset: CIFAR-10

### Settings That DON'T Match Paper:
1. ❌ **SAT Momentum (μ):** Currently 0.9, should be **0.99**
2. ❌ **Start Epoch (ε₀):** Currently 150, should be **0**

### Impact Assessment:
- **High Impact:** SAT momentum being 0.9 instead of 0.99 will significantly affect the EMA behavior and model performance
- **High Impact:** Starting SAT at epoch 150 instead of 0 means the first half of training uses standard cross-entropy, which is fundamentally different from the paper's approach
