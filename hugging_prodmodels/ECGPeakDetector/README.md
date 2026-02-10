---
library_name: pytorch
tags:
- biosignals
- testmodel
metrics:
- validation_loss
---
# Model Card for Testmodel

    
- Collection: NeuralLib: Deep Learning Models for Biosignals Processing

- Description: This is a Test


```json
{
    "architecture": "GRUseq2seq",
    "model_name": "ECGPeakDetector",
    "train_dataset": "private_gib01",
    "biosignal": "ECG",
    "sampling_frequency": 360,
    "task": "peak detection",
    "gpu_model": "NVIDIA GeForce GTX 1080 Ti",
    "epochs": 80,
    "optimizer": "Adam (\nParameter Group 0\n    amsgrad: False\n    betas: (0.9, 0.999)\n    capturable: False\n    differentiable: False\n    eps: 1e-08\n    foreach: None\n    fused: None\n    initial_lr: 0.001\n    lr: 0.001\n    maximize: False\n    weight_decay: 1e-05\n)",
    "learning_rate": 0.001,
    "validation_loss": 0.14879398047924042,
    "training_time": 11375.492486476898,
    "retraining": false,
    "efficiency_flops": 0,
    "efficiency_params": 0
}


## Hyperparameters

bidirectional: true
dropout: 0
hid_dim:
- 32
- 64
- 64
learning_rate: 0.001
model_name: ECGPeakDetector
multi_label: true
n_features: 1
n_layers: 3
num_classes: 1
task: classification


# Example

import torch

from production_models import Testmodel

model = Testmodel()

signal = torch.rand(1, 100, 1)  # Example input signal

predictions = model.predict(signal)

print(predictions)

