---
library_name: pytorch
tags:
- biosignals
- ecgdenoisernl
metrics:
- validation_loss
---
# Model Card for ECGDenoiserNL

    
- Collection: NeuralLib: Deep Learning Models for Biosignals Processing

- Description: GRU-based model for ECG peak detection


```json
{
    "architecture": "GRUseq2seq",
    "model_name": "ECGDenoiser",
    "train_dataset": "PTB-XL+MIT-BIH-Noise-Stress-Test-Database",
    "biosignal": "ECG",
    "sampling_frequency": 360,
    "task": "ecg denoising: removing MA, BW and EM noise",
    "gpu_model": "NVIDIA GeForce GTX 1080 Ti",
    "epochs": 200,
    "optimizer": "Adam",
    "learning_rate": 0.005,
    "validation_loss": 0,
    "training_time": 0,
    "retraining": false,
    "efficiency_flops": 0,
    "efficiency_params": 26121
}


## Hyperparameters

bidirectional: true
dropout: 0
hid_dim:
- 64
- 1
learning_rate: 0.005
model_name: ECGDenoiser
multi_label: false
n_features: 1
n_layers: 2
num_classes: NA
task: regression
fc_out_bool: false


# Example

import torch

from production_models import ECGDenoiserNL

model = ECGDenoiserNL()

signal = torch.rand(1, 100, 1)  # Example input signal

predictions = model.predict(signal)

print(predictions)

