---
library_name: pytorch
tags:
- biosignals
- ecgdenoiser
metrics:
- validation_loss
---
# Model Card for ECGDenoiser

    
Collection: NeuralLib: Deep Learning Models for Biosignals Processing

Description: GRU-based model for ECG noise removal. Model and results published in the paper 'Cleaning ECG with Deep Learning: A Denoiser Tested in Industrial Settings'


- **Architecture**: GRUseq2seq
- **Model Name**: ECGDenoiser
- **Task**: ecg denoising: removing MA, BW and EM noise
- **Train Dataset**: PTB-XL+MIT-BIH-Noise-Stress-Test-Database

Biosignal(s): ECG

Sampling frequency: 360


# Benchmark Results

**Validation Loss**: 0.0000 

**Training Time**: 0.00 seconds 

**FLOPs per timestep**: 0 

**Number of trainable parameters**: 26121 



# Hyperparameters

| Parameter | Value |
|-----------|-------|
| bidirectional | True |
| dropout | 0 |
| hid_dim | [64, 1] |
| learning_rate | 0.005 |
| model_name | ECGDenoiser |
| multi_label | False |
| n_features | 1 |
| n_layers | 2 |
| num_classes | NA |
| task | regression |
| fc_out_bool | False |


# Example

import NeuralLib.model_hub as mh

model_name = ECGDenoiser()

model = mh.ProductionModel(model_name=model_name)

signal = torch.rand(1, 100, 1)  # Example input signal

predictions = model.predict(signal)

print(predictions)

