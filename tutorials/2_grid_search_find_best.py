import NeuralLib.architectures as arc
import torch
from config_ori import DATASETS_ECG_G
import os

# Data paths
X = os.path.join(DATASETS_ECG_G, 'x')
Y_BIN = os.path.join(DATASETS_ECG_G, 'y_bin')

# Step 1: Define architecture and training parameters
architecture_name = 'GRUseq2seq'
print(arc.get_valid_architectures())
archi_params_options = {
    "model_name": 'ECGPeakDetector',
    "n_features": [1],
    "hid_dim": [[32, 64, 64], [64, 64, 64], [64, 128, 64], [64, 128]],
    "n_layers": [3, 2],
    "dropout": [0.3, 0],
    "learning_rate": [0.001],
    "bidirectional": [True],
    "task": ["classification"],
    "multi_label": [True],
    "num_classes": [1],
}

train_params = {
    'path_x': X,
    'path_y': Y_BIN,
    'epochs': 2,
    'batch_size': 1,
    'patience': 2,
    'dataset_name': 'private_gib01',
    'trained_for': 'peak detection',
    'all_samples': False,
    'samples': 3,
    'gpu_id': None,
    'enable_tensorboard': True,
}

# Step 2: Perform grid search
print("Performing grid search...")
best_dir, best_val_loss, val_losses = arc.run_grid_search(
    architecture_name, archi_params_options, train_params
)
print(f"Best model found with validation loss: {best_val_loss:.4f}, saved in {best_dir}")

# Step 3: Test the best model on the test set
print("Testing the best model on the test set...")
# 3.1. Load architecture parameters from the hparams.yaml file
architecture_params = arc.get_hparams_from_checkpoints(best_dir)
# 3.2 Initialize the model using the loaded parameters
architecture_class = getattr(arc, architecture_name, None)
model = architecture_class(**architecture_params)

predictions, avg_loss = model.test_on_test_set(
    path_x=train_params["path_x"],
    path_y=train_params["path_y"],
    checkpoints_dir=best_dir,
    gpu_id=train_params["gpu_id"],
    save_predictions=True,
    all_samples=False,
    samples=5,
)
print(f"Testing complete. Average Test Loss: {avg_loss:.4f}")

# Step 4: Test the best model on a single signal
print("Testing on a single signal...")
single_signal = torch.rand(100, 1)  # Example input signal (sequence length: 100, 1 feature)
single_prediction = model.test_on_single_signal(single_signal, checkpoints_dir=best_dir, gpu_id=train_params["gpu_id"])
print(f"Single Signal Prediction: {single_prediction}")


