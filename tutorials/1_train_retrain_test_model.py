from NeuralLib.architectures import GRUseq2seq
from config_ori import DATASETS_ECG_G
import os

# Data paths
X = os.path.join(DATASETS_ECG_G, 'x')
Y_BIN = os.path.join(DATASETS_ECG_G, 'y_bin')

# Step 1: Define architecture parameters
arch_params = {
    'n_features': 1,  # number of channels / features of the input
    'hid_dim': 16,
    'n_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.01,
    'bidirectional': True,
    'task': 'classification',
    'num_classes': 1,
    'multi_label': True,
}

# Step 2: Define training parameters
train_params_ = {
    'path_x': X,
    'path_y': Y_BIN,
    'epochs': 3,
    'batch_size': 1,
    'patience': 2,
    'dataset_name': 'private_gib01',
    'trained_for': 'peak detection',
    'all_samples': False,
    'samples': 3,
    'gpu_id': None,
    'enable_tensorboard': True
}

# Step 3: Initialize and train the GRUseq2seq model
print("Training from scratch...")
model = GRUseq2seq(**arch_params)
model.train_from_scratch(**train_params_)

# Save checkpoints directory after initial training
checkpoints_dir = model.checkpoints_directory
# Note: the checkpoints_dir is created in RESULTS_BASE_DIR\<model_name> (check config file)

# Step 4: Retrain the model for 2 more epochs
print("Retraining...")
train_params_retrain = train_params_.copy()
train_params_retrain['epochs'] = 2
model.retrain(
    path_x=train_params_retrain['path_x'],
    path_y=train_params_retrain['path_y'],
    patience=train_params_retrain['patience'],
    batch_size=train_params_retrain['batch_size'],
    epochs=train_params_retrain['epochs'],
    gpu_id=train_params_retrain['gpu_id'],
    all_samples=train_params_retrain['all_samples'],
    samples=train_params_retrain['samples'],
    dataset_name=train_params_retrain['dataset_name'],
    trained_for=train_params_retrain['trained_for'],
    checkpoints_directory=checkpoints_dir,
    enable_tensorboard=train_params_retrain['enable_tensorboard'],
)

# Step 5: Test the model on the test set
print("Testing on test set...")
predictions, avg_loss = model.test_on_test_set(
    path_x=train_params_['path_x'],
    path_y=train_params_['path_y'],
    checkpoints_dir=checkpoints_dir,
    gpu_id=train_params_['gpu_id'],
    all_samples=False,  # if True, test on all available samples
    samples=5,
    save_predictions=True
)

print(f"Average Test Loss: {avg_loss:.4f}")
