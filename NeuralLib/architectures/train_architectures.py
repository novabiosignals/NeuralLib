import itertools
import NeuralLib.architectures.biosignals_architectures as arc
from NeuralLib.architectures.base import get_hparams_from_checkpoints, get_hparams_from_hugging, validate_training_context


def get_valid_architectures():
    """
    Get a list of valid architecture classes defined in biosignals_architectures.py.
    """
    # Get all class names defined in biosignals_architectures
    valid_architectures = [
        name for name in dir(arc)
        if isinstance(getattr(arc, name), type) and getattr(
            arc, name).__module__ == 'NeuralLib.architectures.biosignals_architectures'
    ]
    return valid_architectures


def validate_architecture_name(architecture_name):
    """
    Validate if the provided architecture name is valid.
    Ensures that the user provides the name of the class as a string, not the class itself.
    """
    # Check if the user mistakenly passed a class instead of a string
    if isinstance(architecture_name, type):
        raise TypeError(
            f"Invalid input: {architecture_name} is a class. "
            f"Please provide the name of the architecture class as a string (e.g., 'GRUseq2seq')."
        )
    # Validate if the string corresponds to a valid architecture
    valid_architectures = get_valid_architectures()
    if architecture_name not in valid_architectures:
        raise ValueError(f"Invalid architecture name: {architecture_name}. "
                         f"Valid architectures are: {', '.join(valid_architectures)}")
    else:
        print(f"Architecture {architecture_name} is valid.")


def train_architecture_from_scratch(architecture_name, architecture_params, train_params):
    """
    Train a specific architecture dynamically.
    :param architecture_name: Name of the architecture class (e.g., 'GRUseq2seq').
    :param architecture_params: Dictionary of parameters for the architecture.
    :param train_params: Dictionary containing paths and settings for training data.
    """
    validate_architecture_name(architecture_name)
    # Dynamically get the class from biosignals_architectures
    architecture_class = getattr(arc, architecture_name, None)
    if not architecture_class:
        raise ValueError(f"Architecture {architecture_name} not found in biosignals_architectures.")

    # Instantiate the architecture with provided parameters
    model = architecture_class(**architecture_params)

    # Train the model
    model.train_from_scratch(
        path_x=train_params['path_x'],
        path_y=train_params['path_y'],
        all_samples=train_params.get('all_samples', False),
        samples=train_params.get('samples', None),
        epochs=train_params['epochs'],
        batch_size=train_params['batch_size'],
        patience=train_params['patience'],
        dataset_name=train_params['dataset_name'],
        trained_for=train_params['trained_for'],
        gpu_id=train_params.get('gpu_id', None),
        enable_tensorboard=train_params['enable_tensorboard'],
        min_max_norm_sig=train_params.get('min_max_norm_sig', False)
    )

    print("Training completed successfully.")

    return model.checkpoints_directory, model.training_info['validation_loss']


def retrain_architecture(architecture_name, train_params, checkpoints_directory=None, hugging_face_model=None):
    """
    Retrain an architecture using saved checkpoints.
    :param architecture_name:
    :param hugging_face_model:
    :param checkpoints_directory: Directory containing checkpoints and training info.
    :param train_params: Dictionary containing paths and settings for training data.
    """
    validate_training_context(retraining=True, checkpoints_directory=checkpoints_directory,
                              hugging_face_model=hugging_face_model)
    validate_architecture_name(architecture_name)

    # Load architecture's hyperparameters (number of layers, nodes, etc - structure of the model)
    if checkpoints_directory:
        hparams = get_hparams_from_checkpoints(checkpoints_directory)
    else:
        hparams = get_hparams_from_hugging(hugging_face_model)

    # Dynamically get the architecture class from biosignals_architectures
    architecture_class = getattr(arc, architecture_name, None)
    if not architecture_class:
        raise ValueError(f"Architecture {architecture_name} not found in biosignals_architectures.")

    # Instantiate the architecture using the loaded hyperparameters
    model = architecture_class(**hparams)

    # Retrain the model
    model.retrain(
        path_x=train_params['path_x'],
        path_y=train_params['path_y'],
        all_samples=train_params.get('all_samples', False),
        samples=train_params.get('samples', None),
        epochs=train_params['epochs'],
        batch_size=train_params['batch_size'],
        patience=train_params['patience'],
        dataset_name=train_params['dataset_name'],
        trained_for=train_params['trained_for'],
        gpu_id=train_params.get('gpu_id', None),
        checkpoints_directory=checkpoints_directory,
        hugging_face_model=hugging_face_model,
        enable_tensorboard=train_params['enable_tensorboard'],
        min_max_norm_sig=train_params.get('min_max_norm_sig', False),
    )

    print("Retraining completed successfully.")

    return model.checkpoints_directory, model.training_info['validation_loss']


def run_grid_search(architecture_name, architecture_params_options, train_params):
    """
    Perform grid search over the architecture parameters.
    """
    validate_architecture_name(architecture_name)

    # Extract parameter names with multiple options
    param_names = list(architecture_params_options.keys())
    # param_values = [architecture_params_options[key] for key in param_names]
    param_values = [[architecture_params_options[key]] if isinstance(architecture_params_options[key], str)
                    else architecture_params_options[key]
                    for key in param_names]

    # Generate all combinations of hyperparameters
    combinations = list(itertools.product(*param_values))

    print(f"Running grid search with {len(combinations)} combinations.")

    best_val_loss = float('inf')
    best_params = None
    best_dir = None
    val_losses = []

    for i, combination in enumerate(combinations, 1):
        current_params = dict(zip(param_names, combination))
        print(f"\nTraining model {i}/{len(combinations)} with parameters: {current_params}")

        try:  # if there is an error when running a given model, it moves on to the next one
            # Train from scratch
            checkpoints_dir, val_loss = train_architecture_from_scratch(architecture_name, current_params, train_params)

            val_losses.append(val_loss)

            # Update best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = current_params
                best_dir = checkpoints_dir

            print(f"Model {i} completed. Validation Loss: {val_loss:.4f}. Checkpoints saved to: {checkpoints_dir}")
        except Exception as e:
            print(f"Error in model {i}: {e}")

    print("\nGrid search complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Checkpoints saved in: {best_dir}")

    return best_dir, best_val_loss, val_losses


# Example usage
if __name__ == "__main__":

    from config_ori import DATASETS_GIB01
    import os
    X = os.path.join(DATASETS_GIB01, 'x')
    Y_BIN = os.path.join(DATASETS_GIB01, 'y_bin')
    architecture_name_ = 'GRUseq2seq'
    archi_params_ = {
        'model_name': 'ECGPeakDetector',
        'n_features': 1,
        'hid_dim': 8,
        'n_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'bidirectional': True,
        'task': 'classification',
        'num_classes': 1
    }

    train_params_ = {
        'path_x': X,
        'path_y': Y_BIN,
        'epochs': 1,
        'batch_size': 1,
        'patience': 2,
        'dataset_name': 'private_gib01',
        'trained_for': 'peak detection',
        'all_samples': False,
        'samples': 3,
        'gpu_id': None,
        'enable_tensorboard': True
    }

    # Train the GRUseq2seq architecture
    # checkpoints_dir, val_loss = train_architecture_from_scratch('GRUseq2seq', archi_params_, train_params_)
    # print(val_loss)

    # Retrain the architecture (with the same data and parameters in this case, just as an example)
    # retrain_architecture('GRUseq2seq', train_params_, checkpoints_directory=checkpoints_dir)

    print(get_valid_architectures())

    # perform grid search
    architecture_params_options_ = {
        'n_features': [1],
        'hid_dim': [8, 16],
        'n_layers': [1],  # , 2],
        'dropout': [0.2, 0.3],
        'learning_rate': [0.001],  # , 0.0005],
        'bidirectional': [True],
        'task': ['classification'],
        'num_classes': [1]
    }
    best_dir, best_val_loss, val_losses = run_grid_search(architecture_name_, architecture_params_options_,
                                                          train_params_)

    run_grid_search(architecture_name_, architecture_params_options_, train_params_)

# When the best model is found (after performing all the tests, etc) and it is to become a production model, these 3
# files should be copied to the folder ""
