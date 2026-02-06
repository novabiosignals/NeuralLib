import json
import datetime
import time
import os
import torch
import glob
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import importlib.util
from torch.utils.data import DataLoader
from NeuralLib.utils import configure_seed, configure_device, DatasetSequence, collate_fn, LossPlotCallback, \
    save_predictions_with_filename
from NeuralLib.config import RESULTS_BASE_DIR
from calflops import calculate_flops


def get_weights_and_info_from_checkpoints(prev_checkpoints_dir):
    """
    Extract weights (.pth) and training history from a checkpoint directory.
    If a .pth file exists, it will be used. Otherwise, it will be created from a .ckpt file.
    """
    # Extract training info
    training_info_path = os.path.join(prev_checkpoints_dir, 'training_info.json')
    if not os.path.exists(training_info_path):
        raise FileNotFoundError(f"Training info file not found in {prev_checkpoints_dir}")
    with open(training_info_path, 'r') as f:
        training_history = json.load(f)

    # Check for existing .pth file
    weights_pth = os.path.join(prev_checkpoints_dir, 'model_weights.pth')
    if os.path.exists(weights_pth):
        print(f"Found existing .pth file: {weights_pth}")
        return weights_pth, training_history

    # If no .pth file exists, convert .ckpt to .pth
    ckpt_file = glob.glob(os.path.join(prev_checkpoints_dir, '*.ckpt'))
    if not ckpt_file:
        raise FileNotFoundError(f"No checkpoint (.ckpt) file found in {prev_checkpoints_dir}")
    ckpt_file = ckpt_file[0]  # Assume the first .ckpt file is the one to use

    # Load checkpoint and save state_dict as .pth
    try:
        checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=True)
        torch.save(checkpoint['state_dict'], weights_pth)
        print(f"Converted .ckpt to .pth: {weights_pth}")
    except Exception as e:
        raise RuntimeError(f"Error converting .ckpt to .pth: {str(e)}")

    return weights_pth, training_history


def get_weights_and_info_from_hugging(hugging_face_model, local_dir=None):
    """
    Download weights and training history from Hugging Face.
    :param local_dir: local directory where the downloaded files from hugging face are stored
    :param hugging_face_model: Repository ID on Hugging Face (e.g., 'username/model_name').
    :return: weights_pth, training_history
    """
    from huggingface_hub import snapshot_download
    if local_dir:
        snapshot_download(repo_id=hugging_face_model, local_dir=local_dir)
    else:
        local_dir = snapshot_download(repo_id=hugging_face_model)

    # Load weights
    weights_pth = os.path.join(local_dir, 'model_weights.pth')
    if not os.path.exists(weights_pth):
        raise FileNotFoundError(f"Weights file not found in Hugging Face model {hugging_face_model}")

    # Load training info
    training_info_path = os.path.join(local_dir, 'training_info.json')
    if not os.path.exists(training_info_path):
        raise FileNotFoundError(f"Training info file not found in Hugging Face model {hugging_face_model}")
    with open(training_info_path, 'r') as f:
        training_history = json.load(f)

    return weights_pth, training_history


def get_hparams_from_checkpoints(checkpoints_directory):
    """
    Extract the architectures' hyperparameters from the latest version of hparams.yaml in the checkpoints directory.
    :param checkpoints_directory: Directory where the checkpoints and logs are stored.
    :return: Dictionary of hyperparameters.
    """
    lightning_logs_dir = os.path.join(checkpoints_directory, 'lightning_logs')
    tensorboard_logs_dir = os.path.join(checkpoints_directory, 'tensorboard_logs')
    logs_dir = lightning_logs_dir if os.path.exists(lightning_logs_dir) else tensorboard_logs_dir
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"Logs directory not found in {logs_dir}.")

    # Get all version folders and sort by version number
    version_dirs = sorted(
        [d for d in os.listdir(logs_dir) if d.startswith('version_')],
        key=lambda x: int(x.split('_')[-1]),  # Extract version number
        reverse=True
    )
    if not version_dirs:
        raise FileNotFoundError(f"No version folders found in {logs_dir}")

    latest_version_dir = os.path.join(logs_dir, version_dirs[0])
    hparams_path = os.path.join(latest_version_dir, 'hparams.yaml')

    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hparams.yaml not found in {hparams_path}")
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)

    return hparams


def get_hparams_from_hugging(hugging_face_model):
    """
    Download the architectures' hyperparameters as dictionary from Hugging Face.
    :param hugging_face_model: Repository ID on Hugging Face (e.g., 'username/model_name').
    :return: hparams
    """
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=hugging_face_model)

    # Load hyperparameters
    hparams_path = os.path.join(local_dir, 'hparams.yaml')
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hparams.yaml file not found in Hugging Face model {hugging_face_model}")
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)

    return hparams


def validate_training_context(retraining, checkpoints_directory, hugging_face_model):
    if retraining:
        if not (checkpoints_directory or hugging_face_model):
            raise ValueError("For retraining, either checkpoints_directory or hugging_face_model must be provided.")
        elif checkpoints_directory and hugging_face_model:
            print("Warning: both a local checkpoints directory and a hugging facel model were provided. Local"
                  "checkpoints will be used.")
    else:
        if checkpoints_directory or hugging_face_model:
            raise ValueError("For training from scratch, no checkpoints or Hugging Face model should be provided.")


class Architecture(pl.LightningModule):
    def __init__(self, architecture_name):
        super(Architecture, self).__init__()
        self.architecture_name = architecture_name
        self.training_info = {}  # training_info if training_info else {}
        self.checkpoints_directory = 'Directory not available.'  # this is the new checkpoints_directory

    def create_checkpoints_directory(self, retraining):
        """
        Creates a directory where the model's checkpoints will be saved during training
        :return: checkpoints_directory
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if retraining:
            retrain_str = '_retraining'
        else:
            retrain_str = ''
        if self.bidirectional:
            arch_string = f"{self.hid_dim}hid_{self.n_layers}l_bidir{self.bidirectional}_lr{self.learning_rate}_drop{self.dropout}{retrain_str}"
        else:
            arch_string = f"{self.hid_dim}hid_{self.n_layers}l_lr{self.learning_rate}_drop{self.dropout}{retrain_str}"
        dir_name = f"{self.architecture_name}_{arch_string}_dt{timestamp}"
        checkpoints_directory = os.path.join(RESULTS_BASE_DIR, self.model_name, 'checkpoints', dir_name)
        os.makedirs(checkpoints_directory, exist_ok=True)
        self.checkpoints_directory = checkpoints_directory
        return checkpoints_directory

    def efficiency_metrics(self):
        """
        Return simple efficiency metrics without using calflops.
        Returns:
            tuple:
                - flops_per_sample (None): FLOPs computation skipped.
                - params (int): Total number of trainable parameters.
        """
        # Count total trainable parameters
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # FLOPs cannot be computed safely for models requiring extra inputs, return None
        flops_per_sample = None

        return flops_per_sample, params

    def save_training_information(self, trainer, optimizer, train_dataset_name, trained_for, val_loss,
                                  total_training_time, gpu_model, retraining, prev_training_history=None):
        """ Save information about the current training process and append previous training info to the history if
        retraining."""
        current_training_info = {
            'architecture': self.architecture_name,
            'model_name': self.model_name,
            'train_dataset': train_dataset_name,
            'task': trained_for,
            'gpu_model': gpu_model,
            'epochs': trainer.current_epoch,
            'optimizer': str(optimizer),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'validation_loss': val_loss,
            'training_time': total_training_time,
            'retraining': retraining,
            'efficiency_flops': self.efficiency_metrics()[0],
            'efficiency_params': self.efficiency_metrics()[1],
        }
        # Set the new training info as the main training_info
        self.training_info = current_training_info

        # If retraining, append previous training info to training history
        if retraining:
            if prev_training_history:
                # Initialize or update training history field
                self.training_info['training_history'] = prev_training_history
            else:
                print("Warning: Retraining started, but no previous training information is available. "
                      "Training history will not include prior details.")

    def save_training_information_to_file(self, directory):
        """Saves trained information to a JSON file."""
        training_info = self.training_info
        training_info_file = os.path.join(directory, 'training_info.json')

        if not os.path.exists(training_info_file):
            with open(training_info_file, 'w') as f:
                json.dump(training_info, f, indent=4)

    def train_from_scratch(self, path_x, path_y, patience, batch_size, epochs, gpu_id=None,
                           all_samples=False, samples=None, dataset_name=None, trained_for=None,
                           enable_tensorboard=False, min_max_norm_sig=False):
        """
        :param path_x: Path to training data (features).
        :param path_y: Path to training labels (targets).
        :param patience: Early stopping patience.
        :param batch_size: Training batch size.
        :param epochs: Number of training epochs.
        :param gpu_id: GPU to use.
        :param all_samples: Use all training samples.
        :param samples: Subset of samples for training.
        :param dataset_name: Name of the dataset.
        :param trained_for: Task the model is being trained for.
        :param classification: Boolean flag for classification tasks.
        :param enable_tensorboard: Enable TensorBoard logging.
        :param min_max_norm_sig: Boolean flag for performing minmax normalization to data signals from the dataset
        before passing them to the model.
        """

        # Configure seed and device
        configure_seed(42)
        device = configure_device(gpu_id)
        if isinstance(device, int):  # Checking if the device is a valid GPU ID
            gpu_model = torch.cuda.get_device_name(device)
        else:  # device='cpu'
            gpu_model = None

        # Check for TensorBoard availability
        tensorboard_available = importlib.util.find_spec("tensorboard") is not None
        if enable_tensorboard and not tensorboard_available:
            print("Warning: TensorBoard is not installed. Proceeding without TensorBoard logging.")
            enable_tensorboard = False

        if dataset_name is None:
            raise ValueError("You must provide the 'dataset_name' for tracking the model's training process.")
        if trained_for is None:
            raise ValueError("You must provide the 'trained_for' input with the task that this model is intended to "
                             "perform, for tracking the model's training process.")

        # Initialize the model
        model = self  # Model has Initialized as self

        self.create_checkpoints_directory(retraining=False)
        print(f"Checkpoints directory created at {self.checkpoints_directory}")

        seq2one = self.architecture_name.endswith('2one')
        classification = self.task == 'classification'
        # Datasets and Dataloaders
        train_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='train', all_samples=all_samples,
                                        samples=samples, seq2one=seq2one, min_max_norm_sig=min_max_norm_sig,
                                        classification=classification, multi_label=self.multi_label,
                                        n_features=self.n_features, num_classes=self.num_classes)
        val_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='val', all_samples=all_samples,
                                      samples=samples, seq2one=seq2one, min_max_norm_sig=min_max_norm_sig,
                                      classification=classification, multi_label=self.multi_label,
                                      n_features=self.n_features, num_classes=self.num_classes)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Calculate class weights based on the training dataset -- only for classification problems
        # if classification:
        #     class_weights = calculate_class_weights(train_dataset)
        # Convert class_weights to the GPU
        # class_weights = class_weights.to(device=device)

        # Define the model callbacks
        arch_string = f"{self.hid_dim}hid_{self.n_layers}l_lr{self.learning_rate}_drop{self.dropout}"
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.checkpoints_directory,
            filename=f"{self.architecture_name}_{arch_string}",
            save_top_k=1,
            mode='min'
        )

        # Define early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )

        # Loss plotting callback
        plot_callback = LossPlotCallback(save_path=os.path.join(self.checkpoints_directory, "loss_plot.png"))

        # Initialize Trainer with TensorBoardLogger if enabled
        if enable_tensorboard and tensorboard_available:
            logger = TensorBoardLogger(self.checkpoints_directory, name="tensorboard_logs")
            print(f"TensorBoard logs will be saved to {logger.log_dir}")
        else:
            logger = None  # No logging if TensorBoard is not available or enabled

        # Start training with PyTorch Lightning Trainer
        start_time = time.time()
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if device != 'cpu' else 'cpu',
            devices=[device] if device != 'cpu' else 1,
            default_root_dir=self.checkpoints_directory,
            callbacks=[checkpoint_callback, early_stopping_callback, plot_callback],
            logger=logger  # TensorBoard logger added if available
        )

        # Train the model
        print('Starting training...')
        trainer.fit(model, train_dataloader, val_dataloader)

        # Save training information
        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time:.2f} seconds")
        val_loss = checkpoint_callback.best_model_score.item()  # Best validation loss
        optimizer = model.configure_optimizers()[0][0]

        model.save_training_information(
            trainer=trainer,
            optimizer=optimizer,
            train_dataset_name=dataset_name,
            trained_for=trained_for,
            retraining=False,
            gpu_model=gpu_model,
            val_loss=val_loss,
            total_training_time=total_training_time
        )
        print(model.training_info)
        model.save_training_information_to_file(self.checkpoints_directory)

        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Training complete. Best_model_path: {best_checkpoint_path}")

        # After training, save weights as .pth
        final_weights_pth = os.path.join(self.checkpoints_directory, 'model_weights.pth')
        torch.save(self.state_dict(), final_weights_pth)
        print(f"Weights saved as {final_weights_pth}")

    def retrain(self, path_x, path_y, patience, batch_size, epochs, gpu_id=None, all_samples=False,
                samples=None, dataset_name=None, trained_for=None, enable_tensorboard=False,
                checkpoints_directory=None, hugging_face_model=None, min_max_norm_sig=False):

        validate_training_context(retraining=True, checkpoints_directory=checkpoints_directory,
                                  hugging_face_model=hugging_face_model)

        # Configure seed and device
        configure_seed(42)
        device = configure_device(gpu_id)
        if isinstance(device, int):  # Checking if the device is a valid GPU ID
            gpu_model = torch.cuda.get_device_name(device)
        else:  # device='cpu'
            gpu_model = None

        # Check for TensorBoard availability
        tensorboard_available = importlib.util.find_spec("tensorboard") is not None
        if enable_tensorboard and not tensorboard_available:
            print("Warning: TensorBoard is not installed. Proceeding without TensorBoard logging.")
            enable_tensorboard = False

        if dataset_name is None:
            raise ValueError("You must provide the 'dataset_name' for tracking the model's training process.")
        if trained_for is None:
            raise ValueError("You must provide the 'trained_for' input with the task that this model is intended to "
                             "perform, for tracking the model's training process.")

        # Load weights and previous training info
        if checkpoints_directory:
            weights_pth, prev_training_history = get_weights_and_info_from_checkpoints(checkpoints_directory)
        else:
            weights_pth, prev_training_history = get_weights_and_info_from_hugging(hugging_face_model)
        # prev_training_history = json.load(prev_training_pth)

        # Initialize the model
        model = self  # same as doing: model = GRUseq2seq(n_features=self.n_features, hid_dim=self.hid_dim,...)
        # Load weights into the model
        state_dict = torch.load(weights_pth, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict)
        print(f"Weights loaded successfully from {weights_pth}")

        # Create new checkpoints directory
        self.create_checkpoints_directory(retraining=True)

        seq2one = self.architecture_name.endswith('2one')
        classification = self.task == 'classification'
        
        # Datasets and Dataloaders
        train_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='train', all_samples=all_samples,
                                        samples=samples, seq2one=seq2one, min_max_norm_sig=min_max_norm_sig,
                                        classification=classification, multi_label=self.multi_label,
                                        n_features=self.n_features, num_classes=self.num_classes)
        val_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='val', all_samples=all_samples,
                                      samples=samples, seq2one=seq2one, min_max_norm_sig=min_max_norm_sig,
                                      classification=classification, multi_label=self.multi_label,
                                      n_features=self.n_features, num_classes=self.num_classes)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        # collate_fn is necessary for handling signals with different lenghts (they are padded per batch)

        # Define the model callbacks
        arch_string = f"{self.hid_dim}hid_{self.n_layers}l_lr{self.learning_rate}_drop{self.dropout}_retraining"
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.checkpoints_directory,
            filename=f"{self.architecture_name}_{arch_string}",
            save_top_k=1,
            mode='min'
        )

        # Define early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )

        # Loss plotting callback
        plot_callback = LossPlotCallback(save_path=os.path.join(self.checkpoints_directory, "loss_plot.png"))

        # Initialize Trainer with TensorBoardLogger if enabled
        if enable_tensorboard and tensorboard_available:
            logger = TensorBoardLogger(self.checkpoints_directory, name="tensorboard_logs")
            print(f"TensorBoard logs will be saved to {logger.log_dir}")
        else:
            logger = None  # No logging if TensorBoard is not available or enabled

        # Start training with PyTorch Lightning Trainer
        start_time = time.time()
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if device != 'cpu' else 'cpu',
            devices=[device] if device != 'cpu' else 1,
            default_root_dir=self.checkpoints_directory,
            callbacks=[checkpoint_callback, early_stopping_callback, plot_callback],
            logger=logger  # TensorBoard logger added if available
        )

        # Train the model
        trainer.fit(model, train_dataloader, val_dataloader)

        # Save training information
        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time:.2f} seconds")
        val_loss = checkpoint_callback.best_model_score.item()  # Best validation loss
        optimizer = model.configure_optimizers()[0][0]

        model.save_training_information(
            trainer=trainer,
            optimizer=optimizer,
            train_dataset_name=dataset_name,
            trained_for=trained_for,
            gpu_model=gpu_model,
            val_loss=val_loss,
            total_training_time=total_training_time,
            retraining=True,
            prev_training_history=prev_training_history,
        )
        print(model.training_info)
        model.save_training_information_to_file(self.checkpoints_directory)

        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Training complete. Best_model_path: {best_checkpoint_path}")

        # After training, save weights as .pth
        final_weights_pth = os.path.join(self.checkpoints_directory, 'model_weights.pth')
        torch.save(self.state_dict(), final_weights_pth)
        print(f"Weights saved as {final_weights_pth}")

    def test_on_test_set(self, path_x, path_y, all_samples=True, samples=None, checkpoints_dir=None, gpu_id=None,
                         save_predictions=False, post_process_fn=None, min_max_norm_sig=False):
        """
        Evaluate the model on a test set.

        :param path_x: Path to input signals (features).
        :param path_y: Path to ground truth labels (targets).
        :param all_samples: If True, uses all test samples. Defaults to True.
        :param samples: Specific samples to include in testing (if any).
        :param checkpoints_dir: Directory of saved model checkpoints. Required if self.checkpoints_directory is not set.
        :param gpu_id: ID of GPU to use. Defaults to CPU if None.
        :param save_predictions: If True, saves the predictions to a directory.
        :param post_process_fn: post-processing function to apply to the output signal
        :return: List of predictions and average test loss.
        """
        # Configure device
        device = configure_device(gpu_id)
        map_location = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
        print(f"Using device: {map_location}")

        seq2one = self.architecture_name.endswith('2one')
        classification = self.task == 'classification'
        test_dataset = DatasetSequence(path_x=path_x, path_y=path_y, part='test', all_samples=all_samples,
                                       samples=samples, seq2one=seq2one, min_max_norm_sig=min_max_norm_sig,
                                       classification=classification, multi_label=self.multi_label,
                                       n_features=self.n_features, num_classes=self.num_classes)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        checkpoints_dir = checkpoints_dir or self.checkpoints_directory
        if not checkpoints_dir:
            raise ValueError("Please provide a valid checkpoints directory to load the model's weights.")

        # Load weights
        weights_pth = os.path.join(checkpoints_dir, 'model_weights.pth')
        if not os.path.exists(weights_pth):
            raise FileNotFoundError(f"Model weights not found in {weights_pth}.")
        weights = torch.load(weights_pth, map_location=map_location, weights_only=True)
        # print(f"weights:{weights}")
        # print(f"device:{device}, map location:{str(map_location)}")
        self.load_state_dict(weights)
        print(f"Weights successfully loaded from {weights_pth}.")

        model = self
        model.to(device)
        model.eval()

        if save_predictions:
            # Create a directory inside checkpoints to save predictions
            predictions_dir = os.path.join(checkpoints_dir, 'predictions')
            os.makedirs(predictions_dir, exist_ok=True)

        # Run inference
        total_loss = 0.0
        predictions = []
        idx = 0
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # Create lengths as a 1D CPU tensor
                length = torch.tensor([x.size(1)], dtype=torch.int64, device='cpu')
                output = self(x, length)
                loss = self.criterion(output, y)
                total_loss += loss.item()

                # Apply softmax **only for multiclass classification** when storing predictions
                if self.task == 'classification' and not self.multi_label:
                    output = torch.nn.functional.softmax(output, dim=-1)

                # Apply post-processing if provided
                processed_output = post_process_fn(output) if post_process_fn else output
                if isinstance(processed_output, torch.Tensor):
                    predictions.append(processed_output.cpu())
                else:
                    predictions.append(processed_output)

                # Save predictions if required
                if save_predictions:
                    input_file_name = os.path.basename(test_dataset.files_x[idx])
                    save_predictions_with_filename(processed_output, input_file_name, predictions_dir)

            print(f"Sample {idx}: Test Loss: {loss.item():.4f}")
            idx += 1

        avg_loss = total_loss / len(test_dataloader)

        print(f"Average Test Loss: {avg_loss:.4f}")

        # TODO: generate a report of the model performance so that it can be uploaded to hugging face
        # have into account the task (classification vs regression and output dimension)

        return predictions, avg_loss

    def test_on_single_signal(self, X, checkpoints_dir=None, gpu_id=None, post_process_fn=None):
        """
        Evaluate the model on a single input signal.
        :param X: Input signal tensor of shape [seq_len, n_features].
        :param checkpoints_dir: Directory of saved model checkpoints. Required if self.checkpoints_directory is not set.
        :param gpu_id: ID of GPU to use. Defaults to CPU if None.
        :param post_process_fn: post-processing function to apply to the output signal
        :return: Predicted output.
        """
        # Configure device
        device = configure_device(gpu_id)
        map_location = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
        print(f"Using device: {map_location}")

        checkpoints_dir = checkpoints_dir or self.checkpoints_directory
        if not checkpoints_dir:
            raise ValueError("Please provide a valid checkpoints directory to load the model's weights.")

        # Load weights
        weights_pth = os.path.join(checkpoints_dir, 'model_weights.pth')
        if not os.path.exists(weights_pth):
            raise FileNotFoundError(f"Model weights not found in {weights_pth}.")
        self.load_state_dict(torch.load(weights_pth, map_location=map_location, weights_only=True))
        print(f"Weights successfully loaded from {weights_pth}.")

        model = self
        model.to(device)
        model.eval()

        X = X.unsqueeze(0).to(device)  # Add batch dimension
        length = torch.tensor([X.size(1)], dtype=torch.int64, device='cpu')  # Sequence length for batch size 1

        with torch.no_grad():
            output = self(X, length)

        # Apply softmax **only for multiclass classification** when storing predictions
        if self.task == 'classification' and not self.multi_label:
            output = torch.nn.functional.softmax(output, dim=-1)

        # Apply post-processing if provided
        processed_output = post_process_fn(output) if post_process_fn else output

        return processed_output.cpu()
