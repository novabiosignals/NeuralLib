import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import sklearn.preprocessing as pp


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def configure_device(gpu_id=None):
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)} (GPU ID: {gpu_id})")
        return gpu_id
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)  # Default to the first GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (GPU ID: 0)")
        return 0
    else:
        print("No GPU available, using CPU.")
        return 'cpu'


def list_gpus():
    """ List the names of available gpus. """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"GPU ID: {i}, Model: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available.")


# é preciso handle o facto de estarmos a ver ponto a ponto do sinal e nao idx a idx do dataset
def calculate_class_weights(dataset):
    '''
    todo: ALTERAR class weights
    :param dataset:
    :return:
    '''
    labels = []
    for _, label in dataset:
        labels.append(label)

    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()

    return class_weights


def save_model_results(model, results_dir, model_name, best_val_loss):
    checkpoints_dir = os.path.join(results_dir, 'checkpoints', model_name)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Save model hyperparameters and results
    results = {
        'hyperparameters': {
            'n_features': model.n_features,
            'hid_dim': model.hid_dim,
            'n_layers': model.n_layers,
            'dropout': model.dropout,
            'learning_rate': model.learning_rate,
            'bidirectional': model.bidirectional,
        },
        'best_validation_loss': best_val_loss,
        'best_epoch': model.trainer.current_epoch
    }

    results_file = os.path.join(checkpoints_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)


def save_predictions(predictions, batch_idx, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_path = os.path.join(dir, f"predictions_batch_{batch_idx}.npy")
    np.save(file_path, np.array(predictions))


def save_predictions_with_filename(predictions, input_filename, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Use the input filename to name the prediction file
    file_name_without_ext = os.path.splitext(input_filename)[0]
    file_path = os.path.join(dir, f"{file_name_without_ext}.npy")

    # Save the predictions
    np.save(file_path, np.array(predictions))


def collate_fn(batch):
    # Separate inputs (X) and labels (Y)
    X = [item[0] for item in batch]  # List of input sequences
    Y = [item[1] for item in batch]  # List of labels

    # Sort by the length of X in descending order
    X_lengths = [len(x) for x in X]
    sorted_indices = sorted(range(len(X_lengths)), key=lambda i: X_lengths[i], reverse=True)

    X = [X[i] for i in sorted_indices]
    Y = [Y[i] for i in sorted_indices]
    X_lengths = [X_lengths[i] for i in sorted_indices]

    # Pad X
    X_padded = pad_sequence(X, batch_first=True)

    if isinstance(Y[0], torch.Tensor) and Y[0].ndim == 0:  # **Case: Scalar multiclass classification**
        Y_padded = torch.tensor(Y, dtype=torch.long)  # Just make it a tensor
    elif isinstance(Y[0], torch.Tensor) and Y[0].ndim > 0 and Y[0].shape[0] == 1:  # **Case: seq2one with a vector**
        Y_padded = torch.stack(Y)  # Just stack the values without padding
    else:  # **Case: Sequence-to-sequence**
        Y_padded = pad_sequence(Y, batch_first=True)

    # Convert X_lengths to a tensor (fix for previous issue)
    X_lengths = torch.tensor(X_lengths, dtype=torch.long, device=X_padded.device)

    return X_padded, Y_padded, X_lengths


def validate_dataset(self):
    """
    Validates the dataset structure and shapes according to task and, if appropriate, classification type and
    multi-label settings.
    Raises errors if the dataset format does not match expected input/output shapes.
    """

    # Load a sample to check dimensions
    sample_x = np.load(os.path.join(self.dir_x, self.files_x[0]))
    sample_y = np.load(os.path.join(self.dir_y, self.files_y[0]))

    # Validate X shape (should always be [seq_len, num_features])
    if sample_x.ndim == 1:
        print(f"Warning: Input shape {sample_x.shape} should be (sequence length, {self.n_features}). "
              f"It will be automatically converted, but consider storing data in the correct format for efficiency."
              )
    elif sample_x.ndim == 2:
        # Ensure shape follows (seq_len, num_features) and not the other way around
        if sample_x.shape[0] == self.n_features and sample_x.shape[0] < sample_x.shape[1]:  # Likely transposed
            print(
                f"Warning: Input shape seems transposed {sample_x.shape}. It will be corrected but consider storing "
                f"data as (sequence length, {self.n_features}) for efficiency."
                )
        elif sample_x.shape[0] != self.n_features and sample_x.shape[1] != self.n_features:
            raise ValueError(f"Invalid input shape {sample_x.shape}. Expected (sequence length, {self.n_features}).")
    else:
        raise ValueError(f"Invalid input shape {sample_x.shape}. Expected (sequence length, {self.n_features}).")

    # Validate Y shape based on classification, multi_label, seq2one, and num_classes
    if self.seq2one:
        if self.classification:
            if self.multi_label:  # Multi-label classification (BCEWithLogitsLoss)
                if sample_y.shape == (self.num_classes,):  # (num_classes,) → Expected (1, num_classes)
                    print(f"Warning: Output shape {sample_y.shape} should be (1, {self.num_classes}). It will be "
                          f"automatically reshaped, but consider storing it correctly  for efficiency.")
                elif sample_y.shape != (1, self.num_classes):
                    raise ValueError(f"Invalid output shape {sample_y.shape}. Expected (1, {self.num_classes}) "
                                     f"for seq2one multi-label classification.")
            else:  # Multiclass classification (CrossEntropyLoss)
                if sample_y.shape == (1,):  # (1,) → Expected scalar
                    print(f"Warning: Output shape {sample_y.shape} should be a scalar. It will be automatically "
                          f"converted, but consider storing it correctly for efficiency.")
                elif sample_y.shape != ():
                    raise ValueError(f"Invalid output shape {sample_y.shape}. Expected a scalar value for seq2one "
                                     f"multiclass classification.")
        else:  # Regression (MSELoss)
            if sample_y.shape == (self.n_features,):  # (num_features,) → Expected (1, num_features)
                print(f"Warning: Output shape {sample_y.shape} should be (1, {self.n_features}). "
                      "It will be automatically converted, but consider storing data in the correct format for "
                      "efficiency.")
            elif sample_y.shape != (1, self.n_features):
                raise ValueError(
                    f"Invalid output shape {sample_y.shape}. Expected (1, {self.n_features}) for seq2one regression.")
    else:  # Sequence-to-sequence or encoder-decoder
        if self.classification:
            if self.multi_label:  # Multi-label classification (BCEWithLogitsLoss) [seq_len, num_classes]
                if sample_y.ndim == 1 and self.num_classes == 1:  # (seq_len,) → Expected (seq_len, 1)
                    print(f"Warning: Output shape {sample_y.shape} should be (sequence length,1). "
                          "It will be automatically converted, but consider storing data in the correct format for "
                          "efficiency.")
                elif sample_y.ndim != 2 or sample_y.shape[1] != self.num_classes:
                    raise ValueError(
                        f"Invalid output shape {sample_y.shape}. Expected (sequence length, {self.num_classes}) for "
                        f"seq2seq multi-label classification. "
                    )
            else:  # Multiclass classification (CrossEntropyLoss)
                if sample_y.ndim == 2 and sample_y.shape[1] == 1:  # (seq_len, 1) → Expected (seq_len,)
                    print(f"Warning: Output shape {sample_y.shape} should be (sequence length,). "
                          "It will be automatically converted, but consider storing data in the correct format for "
                          "efficiency. "
                          )
                elif sample_y.ndim != 1:
                    raise ValueError(
                        f"Invalid output shape {sample_y.shape}. Expected (sequence length,) for seq2seq multiclass "
                        f"classification. "
                    )
        else:  # Regression (MSELoss)
            if sample_y.ndim == 1 and sample_y.shape[0] > 1:  # (seq_len,) → Expected (seq_len, num_features)
                print(f"Warning: Output shape {sample_y.shape} should be (sequence length, {self.n_features}). "
                      "It will be automatically converted, but consider storing data in the correct format for "
                      "efficiency. "
                      )
            elif sample_y.ndim != 2:
                raise ValueError(
                    f"Invalid output shape {sample_y.shape}. Expected (sequence length, {self.n_features}) for "
                    f"seq2seq regression. "
                )

    print("✅ Dataset validation passed.")


class DatasetSequence(Dataset):

    def __init__(self, path_x, path_y, part='train', all_samples=False, samples=None, seq2one=False, n_features=1,
                 classification=True, multi_label=False, num_classes=1, min_max_norm_sig=False, window_size=None,
                 overlap=None):
        """
        :param path_x: (str) Path to the directory containing input `.npy` files.
        :param path_y: (str) Path to the directory containing output `.npy` files.
        :param part: (str) Dataset partition to use, one of ['train', 'val', 'test']. Defaults to 'train'.
        :param all_samples: (bool) If True, uses all available samples; otherwise, uses a limited number of samples.
        :param samples: (int, optional) Number of samples to use if `all_samples=False`. Must be provided when `all_samples=False`.
        :param seq2one: (bool) If True, the dataset is sequence-to-one (e.g., classification or regression with one output per sequence).
                        If False, the dataset is sequence-to-sequence (e.g., denoising or forecasting tasks where output is a full sequence).
        :param min_max_norm_sig: (bool) Whether to apply Min-Max normalization to input/output signals.
                                 This is useful if input signals are not pre-processed beforehand.
        :param window_size: (int, optional) Not yet implemented - Placeholder for defining windowing functionality.
        :param overlap: (float, optional) Not yet implemented - Placeholder for defining overlap percentage when applying windowing.
        """
        self.dir_x = os.path.join(path_x, part)  # parts: train, val, test
        self.dir_y = os.path.join(path_y, part)
        self.all_samples = all_samples
        self.samples = samples
        self.min_max_norm_sig = min_max_norm_sig
        self.seq2one = seq2one
        self.classification = classification
        self.multi_label = multi_label
        self.n_features = n_features
        self.num_classes = num_classes
        self.window_size = window_size  # still to be done
        self.overlap = overlap  # still to be done

        # Check if directories exist
        if not os.path.isdir(self.dir_x):
            print(f"Error: Directory {self.dir_x} does not exist.")
        if not os.path.isdir(self.dir_y):
            print(f"Error: Directory {self.dir_y} does not exist.")

        # Check if there are any .npy files in the directories
        self.files_x = [f for f in os.listdir(self.dir_x) if f.endswith('.npy')]
        self.files_y = [f for f in os.listdir(self.dir_y) if f.endswith('.npy')]
        # print(self.files_x)
        # print(self.files_y)

        if len(self.files_x) == 0:
            print(f"Error: No .npy files found in {self.dir_x}.")
        if len(self.files_y) == 0:
            print(f"Error: No .npy files found in {self.dir_y}.")

        # Check if samples is an integer when all_samples is False
        if not self.all_samples:
            if not isinstance(self.samples, int):
                raise ValueError("Error: The number of samples to be used should be provided.")
            elif self.samples > len(self.files_x):
                print(f"Warning: Requested {self.samples} samples, but only {len(self.files_x)} files are available.")
                self.samples = len(self.files_x)  # Adjust to the maximum available

        validate_dataset(self)

    def __len__(self):
        if self.all_samples:
            print('Using all data samples')
            return len(self.files_x)
        else:
            # print(f"Using {min(self.samples, len(self.files_x))} data samples")
            return min(self.samples, len(self.files_x))

    def __getitem__(self, idx):
        if idx >= len(self.files_x):
            print(f"Error: Index {idx} is out of bounds for the dataset with {len(self.files_x)} samples.")
            return None

        # Load the data
        x_path = os.path.join(self.dir_x, self.files_x[idx])
        y_path = os.path.join(self.dir_y, self.files_x[idx])  # files_x is correct. to make sure it is loading the file
        # with the same name for x and y.

        try:
            item_x = np.load(x_path)
        except Exception as e:
            print(f"Error loading file {x_path}: {e}")
            return None

        try:
            item_y = np.load(y_path)
        except Exception as e:
            print(f"Error loading file {y_path}: {e}")
            return None

        # print(f"x shape: {item_x.shape}")
        # print(f"y shape: {item_y.shape}")

        # Ensure item_x has shape (seq_len, num_features)
        if item_x.ndim == 1:
            item_x = item_x.reshape(-1, 1)  # Convert (seq_len,) to (seq_len, 1)
        elif item_x.ndim == 2 and item_x.shape[0] == self.n_features:
            # Check only if it is 2D and appears to be (num_features, seq_len)
            item_x = item_x.T  # Transpose to (seq_len, num_features)

        # Handle item_y based on its shape
        if self.seq2one:
            if item_y.ndim == 0:
                if self.classification and not self.multi_label:  # multiclass classification
                    item_y = int(item_y)  # CrossEntropyLoss expects a scalar
                else:  # regression or binary (multi-label) classification
                    item_y = np.array([[item_y]])  # Convert scalar to (1,1)
            elif item_y.ndim == 1:
                if self.classification and not self.multi_label:  # multiclass classification
                    item_y = item_y.item()  # Convert (1,) to ()
                else:  # regression or multiclass classification
                    item_y = item_y.reshape(1, -1)  # Convert (num_classes,) to (1, num_classes or num_features)
        else:  # Sequence-to-sequence tasks
            if item_y.ndim == 1:
                if self.classification and not self.multi_label:  # classification multiclass
                    pass
                else:  # regression or multi-label classification (1 class)
                    item_y = item_y.reshape(-1, 1)  # Convert (seq_len,) to (seq_len, 1)
            elif item_y.ndim == 2 and item_y.shape[0] < item_y.shape[1]:
                # Check only if it is 2D and appears to be (num_features, seq_len)
                item_y = item_y.T  # Transpose to (seq_len, num_features)

        if self.min_max_norm_sig:
            item_x = pp.minmax_scale(item_x)
            if not self.seq2one:
                item_y = pp.minmax_scale(item_y)

        # print(f"x shape: {item_x.shape}")
        # print(f"y shape: {item_y.shape}")

        if self.classification and self.seq2one and not self.multi_label:
            # For multiclass classification (seq2one), ensure y is an integer scalar
            print('seq2one multi-class classification')
            return torch.tensor(item_x, dtype=torch.float32), torch.tensor(item_y, dtype=torch.long)
        else:
            # For all other cases (regression, sequence-to-sequence, multilabel classification)
            return torch.tensor(item_x, dtype=torch.float32), torch.tensor(item_y, dtype=torch.float32)

        # return torch.tensor(item_x).float(), torch.tensor(item_y).float()
