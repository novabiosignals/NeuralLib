import NeuralLib.architectures as arc
from NeuralLib.config import HUGGING_MODELS_BASE_DIR
from NeuralLib.utils import configure_device
import torch
import json
import os
import yaml
import numpy as np
from huggingface_hub import snapshot_download, get_collection


class ProductionModel(arc.Architecture):
    """
    Production Models
    A class for trained models, extending Architecture by adding weights and training information.
    Includes methods for importing from Hugging Face.

    Models can be retrieved from:
    - The predefined Hugging Face collection: "NeuralLib: Deep Learning Models for Biosignals Processing".
    - Any public or private Hugging Face repository if explicitly specified.
    """

    DEFAULT_MODEL_REPOS = {
        "ECGPeakDetector": "marianaagdias/ECGPeakDetector",
        "ECGDenoiser": "marianaagdias/ECGDenoiser",
        # Add more predefined models here as needed
    }

    def __init__(self, model_name, hugging_repo=None):
        """
        Initialize the production model.

        :param model_name: (str) The name of the model to be loaded.
        :param hugging_repo: (str, optional) The full Hugging Face repository ID (e.g., "username/model_name").
            - If provided, the model will be loaded from this repository.
            - If not provided, the model will be searched in the NeuralLib collection.
            - The repository **must** contain the required files: `model_weights.pth`, `hparams.yaml`, `training_info.json`.
        """
        self.model_name = model_name
        self.local_dir = os.path.join(HUGGING_MODELS_BASE_DIR, self.model_name)

        if hugging_repo is not None:
            self.hugging_repo = hugging_repo
        else:
            try:
                self.hugging_repo = hugging_repo
            except KeyError:
                raise ValueError(
                    f"Model '{model_name}' not found in the DEFAULT_MODEL_REPOS registry. "
                    f"Please provide a valid Hugging Face repository (hugging_repo) "
                    f"containing the necessary files."
                )

        if not os.path.exists(self.local_dir):
            # Ensure model files are cached locally
            self._download_and_cache_files()
        else:
            print(f"Using cached model files at: {self.local_dir}")

        # Load model components
        self.weights_path = os.path.join(self.local_dir, "model_weights.pth")
        self.hparams_path = os.path.join(self.local_dir, "hparams.yaml")
        self.training_info_path = os.path.join(self.local_dir, "training_info.json")

        # Validate files self.weights_path, self.hparams_path, self.training_info_path
        self._validate_model_files()

        self.training_info = self._load_json(self.training_info_path)

        self.architecture_name = self.training_info['architecture']
        # Dynamically get the architecture class from biosignals_architectures
        self.architecture_class = getattr(arc, self.architecture_name, None)
        if not self.architecture_class:
            raise ValueError(f"Architecture {self.architecture_name} not found in biosignals_architectures.")

        super().__init__(architecture_name=self.architecture_name)  # initializing parent class

        # Initialize model state with weights
        self._initialize_model()


    def _download_and_cache_files(self):
        """Download model files from Hugging Face if not already cached locally."""
        try:
            print(f"Downloading model files for {self.model_name} from Hugging Face...")
            snapshot_download(repo_id=self.hugging_repo, local_dir=self.local_dir, max_workers=1,
                                resume_download=True)
            print(f"Model files saved to: {self.local_dir}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model files for {self.model_name}. "
                f"Ensure the Hugging Face repo '{self.hugging_repo}' exists and your internet connection is stable."
                f"Error: {e}")

    def _validate_model_files(self):
        """Ensure the required model files are present in the downloaded repository."""
        required_files = [self.weights_path, self.hparams_path, self.training_info_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            raise FileNotFoundError(
                f"The repository '{self.hugging_repo}' is missing the following required files:\n"
                + "\n".join(missing_files)
                + "\nEnsure that the repository contains the expected files before attempting to load the model."
            )
        
    def _infer_bidir_per_layer(self, state_dic, n_layers):
        """
        Infer per-layer GRU bidirectionality from a checkpoint state_dict.

        Parameters
        ----------
        state_dict : dict
            Checkpoint state_dict loaded from model_weights.pth.
        n_layers : int
            Number of GRU layers expected by the architecture hyperparameters.

        Returns
        -------
        bidir_per_layer : list[bool]
            A list of length n_layers. bidir_per_layer[i] is True if layer i has the full
            reverse-direction parameter set in the checkpoint, otherwise False.
        """
        bidir_layers = []
        
        for i in range(n_layers):
            reverse_keys = [
                f"gru_layers.{i}.weight_ih_l0_reverse",
                f"gru_layers.{i}.weight_hh_l0_reverse",
                f"gru_layers.{i}.bias_ih_l0_reverse",
                f"gru_layers.{i}.bias_hh_l0_reverse",
            ]
            
            present = [k in state_dic for k in reverse_keys]

            if all(present):
                bidir_layers.append(True)
            elif not any(present):
                bidir_layers.append(False)
            else:
                print(
                    f"Inconsistent reverse keys in layer {i}:"
                    f"{[k for k, p in zip(reverse_keys, present) if p]}"
                    )
                bidir_layers.append(False)  # default to False if inconsistent
        
        return bidir_layers
        
    def _initialize_model(self):
        """Dynamically initialize the model with hyperparameters and load weights."""
        # Load hyperparameters
        self.hyperparams = self._load_yaml(self.hparams_path)
        self.task = self.hyperparams['task']
        self.multi_label = self.hyperparams['multi_label']

        # Load checkpoint first
        state_dict = torch.load(self.weights_path, map_location='cpu')

        # Infer per-layer bidirectionality from checkpoint
        bidir_per_layer = self._infer_bidir_per_layer(
            state_dict, self.hyperparams['n_layers']
        )

        self.hyperparams['bidir_per_layer'] = bidir_per_layer

        print("Inferred bidirectionality from checkpoint:")
        for i, b in enumerate(bidir_per_layer):
            print(f"  Layer {i}: {'bidirectional' if b else 'unidirectional'}")

        # Dynamically initialize the architecture
        self.model = self.architecture_class(**self.hyperparams)
        self.model.training_info = self.training_info

        # Load weights
        incompat = self.model.load_state_dict(state_dict, strict=False)

        if incompat.missing_keys or incompat.unexpected_keys:
            print(f"Warning: Incompatible keys when loading {self.model_name}:")
            if incompat.missing_keys:
                print("   Missing keys:", incompat.missing_keys)
            if incompat.unexpected_keys:
                print("   Unexpected keys:", incompat.unexpected_keys)

        self.model.eval()  # Set model to evaluation mode
        print(f"{self.model_name} successfully initialized.")

    def predict(self, X, gpu_id=None, post_process_fn=None, **post_process_kwargs):
        """
        Run inference on input data.
        :param X: Input tensor of shape [batch_size, seq_len, features].
        :param gpu_id: GPU ID to use for inference (if applicable).
        :param post_process_fn: Optional function for post-processing predictions.
        :return: Model predictions (post-processed if a function is provided).
        """
        # Configure device
        device = configure_device(gpu_id)
        map_location = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
        print(f"Using device: {map_location}")

        # Ensure the model is on the correct device
        self.model.to(map_location)

        # Convert NumPy array to PyTorch tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor

        # Ensure input is a PyTorch tensor
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a PyTorch Tensor or NumPy array.")

        # Handle different tensor shapes
        if X.dim() == 1:
            # Case 1: 1D time series → [seq_len] → [1, seq_len, 1]
            X = X.unsqueeze(0).unsqueeze(-1)
        elif X.dim() == 2:
            # Case 2: [seq_len, features] → [1, seq_len, features]
            X = X.unsqueeze(0)
        elif X.dim() == 3:
            # Case 3: Correct shape [batch_size, seq_len, features]
            pass
        else:
            # Invalid shape
            raise ValueError(
                "Input X must have dimensions [batch_size, seq_len, features], "
                "[seq_len, features], or [seq_len]"
            )

        # Ensure input data is on the correct device
        X = X.to(map_location)

        lengths = [X.size(1)]  # Sequence length for batch size 1

        with torch.no_grad():
            output = self.model(X, lengths)  # go through the forward method of architecture

        # Apply softmax **only for multiclass classification** when storing predictions
        if self.task == 'classification' and not self.multi_label:
            output = torch.nn.functional.softmax(output, dim=-1)

        # Apply post-processing if provided
        processed_output = post_process_fn(output, **post_process_kwargs) if post_process_fn else output.cpu().numpy()

        return processed_output

    @staticmethod
    def _load_json(file_path):
        """Load JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_yaml(file_path):
        """Load YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)


def list_production_models():
    """
    Lists all models in the NeuralLib Hugging Face collection.
    """
    
    collection_id = "novabiosignals/neurallib-deep-learning-models-for-biosignals-processing-6813ee129bc1bba8210b6948"  # "marianaagdias/neurallib-deep-learning-models-for-biosignals-processing-67473f72e30e1f0874ec5ebe"
    print(f"Models in NeuralLib collection ({collection_id}):")
    collection = get_collection(collection_id)
    for item in collection.items:
        print(item.item_id.split("/")[-1])


def build_tl_arch_config(
    prod_model,
    reuse_n_gru_layers=None, 
    extra_hid_dims=None,
    bidir_per_layer=None,
    learning_rate=1e-3,
    model_name_suffix="_TL_PeakDetector",
    task="classification",
    num_classes=1,
    multi_label=True,
    fc_out_bool=True,
):
    """
    Build TL architecture hyperparameters based on a pretrained ProductionModel.

    Args:
        prod_model: ProductionModel instance (already loaded).
        reuse_n_gru_layers:
                - None: reuse all pretrained layers
                - int: reuse the first n layers
        extra_hid_dims: list[int], extra hidden dims to append as new GRU layers after reuse layers.
        bidir_per_layer: List[bool], bidirectionality for all layers (reused + extra).
                - If None, reuse the bidirectionality of the pretrained layers and set Same for extra layers.
        learning_rate: float, learning rate for TL model.
        model_name_suffix: suffix added to prod_model.model_name for the TL model name.
        task: 'classification' or 'regression'.
        num_classes: number of classes (1 for binary with BCEWithLogits).
        multi_label: True for BCEWithLogits, False for softmax CrossEntropy.
        fc_out_bool: whether to include a fully-connected output head.

    Returns:
        arch_params: dict of hyperparameters for TLModel/GRUseq2seq.
        reuse_n:     int, actual number of reused GRU layers (clamped to available ones).
    """

    base_hparams = prod_model.hyperparams
    base_hid_dims = base_hparams["hid_dim"]
    base_n_layers = base_hparams["n_layers"]

    # Clamp reuse_n to available layers
    if extra_hid_dims is None:
        extra_hid_dims = []
    
    if reuse_n_gru_layers is None:
        reuse_n = base_n_layers
    else:
        reuse_n = min(reuse_n_gru_layers, base_n_layers)
    
    # Reused part of the encoder
    reused_hid_dims = base_hid_dims[:reuse_n]

    # Final GRU hidden dims = reused + extra
    final_hid_dims = reused_hid_dims + list(extra_hid_dims)
    final_n_layers = len(final_hid_dims)
    
    arch_params = {
        "model_name": f"{prod_model.model_name}{model_name_suffix}",
        "n_features": base_hparams["n_features"],
        "hid_dim": final_hid_dims,
        "bidir_per_layer": bidir_per_layer,
        "n_layers": final_n_layers,
        "dropout": base_hparams["dropout"],
        "learning_rate": learning_rate,
        "bidirectional": base_hparams["bidirectional"],
        "task": task,
        "num_classes": num_classes,
        "multi_label": multi_label,
        "fc_out_bool": fc_out_bool,
    }

    return arch_params, reuse_n


def build_pretrained_layer_map(prod_model, reuse_n):
    """
    Build a layer_mapping dict for TLModel.inject_weights, reusing GRU layers [0..reuse_n-1].

    Args:
        prod_model: ProductionModel instance.
        reuse_n:    int, how many GRU layers to reuse.

    Returns:
        layer_mapping: dict like {'gru_layers.0': state_dict, 'gru_layers.1': state_dict, ...}
    """
    layer_mapping = {}
    for i in range(reuse_n):
        try:
            src_layer = prod_model.model.gru_layers[i]
            layer_mapping[f"gru_layers.{i}"] = src_layer.state_dict()
            print(f"[TL] Reusing pretrained layer {i}.")
        except (AttributeError, IndexError):
            print(f"[TL] Cannot reuse layer {i} (layer does not exist). "
                  f"Stopping at {i} reused layers.")
            # if pretrained model has fewer layers than requested, stop
            break
    print(f"[TL] Total reused GRU layers: {len(layer_mapping)}.")
    return layer_mapping


def build_freeze_phase(tl_model, reuse_n, strategy: str = "all", unfreeze_depth: int=1):
    """
    Generate a freeze / unfreeze list

    Args:
        - tl_model: TLModel Instance
        - reuse_n: reuse encoder GRU layers
        strategy:
            - "last_only": only unfreeze the last reuse layer
            - "all": unfreeze all the reuse layer
            - "gradual" unfreeze the last "unfreeze_depth" layers
            - "new_only": only unfreeze the new layers, freeze all reuse layers
        unfreeze_depth:
            - only use when strategy="gradual", such as:
                reuse_n=3, unfreeze_depth = 2
            -> unfreeze gru_layers.1 and gru_layers.2, freeze gry_laters.0
    
    Return:
        freeze_layers, unfreeze_layers
    """

    # Get the number of TL model layers
    n_layers = tl_model.model.n_layers

    reuse_n = min(reuse_n, n_layers)

    freeze_layers = []
    unfreeze_layers = []

    # freezing strategy for encoder
    if strategy == "all":
        encoder_unfreeze_start = 0
    elif strategy == "last_only":
        if reuse_n > 0:
            encoder_unfreeze_start = reuse_n - 1
        else:
            encoder_unfreeze_start = 0
    elif strategy == "gradual":
        if unfreeze_depth > 0:
            encoder_unfreeze_start = max(0, reuse_n - unfreeze_depth)
        else:
            encoder_unfreeze_start = 0
    elif strategy == "new_only":
        encoder_unfreeze_start = reuse_n  # freeze all reused layers
    else:
        raise ValueError(f"Unknown strategy '{strategy}'."
                         f"Use 'last_only', 'all', 'gradual'.")
    
    # [0, encoder_unfreeze_start] -> freeze
    # [encoder_unfreeze_start, reuse_n] -> unfreeze
    for i in range(reuse_n):
        layer_name = f"gru_layers.{i}"
        if i < encoder_unfreeze_start:
            freeze_layers.append(layer_name)
        else:
            unfreeze_layers.append(layer_name)
        
    # Unfreeze all new adding layers
    for i in range(reuse_n, n_layers):
        unfreeze_layers.append(f"gru_layers.{i}")
    
    # if fc_out in TL_model, unfreeze
    if hasattr(tl_model.model, "fc_out"):
        unfreeze_layers.append("fc_out")

    # Deduplicate the layers and keep order
    def _unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    freeze_layers = _unique(freeze_layers)
    unfreeze_layers = _unique(unfreeze_layers)
    
    print(f"  strategy = {strategy}, reuse_n = {reuse_n}, n_layers = {n_layers}")
    print("  freeze_layers:   ", freeze_layers)
    print("  unfreeze_layers: ", unfreeze_layers)
    print()

    return freeze_layers, unfreeze_layers