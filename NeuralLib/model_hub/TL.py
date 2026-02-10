import NeuralLib.architectures as arc
from NeuralLib.model_hub.production_model import ProductionModel


class TLModel(arc.Architecture):
    """
    Transfer Learning Model that supports layer-wise weight injection and flexible training strategies.
    """
    def __init__(self, architecture_name, **kwargs):
        """
        Initialize the architecture just like any other model.
        :param architecture_class: The architecture class (e.g., GRUseq2seq).
        :param kwargs: Architecture hyperparameters.
        """
        # Normalize architecture_name
        if isinstance(architecture_name, str):
            arch_name = architecture_name
            architecture_class = getattr(arc, arch_name, None)
            if architecture_class is None:
                raise ValueError(f"Architecture {arch_name} not found in biosignals_architectures.")
        else:
            # assume it's a class
            architecture_class = architecture_name
            arch_name = architecture_class.__name__

        super().__init__(architecture_name=architecture_name)
        self.architecture_name = arch_name
        self.architecture_class = architecture_class

        self.hyperparams = kwargs  # model_name is part of the biosignals_architecture kwargs
        self.model = self.architecture_class(**self.hyperparams)
        
        # Dynamically set kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        print(f"TransferLearningModel initialized with architecture: {self.architecture_class.__name__}")

    def inject_weights(self, layer_mapping):
        """
        Inject weights into specific layers of the model.
        :param layer_mapping: Dict mapping target layers in TLModel to source state_dicts.
                              Example: {'model.gru_layers.0': source_GRU_layer.state_dict()}
        """
        print("Injecting weights into TLModel layers...")
        injected = []
        failed = []

        for layer_name, source_state_dict in layer_mapping.items():
            try:
                # Navigate to the target layer
                parts = layer_name.split('.')
                current_attr = self.model
                for part in parts:
                    if part.isdigit():
                        current_attr = current_attr[int(part)]
                    else:
                        current_attr = getattr(current_attr, part)

                    # Load the state_dict into the target layer
                current_attr.load_state_dict(source_state_dict)
                injected.append(layer_name)
            except (AttributeError, IndexError) as e:
                failed.append((layer_name, str(e)))
        
        if injected:
            print(f"Successfully injected weights into layers: {injected}")
        else:
            print("No layers were injected.")
        if failed:
            for name, err in failed:
                print(f"Failed to inject layer '{name}': {err}")
        else:
            print("All specified layers were injected successfully.")
        
    def forward(self, X, lengths=None):
        """
        Forward pass through the architecture.
        :param X: Input tensor.
        :param lengths: Sequence lengths for variable-length input support.
        :return: Model output.
        """
        return self.model(X, lengths)

    def train_tl(self, *args, **kwargs):
        """
        Train the model using injected pre-trained weights.
        This method uses 'train_from_scratch' for TLModel.
        """
        print("Training Transfer Learning Model with pre-trained weights...")
        super().train_from_scratch(*args, **kwargs)

    # Delegate training_step to self.model
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    # Delegate validation_step to self.model
    def validation_step(self, batch, batch_idx):
        # Pass `self.log` to the model so it logs in the Trainer's context
        self.model.log = self.log
        return self.model.validation_step(batch, batch_idx)

    # Delegate configure_optimizers to self.model
    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def freeze_layers(self, layers_to_freeze):
        """
        Freeze specific layers.
        :param layers_to_freeze: List of layer names to freeze.
        """
        for layer_name in layers_to_freeze:
            if hasattr(self.model, layer_name):
                for param in getattr(self.model, layer_name).parameters():
                    param.requires_grad = False
                print(f"Layer '{layer_name}' has been frozen.")
            else:
                raise AttributeError(f"Layer '{layer_name}' not found in the model.")

    def unfreeze_layers(self, layers_to_unfreeze):
        """
        Unfreeze specific layers.
        :param layers_to_unfreeze: List of layer names to unfreeze.
        """
        for layer_name in layers_to_unfreeze:
            if hasattr(self.model, layer_name):
                for param in getattr(self.model, layer_name).parameters():
                    param.requires_grad = True
                print(f"Layer '{layer_name}' has been unfrozen.")
            else:
                raise AttributeError(f"Layer '{layer_name}' not found in the model.")


class TLFactory:
    """
    Factory for configuring TransferLearningModel with weights from ProductionModels.
    """

    def __init__(self):
        self.models = {}

    def load_production_model(self, model_name):
        """
        Load a production model (from the hugging neurallib collection) and store it in the factory.
        """
        model = ProductionModel(model_name=model_name)
        self.models[model_name] = model
        print(f"Loaded ProductionModel: {model_name}")

    def configure_tl_model(self, tl_model, layer_mapping, freeze_layers=None, unfreeze_layers=None):
        """
        Configure a TransferLearningModel by injecting weights and managing layer freezing/unfreezing.

        :param tl_model: Instance of TLModel to configure.
        :param layer_mapping: Dict mapping target layers in TLModel to source state_dicts.
                              Example: {'model.gru_layers.0': source_GRU_layer.state_dict()}
        :param freeze_layers: List of layer names to freeze.
        :param unfreeze_layers: List of layer names to unfreeze.
        """
        # Step 1: Inject Weights
        tl_model.inject_weights(layer_mapping)

        # Step 2: Freeze Layers
        if freeze_layers:
            print("-"*40)
            print("Freezing specified layers...")
            for layer_name in freeze_layers:
                try:
                    parts = layer_name.split('.')
                    current_attr = tl_model.model
                    for part in parts:
                        if part.isdigit():
                            current_attr = current_attr[int(part)]
                        else:
                            current_attr = getattr(current_attr, part)
                    for param in current_attr.parameters():
                        param.requires_grad = False
                    print(f"Layer '{layer_name}' frozen.")
                except (AttributeError, IndexError) as e:
                    raise ValueError(f"Failed to freeze layer '{layer_name}': {e}")

        # Step 3: Unfreeze Layers
        if unfreeze_layers:
            print("-"*40)
            print("Unfreezing specified layers...")
            for layer_name in unfreeze_layers:
                try:
                    parts = layer_name.split('.')
                    current_attr = tl_model.model
                    for part in parts:
                        if part.isdigit():
                            current_attr = current_attr[int(part)]
                        else:
                            current_attr = getattr(current_attr, part)
                    for param in current_attr.parameters():
                        param.requires_grad = True
                    print(f"Layer '{layer_name}' unfrozen.")
                except (AttributeError, IndexError) as e:
                    raise ValueError(f"Failed to unfreeze layer '{layer_name}': {e}")

        print("TLModel successfully configured!")

    def configure_tl_model_old(self, tl_model, layer_mapping, freeze_layers=None, unfreeze_layers=None):
        """
        Configure a TransferLearningModel with layer mappings and freezing strategies.
        :param tl_model: An instance of TransferLearningModel.
        :param layer_mapping: Dict mapping TLModel layers to ProductionModel layers.
        :param freeze_layers: List of layers to freeze.
        :param unfreeze_layers: List of layers to unfreeze.
        """
        # Map weights
        mapped_layers = {}
        for tl_layer, mapping_info in layer_mapping.items():
            source_model = self.models.get(mapping_info['source_model'])
            if not source_model:
                raise ValueError(f"Production model '{mapping_info['source_model']}' not loaded.")
            source_layer = getattr(source_model.model, mapping_info['source_layer'])
            mapped_layers[tl_layer] = source_layer

        tl_model.inject_weights(mapped_layers)

        # Freeze layers
        if freeze_layers:
            tl_model.freeze_layers(freeze_layers)

        # Unfreeze layers
        if unfreeze_layers:
            tl_model.unfreeze_layers(unfreeze_layers)

        print("TransferLearningModel successfully configured.")