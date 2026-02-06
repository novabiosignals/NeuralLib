import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from NeuralLib.architectures import Architecture
import inspect
import math


def list_architectures():
    """
    Lists all defined architecture classes in this module.
    """
    architectures = []

    # Iterate over all attributes in the module
    for name, obj in inspect.getmembers(__import__(__name__)):
        # Check if the object is a class and if it is defined in this module
        if inspect.isclass(obj) and obj.__module__ == __name__:
            architectures.append(name)

    return architectures


# TODO: ADD INPUT VERIFICATION TO EACH ARCHITECTURE (TASK MUST BE CLASSIFICATION OR REGRESSION, ETC.)

# Sequence-to-sequence (direct correspondence between input and output) module

class GRUseq2seq(Architecture):
    def __init__(
            self, model_name, n_features, hid_dim, n_layers, dropout, learning_rate, 
            bidirectional=False, bidir_per_layer=None, task='classification',
            num_classes=1, multi_label=False, fc_out_bool=True
    ):
        """
        :param n_features: Number of input channels/features per time step.
        :param hid_dim: Hidden dimension(s) of the GRU layers (int or vector of int - in the last case, the length
        should match the number of hidden layers (n_layers)).
        :param n_layers: Number of GRU layers.
        :param dropout: Dropout rate(s).
        :param learning_rate: Learning rate.
        :param bidirectional: Whether the GRU layers are bidirectional.
        :param bidir_per_layer: List indicating bidirectionality per layer (overrides bidirectional if provided).
        :param task: 'classification' or 'regression'.
        :param num_classes: Number of classes (for classification tasks). if binary, num_classes=1
        :param multi_label: Whether the classification task is multilabel.
        :param fc_out_bool: Whether to include a fully connected output layer, for classification it's required;
        for regression it's optional.
        """
        super(GRUseq2seq, self).__init__(architecture_name="GRUseq2seq")
        self.model_name = model_name
        self.n_features = n_features
        self.hid_dim = hid_dim if isinstance(hid_dim, list) else [hid_dim] * n_layers
        self.n_layers = n_layers
        self.dropout = dropout if isinstance(dropout, list) else [dropout] * n_layers
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        
        self.task = task  # classification or regression
        self.num_classes = num_classes if task == 'classification' else None  # only used if the task is classification
        self.multi_label = True if self.num_classes == 1 else multi_label
        self.fc_out_bool = True if task == 'classification' else fc_out_bool  # fc_out is mandatory if task is classification

        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Invalid task '{self.task}'. Task must be either 'classification' or 'regression'.")
        if self.task == "regression" and self.multi_label:
            raise ValueError("Multi-label classification cannot be set to True when task is 'regression'.")
        
        # Set bidirectionality per layer
        if bidir_per_layer is None:
            bidir_per_layer = [bidirectional] * n_layers
        #TODO: Make this available for users set less or more layers than n_layers
        if len(bidir_per_layer) != n_layers:
            raise ValueError(f"The length of bidir_per_layer ({len(bidir_per_layer)}) must match n_layers ({n_layers}).")
        
        self.bidir_per_layer = bidir_per_layer

        # Ensure hid_dim matches n_layers
        if len(self.hid_dim) != n_layers:
            raise ValueError(f"The length of hid_dim ({len(self.hid_dim)}) must match n_layers ({n_layers}).")
        if len(self.dropout) != n_layers:
            raise ValueError(f"The length of dropout ({len(self.dropout)}) must match n_layers ({n_layers}).")

        # Dynamically create GRU layers
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  # Separate dropout for intermediate layers
        input_dim = n_features
        for i in range(n_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=input_dim,
                    hidden_size=self.hid_dim[i],
                    bidirectional=self.bidir_per_layer[i],
                    batch_first=True
                )
            )
            self.dropout_layers.append(nn.Dropout(p=self.dropout[i]))
            input_dim = self.hid_dim[i] * (2 if self.bidir_per_layer[i] else 1)

        # Fully connected output layer
        if self.fc_out_bool:
            self.fc_out = nn.Linear(input_dim, self.num_classes if self.task == 'classification' else self.n_features)

        # Set loss function based on task_type
        if self.task == 'classification':
            if self.multi_label or self.num_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss() # For regression tasks

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, x, lengths):
        # make sure lengths is CPU int64
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.as_tensor(lengths, dtype=torch.long)
        else:
            lengths = lengths.to(dtype=torch.long)
        lengths = lengths.cpu()
        
        # Pack the padded sequence (expects inputs in shape [batch_size, seq_len, input_size])
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through GRU layers with dropout applied conditionally
        for i, gru in enumerate(self.gru_layers):
            packed_x, _ = gru(packed_x)  # Pass through the GRU layer
            output, _ = pad_packed_sequence(packed_x, batch_first=True)  # Unpack the output

            # Apply dropout only if defined for this layer
            if self.dropout[i] > 0:
                output = self.dropout_layers[i](output)

            # Repack the sequence if it's not the last layer
            if i < self.n_layers - 1:
                packed_x = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)

        # Unpack to apply the fully connected
        output, _ = pad_packed_sequence(packed_x, batch_first=True)
        if self.fc_out_bool:
            logits = self.fc_out(output)
        else:
            logits = output

        return logits

    def training_step(self, batch, batch_idx):
        X, Y, lengths = batch
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, lengths = batch
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler] if scheduler else [optimizer]


class GRUseq2one(Architecture):
    def __init__(
            self, model_name, n_features, hid_dim, n_layers, dropout, learning_rate, 
            bidirectional=False, bidir_per_layer=None,
            task='classification', num_classes=1, multi_label=False, fc_out_bool=True
    ):
        """
        :param n_features: Number of input channels/features per time step.
        :param hid_dim: Hidden dimension(s) of the GRU layers (int or vector of int - in the latter case, the length
        should match the number of hidden layers (n_layers)).
        :param n_layers: Number of GRU layers.
        :param dropout: Dropout rate(s).
        :param learning_rate: Learning rate.
        :param bidirectional: Whether the GRU layers are bidirectional.
        :param task: 'classification' or 'regression'.
        :param num_classes: Number of classes (for classification tasks). If binary, num_classes=1.
        :param multi_label: Whether the classification task is multilabel.
        """
        super(GRUseq2one, self).__init__(architecture_name="GRUseq2one")
        self.model_name = model_name
        self.n_features = n_features
        self.n_layers = n_layers
        self.hid_dim = hid_dim if isinstance(hid_dim, list) else [hid_dim] * self.n_layers
        self.dropout = dropout if isinstance(dropout, list) else [dropout] * self.n_layers
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        
        self.task = task  # classification or regression
        self.num_classes = num_classes if task == 'classification' else None  # only used if the task is classification
        self.multi_label = True if self.num_classes == 1 else multi_label
        self.fc_out_bool = True if task == 'classification' else fc_out_bool  # fc_out is mandatory if task is classification

        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Invalid task '{self.task}'. Task must be either 'classification' or 'regression'.")
        if self.task == "regression" and self.multi_label:
            raise ValueError("Multi-label classification cannot be set to True when task is 'regression'.")

        # Ensure hid_dim and dropout match n_layers
        if len(self.hid_dim) != n_layers:
            raise ValueError(f"The length of hid_dim ({len(self.hid_dim)}) must match n_layers ({n_layers}).")
        if len(self.dropout) != n_layers:
            raise ValueError(f"The length of dropout ({len(self.dropout)}) must match n_layers ({n_layers}).")

        # Set per-layer bidirectionality
        if bidir_per_layer is None:
            bidir_per_layer = [bidirectional] * n_layers
        if len(bidir_per_layer) != n_layers:
            raise ValueError(f"len(bidir_per_layer) must match n_layers.")
        self.bidir_per_layer = list(map(bool, bidir_per_layer))

        # Dynamically create GRU layers
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        input_dim = n_features
        
        for i in range(n_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=input_dim,
                    hidden_size=self.hid_dim[i],
                    bidirectional=self.bidir_per_layer[i],
                    batch_first=True,
                )
            )
            self.dropout_layers.append(nn.Dropout(p=self.dropout[i]))
            input_dim = self.hid_dim[i] * (2 if self.bidir_per_layer[i] else 1)

        # Fully connected output layer - only applied to the last timestep of the sequence
        if fc_out_bool:
            self.fc_out = nn.Linear(input_dim, self.num_classes if self.task == 'classification' else self.n_features)

        # Set loss function based on task_type
        if task == 'classification':
            if self.multi_label or self.num_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, x, lengths):
        # Pack the padded sequence (expects inputs in shape [batch_size, seq_len, input_size])
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through GRU layers with dropout applied conditionally
        for i, gru in enumerate(self.gru_layers):
            packed_x, _ = gru(packed_x)
            output, _ = pad_packed_sequence(packed_x, batch_first=True)

            if self.dropout[i] > 0:
                output = self.dropout_layers[i](output)

            # Repack the sequence if it's not the last layer
            if i < self.n_layers - 1:
                packed_x = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)

        # Take the output of the last timestep for each sequence
        last_outputs = output[torch.arange(output.size(0)), lengths - 1]

        # Pass through the fully connected layer
        if self.fc_out_bool:
            logits = self.fc_out(last_outputs)
        else:
            logits = last_outputs

        # Only apply softmax for multi-class classification during inference
        # if self.task == 'classification' and not self.multi_label and self.num_classes > 1:
        #     return torch.softmax(logits, dim=-1)

        # Else (num_classes==1 or multi-label is True), return raw logits
        return logits

    def training_step(self, batch, batch_idx):
        X, Y, lengths = batch
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, lengths = batch
        output = self(X, lengths)
        loss = self.criterion(output, Y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler] if scheduler else [optimizer]


class GRUED(Architecture):
    def __init__(self, model_name, n_features, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout, learning_rate,
                 bidirectional=False, fc_out_bool=True):

        super(GRUED, self).__init__(architecture_name="GRUED")
        self.model_name = model_name
        self.n_features = n_features
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.d = 2 if bidirectional else 1  # Double the hidden size if the encoder is bidirectional
        self.task = 'regression'
        self.num_classes = None
        self.multi_label = False
        self.fc_out_bool = fc_out_bool  # fc_out is mandatory if task is classification

        # Encoder GRU
        self.encoder = nn.GRU(input_size=n_features, hidden_size=enc_hid_dim, num_layers=enc_layers,
                              bidirectional=bidirectional, batch_first=True, dropout=dropout)

        # Decoder GRU
        self.decoder = nn.GRU(input_size=n_features, hidden_size=dec_hid_dim, num_layers=dec_layers,
                              batch_first=True, dropout=dropout)

        # Fully connected output layer to map hidden states to output features
        if self.fc_out_bool:
            self.fc_out = nn.Linear(dec_hid_dim, n_features)

        self.criterion = nn.MSELoss()  # For regression tasks

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src, tgt, src_lengths, tgt_lengths):
        """
        Forward pass through the GRU Encoder-Decoder model.
        :param src: Input sequence of shape [batch_size, src_seq_len, n_features].
        :param tgt: Target sequence of shape [batch_size, tgt_seq_len, n_features].
        :param src_lengths: Lengths of each sequence in the batch (for packing).
        :param tgt_lengths: Lengths of each target sequence (optional, could be ignored).
        :return: Reconstructed sequence of shape [batch_size, tgt_seq_len, n_features].
        """

        # Pack the source sequence for the encoder
        packed_src = pack_padded_sequence(src, src_lengths, batch_first=True, enforce_sorted=False)

        # Pass through the encoder
        packed_output, hidden = self.encoder(packed_src)

        # For the decoder, we need the last hidden state from the encoder
        # If encoder is bidirectional, we need to concatenate the forward and backward hidden states
        if self.bidirectional:
            hidden = self._concat_bidirectional_hidden(hidden)

        # Unpack the packed sequence (to apply dropout)
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # We can now run the decoder, starting with the hidden state from the encoder
        # Assuming `tgt` has been provided for teacher forcing
        packed_tgt = pack_padded_sequence(tgt, tgt_lengths, batch_first=True, enforce_sorted=False)

        # Pass through the decoder using the encoder's last hidden state
        packed_dec_output, _ = self.decoder(packed_tgt, hidden)

        # Unpack the decoder outputs
        dec_output, _ = pad_packed_sequence(packed_dec_output, batch_first=True)

        # Final fully connected layer to map decoder hidden states to predicted output
        if self.fc_out_bool:
            output = self.fc_out(dec_output)
        else:
            output = dec_output

        return output

    # @staticmethod
    # def _concat_bidirectional_hidden(hidden):
    #     """Concatenate forward and backward hidden states for bidirectional GRU."""
    #     # The hidden state has shape [num_layers * num_directions, batch_size, hidden_dim]
    #     forward_hidden = hidden[0:hidden.size(0):2]  # Extract forward hidden states
    #     backward_hidden = hidden[1:hidden.size(0):2]  # Extract backward hidden states
    #     return torch.cat((forward_hidden, backward_hidden), dim=2)  # Concatenate along hidden_dim

    @staticmethod
    def _concat_bidirectional_hidden(hidden):
        """Concatenate forward and backward hidden states for bidirectional GRU."""
        num_layers = hidden.shape[0] // 2  # Half the number of layers due to bidirectionality
        forward_hidden = hidden[0:num_layers]  # Extract forward hidden states
        backward_hidden = hidden[num_layers:2 * num_layers]  # Extract backward hidden states

        # Concatenate along hidden_dim to match the decoder's expectation
        return torch.cat((forward_hidden, backward_hidden), dim=2)  # Shape: (num_layers, batch, hidden_dim * 2)

    def training_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        output = self(src, tgt, src_lengths, tgt_lengths)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        output = self(src, tgt, src_lengths, tgt_lengths)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0)]


class Transformerseq2seq(Architecture):
    def __init__(self, model_name, n_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout, learning_rate, task='regression', num_classes=1, multi_label=False):
        """
        Sequence-to-sequence model using Transformer encoders and decoders.
        :param model_name: Name of the model instance, used for logging and checkpointing.
        :param n_features: Number of input channels/features per time step (i.e., signal channels).
        :param d_model: Dimension of the transformer's embedding space. Must match the expected input feature size for the transformer layers.
        :param nhead: Number of attention heads in the multi-head attention layers.
        :param num_encoder_layers: Number of stacked Transformer encoder layers.
        :param num_decoder_layers: Number of stacked Transformer decoder layers.
        :param dim_feedforward: Size of the hidden layer in the feedforward network within each transformer block.
        :param dropout: Dropout probability applied to the transformer layers.
        :param learning_rate: Learning rate for the optimizer during training.
        :param task: regression or classification
        """
        super(Transformerseq2seq, self).__init__(architecture_name="Transformerseq2seq")
        self.model_name = model_name
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.task = task  # regression or classification
        self.num_classes = num_classes if task == 'classification' else None  # only used if the task is classification
        self.multi_label = True if self.num_classes == 1 else multi_label

        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Invalid task '{self.task}'. Task must be either 'classification' or 'regression'.")
        if self.task == "regression" and self.multi_label:
            raise ValueError("Multi-label classification cannot be set to True when task is 'regression'.")

        # Linear layer to project input features to d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = FixedPositionalEncoding(d_model=d_model)
        # self.pos_encoder = LearnedPositionalEncoding(seq_len=some_max_seq_len, d_model=d_model)

        # Transformer encoder and decoder
        # Encoder Layer (self.encoder_layer): Defines a single Transformer encoder block.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        # Encoder: Stacks multiple encoder layers to process the input sequence.
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Decoder Layer: Defines a single Transformer decoder block.
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        # Decoder: Stacks multiple decoder layers to transform the encoded sequence into an output sequence.
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layer: Projects the output of the decoder to feature space (n_features).
        self.fc_out = nn.Linear(d_model, n_features)

        # Set loss function based on task_type
        if self.task == 'classification':
            if self.multi_label or self.num_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src, tgt):
        # src and tgt original shapes: [batch_size, seq_len, n_features]

        # Project input to match d_model size
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        # [batch_size, seq_len, n_features] -> [batch_size, seq_len, d_model]
        # maps your feature at each time step (e.g., ECG amplitude) to a d_model-dimensional vector (ex, 64 dimensions)

        # Transform input and target shapes to [seq_len, batch_size, d_model]
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Add positional encoding (shape remains the same: [seq_len, batch_size, d_model])
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        # The transformer is permutation-invariant (it doesn’t know the order of time steps unless explicited)
        # The positional encoder adds a position-dependent vector to each time step embedding

        # Encode the source sequence
        memory = self.encoder(src)

        # Decode the target sequence
        output = self.decoder(tgt, memory)

        # Project to the output feature space
        output = self.fc_out(output)

        # Revert back to [batch_size, seq_len, n_features]
        return output.permute(1, 0, 2)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Transformerseq2one(Architecture):  # Encoder-only Transformer
    def __init__(self, model_name, n_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, learning_rate,
                 num_classes=1, task='classification', multi_label=False):
        super(Transformerseq2one, self).__init__(architecture_name="Transformerseq2one")
        self.model_name = model_name
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.task = task
        self.num_classes = num_classes if task == 'classification' else None  # only used if the task is classification
        self.multi_label = True if self.num_classes == 1 else multi_label

        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Invalid task '{self.task}'. Task must be either 'classification' or 'regression'.")
        if self.task == "regression" and self.multi_label:
            raise ValueError("Multi-label classification cannot be set to True when task is 'regression'.")

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Fully connected layer for classification or regression
        self.fc_out = nn.Linear(d_model, num_classes)

        # Set loss function
        if num_classes == 1:
            self.criterion = nn.MSELoss()  # For regression
        else:
            self.criterion = nn.CrossEntropyLoss()  # For classification
            # TODO: FALTA A MULTILABEL

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src):
        # src shape: [batch_size, seq_len, n_features]

        # Transform input to [seq_len, batch_size, d_model]
        src = src.permute(1, 0, 2)

        # Encode the source sequence
        memory = self.encoder(src)

        # Take the last hidden state for sequence-to-one
        output = self.fc_out(memory[-1])  # Last time step

        return output

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TransformerED(Architecture):
    def __init__(self, model_name, n_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
                 learning_rate):
        super(TransformerED, self).__init__(architecture_name="TransformerED")
        self.model_name = model_name
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.task = 'regression'
        self.num_classes = None
        self.multi_label = False

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layer to map output of the transformer to feature space
        self.fc_out = nn.Linear(d_model, n_features)

        # Loss function for encoder-decoder
        self.criterion = nn.MSELoss()  # Assuming regression; change for classification tasks

        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, src, tgt):
        # src and tgt shapes: [batch_size, seq_len, n_features]

        # Transform input and target to [seq_len, batch_size, d_model]
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Encode the source sequence
        memory = self.encoder(src)

        # Decode the target sequence
        output = self.decoder(tgt, memory)

        # Project to the output feature space
        output = self.fc_out(output)

        # Revert back to [batch_size, seq_len, n_features]
        return output.permute(1, 0, 2)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt)
        loss = self.criterion(output, tgt)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
