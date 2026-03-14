import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DiscriminatorV1(nn.Module):
    """
    First-version discriminator.

    Uses:
    Embedding + LSTM + Linear
    to perform binary classification of a trajectory.
    """

    def __init__(self, config, data_feature):
        """
        Initialize the discriminator.

        Args:
            config (dict): Model configuration.
            data_feature (dict): Dataset-related feature description.
        """
        super(DiscriminatorV1, self).__init__()

        # =========================================================
        # MODEL PARAMETERS
        # =========================================================
        self.road_emb_size = config["road_emb_size"]
        self.hidden_size = config["hidden_size"]
        self.lstm_layer_num = config["lstm_layer_num"]
        self.dropout_p = config["dropout_p"]

        # =========================================================
        # MODEL STRUCTURE
        # =========================================================
        self.input_size = self.road_emb_size

        # Road embedding layer
        self.road_emb = nn.Embedding(
            num_embeddings=data_feature["road_num"],
            embedding_dim=self.road_emb_size,
            padding_idx=data_feature["road_pad"]
        )

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layer_num,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_p)

        # Output layer for binary classification
        self.out_linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=2
        )

        # Loss function
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, trace_loc, trace_time, trace_mask=None):
        """
        Forward pass.

        Args:
            trace_loc (tensor): Trajectory road-id sequence, shape (batch_size, seq_len)
            trace_time (tensor): Trajectory time sequence, shape (batch_size, seq_len)
            trace_mask (tensor, optional):
                Padding mask where 1 indicates valid values and 0 indicates padding.
                Shape: (batch_size, seq_len)

        Returns:
            tensor: Trajectory classification scores, shape (batch_size, 2)
                    Index 0 = fake trajectory score
                    Index 1 = real trajectory score
        """
        trace_loc_emb = self.road_emb(trace_loc)

        input_emb = trace_loc_emb

        if trace_mask is not None:
            trace_lengths = torch.sum(trace_mask, dim=1).tolist()

            packed_input = pack_padded_sequence(
                input_emb,
                lengths=trace_lengths,
                batch_first=True,
                enforce_sorted=False
            )

            packed_lstm_hidden, _ = self.lstm(packed_input)
            lstm_hidden, _ = pad_packed_sequence(
                packed_lstm_hidden,
                batch_first=True
            )
        else:
            lstm_hidden, _ = self.lstm(input_emb)

        if trace_mask is not None:
            last_indices = torch.sum(trace_mask, dim=1) - 1
            last_indices = last_indices.reshape(last_indices.shape[0], 1, 1)
            last_indices = last_indices.repeat(1, 1, self.hidden_size)

            lstm_last_hidden = torch.gather(
                lstm_hidden,
                dim=1,
                index=last_indices
            ).squeeze(1)
        else:
            lstm_last_hidden = lstm_hidden[:, -1]

        lstm_last_hidden = self.dropout(lstm_last_hidden)
        trace_score = self.out_linear(lstm_last_hidden)

        return trace_score

    def predict(self, trace_loc, trace_time, trace_mask=None):
        """
        Predict class probabilities.

        Args:
            trace_loc (tensor): Trajectory road-id sequence, shape (batch_size, seq_len)
            trace_time (tensor): Trajectory time sequence, shape (batch_size, seq_len)
            trace_mask (tensor, optional): Padding mask, shape (batch_size, seq_len)

        Returns:
            tensor: Softmax probabilities, shape (batch_size, 2)
        """
        score = self.forward(trace_loc, trace_time, trace_mask)
        return torch.softmax(score, dim=1)

    def calculate_loss(self, trace_loc, trace_time, target, trace_mask=None):
        """
        Compute cross-entropy loss.

        Args:
            trace_loc (tensor): Trajectory road-id sequence, shape (batch_size, seq_len)
            trace_time (tensor): Trajectory time sequence, shape (batch_size, seq_len)
            target (tensor): Ground-truth binary labels, shape (batch_size)
            trace_mask (tensor, optional): Padding mask, shape (batch_size, seq_len)

        Returns:
            tensor: Cross-entropy loss
        """
        score = self.forward(trace_loc, trace_time, trace_mask)
        loss = self.loss_func(score, target)
        return loss