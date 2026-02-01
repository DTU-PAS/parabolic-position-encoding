import math

import torch
import torch.nn as nn

from pape.configs import Config
from pape.data_types import ImageClassificationBatch
from pape.nn.encoder import TransformerEncoder


class ImageClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        cls = torch.empty(1, 1, config.model.hidden_size)
        nn.init.normal_(cls, mean=0.0, std=0.02)
        self.cls = nn.Parameter(cls)
        self.embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.model.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.encoder = TransformerEncoder(config)

        self.classifier = nn.Linear(config.model.hidden_size, config.num_classes)
        self.classifier.weight.data.zero_()
        self.classifier.bias.data.fill_(-math.log(config.num_classes))

        num_rows = math.ceil(config.height / config.patch_size)
        num_cols = math.ceil(config.width / config.patch_size)

        rows = torch.arange(start=1, end=num_rows + 1)
        cols = torch.arange(start=1, end=num_cols + 1)
        row_positions = rows.unsqueeze(1).repeat(1, num_cols).view(-1)
        col_positions = cols.unsqueeze(0).repeat(num_rows, 1).view(-1)
        positions = torch.stack((col_positions, row_positions), dim=-1).unsqueeze(0)  # (1, seq_length, 2)
        cls_position = torch.zeros((1, 1, 2), dtype=positions.dtype)
        positions = torch.cat((cls_position, positions), dim=1)  # (1, seq_length + 1, 2)
        if config.normalize:
            max_pos = torch.tensor([[[num_rows, num_cols]]], dtype=positions.dtype)
            positions = positions / max_pos
        self.register_buffer("positions", positions, persistent=False)

        self.num_layers = config.model.num_layers
        self.weight_decay = config.optimizer.weight_decay
        self.layer_decay = config.optimizer.layer_decay

    def forward(self, batch: ImageClassificationBatch, return_attention_maps: bool = False) -> tuple[torch.Tensor]:
        batch_size = batch.images.size(0)
        embeddings = self.embedding(batch.images)  # (batch_size, hidden_size, num_rows, num_cols)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (batch_size, seq_length, hidden_size)
        cls_token = self.cls.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_token, embeddings), dim=1)

        encoder_output = self.encoder(
            embeddings,
            self.positions,
            return_attention_maps=return_attention_maps,
        )

        if return_attention_maps:
            encoded, attention_maps = encoder_output
        else:
            encoded = encoder_output

        encoded_cls = encoded[:, 0, :]
        logits = self.classifier(encoded_cls)

        if return_attention_maps:
            return logits, attention_maps

        return logits
