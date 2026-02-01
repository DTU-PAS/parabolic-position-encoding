import math

import torch
import torch.nn as nn

from pape.configs import Config
from pape.data_types import EventsClassificationBatch
from pape.data_types import EventTokens
from pape.nn.encoder import TransformerEncoder


class EventClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        cls = torch.empty(1, 1, config.model.hidden_size)
        nn.init.normal_(cls, mean=0.0, std=0.02)
        self.cls = nn.Parameter(cls)
        self.embedding = Embedding(config)
        self.encoder = TransformerEncoder(config)

        self.classifier = nn.Linear(config.model.hidden_size, config.num_classes)
        self.classifier.weight.data.zero_()
        self.classifier.bias.data.fill_(-math.log(config.num_classes))

        self.normalize = config.normalize
        self.time_scale = config.events.time_scale
        self.num_rows = math.ceil(config.height / config.patch_size)
        self.num_cols = math.ceil(config.width / config.patch_size)

        self.num_layers = config.model.num_layers
        self.weight_decay = config.optimizer.weight_decay
        self.layer_decay = config.optimizer.layer_decay

    def forward(self, batch: EventsClassificationBatch) -> torch.Tensor:
        batch_size = batch.tokens.x.size(0)
        embeddings = self.embedding(batch.tokens)
        cls_token = self.cls.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_token, embeddings), dim=1)

        x = batch.tokens.x.to(embeddings.dtype)
        y = batch.tokens.y.to(embeddings.dtype)
        t = batch.tokens.t.to(embeddings.dtype) / self.time_scale

        positions = torch.stack((x, y, t), dim=-1)
        positions = torch.cat(
            [
                torch.zeros(batch_size, 1, 3, dtype=positions.dtype, device=positions.device),
                positions,
            ],
            dim=1,
        )

        if self.normalize:
            min_x = torch.zeros(batch_size, dtype=positions.dtype, device=positions.device)
            max_x = torch.full((batch_size,), self.num_cols - 1, dtype=positions.dtype, device=positions.device)
            min_y = torch.zeros(batch_size, dtype=positions.dtype, device=positions.device)
            max_y = torch.full((batch_size,), self.num_rows - 1, dtype=positions.dtype, device=positions.device)
            min_t = t.min(dim=1).values
            max_t = t.max(dim=1).values
            min_pos = torch.stack((min_x, min_y, min_t), dim=-1).unsqueeze(1)
            max_pos = torch.stack((max_x, max_y, max_t), dim=-1).unsqueeze(1)
            positions = (positions - min_pos) / (max_pos - min_pos)

        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=batch.padding_mask.device),
                batch.padding_mask,
            ],
            dim=1,
        )

        encoded = self.encoder(embeddings, positions, padding_mask=padding_mask)

        encoded_cls = encoded[:, 0, :]
        logits = self.classifier(encoded_cls)

        return logits


class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.patch_size = config.patch_size
        self.channels = 2 * config.events.buckets  # multiply by 2 for positive and negative events
        self.hidden_size = config.model.hidden_size

        self.projection = nn.Conv2d(
            self.channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, tokens: EventTokens):
        patches = tokens.patches.to(torch.float32)

        # Join batch/sequence dimensions and polarity/bucket dimensions
        batch_size = patches.size(0)
        sequence_length = patches.size(1)
        shape = (-1, self.channels, self.patch_size, self.patch_size)
        patches = patches.view(*shape)

        # Logarithmically scale the input to account for poorly distributed values
        patches = torch.log(patches + 1)

        embeddings = self.projection(patches)

        embeddings = embeddings.view(batch_size, sequence_length, self.hidden_size)

        return embeddings
