import math

import torch
import torch.nn as nn

from pape.configs import Config
from pape.data_types import EventDetectionBatch
from pape.data_types import EventTokens
from pape.nn.encoder import TransformerEncoder
from pape.nn.yolov10.head import YOLOv10Head


class EventDetector(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.hidden_size = config.model.hidden_size
        self.num_rows = math.ceil(config.height / config.patch_size)
        self.num_cols = math.ceil(config.width / config.patch_size)

        self.embedding = Embedding(config)
        self.encoder = TransformerEncoder(config)

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, config.detection.fpn_size, kernel_size=2, stride=2),
            LayerNorm2d(config.detection.fpn_size),
        )

        self.fpn2 = nn.Sequential(
            nn.Conv2d(self.hidden_size, config.detection.fpn_size, kernel_size=1),
            LayerNorm2d(config.detection.fpn_size),
        )

        self.fpn3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.hidden_size, config.detection.fpn_size, kernel_size=1),
            LayerNorm2d(config.detection.fpn_size),
        )

        self.head = YOLOv10Head(config)

        self.normalize = config.normalize
        self.time_scale = config.events.time_scale
        self.num_rows = math.ceil(config.height / config.patch_size)
        self.num_cols = math.ceil(config.width / config.patch_size)

    def forward(self, batch: EventDetectionBatch, return_attention_maps: bool = False):
        batch_size = batch.tokens.x.size(0)
        embeddings = self.embedding(batch.tokens)

        x = batch.tokens.x.to(embeddings.dtype)
        y = batch.tokens.y.to(embeddings.dtype)
        t = (batch.tokens.t / self.time_scale).to(embeddings.dtype)
        positions = torch.stack((x, y, t), dim=-1)

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

        encoder_outputs = self.encoder(
            embeddings,
            positions,
            padding_mask=batch.padding_mask,
            return_attention_maps=return_attention_maps,
        )

        if return_attention_maps:
            encoded, attention_maps = encoder_outputs
        else:
            encoded = encoder_outputs

        # Gather tokens into a 2D grid
        mask = batch.padding_mask.logical_not()
        flattened_mask = mask.view(-1)
        flattened_tokens = encoded.reshape(-1, self.hidden_size)[flattened_mask]
        pos_x = batch.tokens.x.view(-1).to(torch.long)[flattened_mask]
        pos_y = batch.tokens.y.view(-1).to(torch.long)[flattened_mask]
        lengths = mask.sum(dim=1)
        batch_ids = torch.cat(
            [torch.full((length,), i, dtype=torch.long, device=pos_x.device) for i, length in enumerate(lengths)],
            dim=0,
        )
        encoded_image = torch.zeros(
            batch_size,
            self.num_rows,
            self.num_cols,
            self.hidden_size,
            device=flattened_tokens.device,
            dtype=flattened_tokens.dtype,
        )
        encoded_image[batch_ids, pos_y, pos_x] = flattened_tokens

        encoded_image = encoded_image.permute(0, 3, 1, 2)  # (batch_size, hidden_size, num_rows, num_cols)

        # Build feature pyramid as in ViTDet
        fpn1 = self.fpn1(encoded_image)
        fpn2 = self.fpn2(encoded_image)
        fpn3 = self.fpn3(encoded_image)

        # YOLOv10 head
        predictions = self.head([fpn1, fpn2, fpn3])

        if return_attention_maps:
            return predictions, attention_maps

        return predictions


class LayerNorm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


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
