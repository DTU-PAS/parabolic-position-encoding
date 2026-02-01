import math

import torch
import torch.nn as nn

from pape.configs import Config
from pape.data_types import ImageDetectionBatch
from pape.nn.encoder import TransformerEncoder
from pape.nn.yolov10.head import YOLOv10Head


class ImageDetector(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.hidden_size = config.model.hidden_size
        self.num_rows = math.ceil(config.height / config.patch_size)
        self.num_cols = math.ceil(config.width / config.patch_size)

        self.embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.model.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
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

        rows = torch.arange(start=1, end=self.num_rows + 1)
        cols = torch.arange(start=1, end=self.num_cols + 1)
        row_positions = rows.unsqueeze(1).repeat(1, self.num_cols).view(-1)
        col_positions = cols.unsqueeze(0).repeat(self.num_rows, 1).view(-1)
        positions = torch.stack((col_positions, row_positions), dim=-1).unsqueeze(0)  # (1, seq_length, 2)
        if config.normalize:
            max_pos = torch.tensor([[[self.num_rows, self.num_cols]]], dtype=positions.dtype)
            positions = positions / max_pos
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, batch: ImageDetectionBatch, return_attention_maps: bool = False):
        embeddings = self.embedding(batch.images)  # (batch_size, hidden_size, num_rows, num_cols)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (batch_size, seq_length, hidden_size)

        encoder_output = self.encoder(
            embeddings,
            self.positions,
            return_attention_maps=return_attention_maps,
        )

        if return_attention_maps:
            encoded, attention_maps = encoder_output
        else:
            encoded = encoder_output

        encoded_image = encoded.view(-1, self.num_rows, self.num_cols, self.hidden_size)
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
