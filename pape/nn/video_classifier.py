import math

import torch
import torch.nn as nn

from pape.configs import Config
from pape.data_types import VideoClassificationBatch
from pape.nn.encoder import TransformerEncoder


class VideoClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        cls = torch.empty(1, 1, config.model.hidden_size)
        nn.init.normal_(cls, mean=0.0, std=0.02)
        self.cls = nn.Parameter(cls)
        self.embedding = nn.Conv3d(
            in_channels=3,
            out_channels=config.model.hidden_size,
            kernel_size=(config.video.frame_length, config.patch_size, config.patch_size),
            stride=(config.video.frame_length, config.patch_size, config.patch_size),
        )
        self.encoder = TransformerEncoder(config)

        self.classifier = nn.Linear(config.model.hidden_size, config.num_classes)
        self.classifier.weight.data.zero_()
        self.classifier.bias.data.fill_(-math.log(config.num_classes))

        num_times = config.video.max_samples
        num_rows = math.ceil(config.height / config.patch_size)
        num_cols = math.ceil(config.width / config.patch_size)

        times = torch.arange(start=1, end=num_times + 1)
        rows = torch.arange(start=1, end=num_rows + 1)
        cols = torch.arange(start=1, end=num_cols + 1)
        time_positions = times.unsqueeze(1).unsqueeze(2).repeat(1, num_rows, num_cols).view(-1)
        row_positions = rows.unsqueeze(0).unsqueeze(2).repeat(num_times, 1, num_cols).view(-1)
        col_positions = cols.unsqueeze(0).unsqueeze(1).repeat(num_times, num_rows, 1).view(-1)
        positions = torch.stack((time_positions, row_positions, col_positions), dim=-1)
        positions = positions.unsqueeze(0)  # (1, seq_length, 3)
        cls_position = torch.zeros((1, 1, 3), dtype=positions.dtype)
        positions = torch.cat((cls_position, positions), dim=1)  # (1, seq_length + 1, 3)
        if config.normalize:
            max_pos = torch.tensor([[[num_times, num_rows, num_cols]]], dtype=positions.dtype)
            positions = positions / max_pos
        self.register_buffer("positions", positions, persistent=False)

        self.tokens_per_frame = num_rows * num_cols
        self.seq_length = 1 + config.video.max_samples * self.tokens_per_frame

        self.num_layers = config.model.num_layers
        self.weight_decay = config.optimizer.weight_decay

    def forward(self, batch: VideoClassificationBatch, return_attention_maps: bool = False) -> tuple[torch.Tensor]:
        batch_size = batch.videos.size(0)
        videos = batch.videos.to(self.cls.dtype)
        embeddings = self.embedding(videos)  # (batch_size, hidden_size, num_rows, num_cols)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (batch_size, seq_length, hidden_size)
        cls_token = self.cls.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_token, embeddings), dim=1)

        # create padding mask from lengths
        lengths = 1 + batch.lengths * self.tokens_per_frame
        max_indices = torch.arange(self.seq_length, device=lengths.device).expand(batch_size, -1)
        padding_mask = max_indices >= lengths.unsqueeze(1)

        encoder_output = self.encoder(
            embeddings,
            self.positions,
            padding_mask=padding_mask,
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
