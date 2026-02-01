from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pape.configs import Config
from pape.nn.drop_path import DropPath
from pape.nn.positions import get_position_encoder


class TransformerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.position_encoder = get_position_encoder(config)
        self.position_encoder.register_model_weights()
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.model.num_layers)])
        self.norm_pre = nn.LayerNorm(config.model.hidden_size)
        self.norm_post = nn.LayerNorm(config.model.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_attention_maps: bool = False,
    ) -> tuple[torch.Tensor]:
        if padding_mask is None:
            attn_mask = None
        else:
            attn_mask = padding_mask.logical_not().unsqueeze(1).unsqueeze(2)

        positions = self.position_encoder.prepare_positions(positions)
        x = self.position_encoder.encode_absolute(x, positions)

        x = self.norm_pre(x)

        attention_maps = []
        for layer in self.layers:
            layer_outputs = layer(x, positions, attn_mask=attn_mask, return_attention_maps=return_attention_maps)
            if return_attention_maps:
                x, layer_attention_maps = layer_outputs
                attention_maps.append(layer_attention_maps)
            else:
                x = layer_outputs

        x = self.norm_post(x)

        if return_attention_maps:
            return x, attention_maps

        return x


class Layer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.drop_path = DropPath(config.model.drop_path)
        self.norm1 = nn.LayerNorm(config.model.hidden_size)
        self.attention = Attention(config)
        self.norm2 = nn.LayerNorm(config.model.hidden_size)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: Any,
        attn_mask: torch.Tensor | None = None,
        return_attention_maps: bool = False,
    ) -> tuple[torch.Tensor]:
        attention_outputs = self.attention(
            self.norm1(x), positions, attn_mask=attn_mask, return_attention_maps=return_attention_maps
        )

        if return_attention_maps:
            x = x + self.drop_path(attention_outputs[0])
            attention_maps = attention_outputs[1]
        else:
            x = x + self.drop_path(attention_outputs)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        if return_attention_maps:
            return x, attention_maps

        return x


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.head_size = config.model.head_size
        self.num_heads = config.model.num_heads
        self.proj_size = config.model.num_heads * config.model.head_size
        self.scale = self.head_size**-0.5
        self.dropout = config.model.dropout
        self.qkv_proj = nn.Linear(config.model.hidden_size, 3 * self.proj_size)
        self.out_proj = nn.Linear(self.proj_size, config.model.hidden_size)
        self.position_encoder = get_position_encoder(config)
        self.position_encoder.register_layer_weights()
        self.has_position_bias = self.position_encoder.has_bias()

    def forward(
        self,
        x: torch.Tensor,
        positions: Any,
        attn_mask: torch.Tensor | None = None,
        return_attention_maps: bool = False,
    ) -> tuple[torch.Tensor]:
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_length, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_size)

        query = qkv[0]
        key = qkv[1]
        value = qkv[2]

        query, key = self.position_encoder.encode_query_key(x, query, key, positions)

        if return_attention_maps or self.has_position_bias:
            logits = (query @ key.transpose(-2, -1)) * self.scale
            if self.has_position_bias:
                bias = self.position_encoder.get_bias(positions)
                logits = logits + bias
            attention_maps = torch.softmax(logits, dim=-1)
            attention_maps = torch.dropout(attention_maps, self.dropout, train=self.training)
            x = attention_maps @ value

            x = x.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_size)
            x = x.reshape(batch_size, seq_length, self.proj_size)
            x = self.out_proj(x)

            if return_attention_maps:
                return x, attention_maps

            return x
        else:
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )

            x = x.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_size)
            x = x.reshape(batch_size, seq_length, self.proj_size)
            x = self.out_proj(x)

            return x


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.model.hidden_size, config.model.intermediate_size)
        self.fc2 = nn.Linear(config.model.intermediate_size, config.model.hidden_size)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
