
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import os
import sys
from datetime import date, timedelta
import gc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import random
import requests
from math import sqrt
import math
import yaml
import traceback
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 1) A small CNN to extract local features in (C, H, W)
# -------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    A small CNN block to process input weather data (B, 16, H, W) into (B, embed_dim, H, W).
    You can expand with more layers, skip connections, etc. for better performance.
    """
    def __init__(self, in_channels=16, out_channels=32):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: (B, in_channels, H, W)
        return self.conv_block(x)  # => (B, out_channels, H, W)

# -------------------------------------------------------------------------
# 1) DateTime Embedding
# -------------------------------------------------------------------------
class DateTimeEmbedding(nn.Module):
    """
    Embeds (month, day, hour) each of shape (batch_size, 3)
    into a single vector of shape (batch_size, embed_dim).
    """
    def __init__(self, embed_dim=16):
        super().__init__()
        self.month_embed = nn.Embedding(12, embed_dim)
        self.day_embed   = nn.Embedding(31, embed_dim)
        self.hour_embed  = nn.Embedding(24, embed_dim)
        
    def forward(self, month, day, hour):
        month = month - 1
        day = day - 1
        hour = hour - 1
        # Each is (batch_size, 3). We embed them individually:
        # month_embed(month) => (batch_size, 3, embed_dim)
        
        month_emb = self.month_embed(month).mean(dim=1)  # => (batch_size, embed_dim)
        day_emb   = self.day_embed(day).mean(dim=1)      # => (batch_size, embed_dim)
        hour_emb  = self.hour_embed(hour).mean(dim=1)    # => (batch_size, embed_dim)
        
        # Combine them by summation (or you can concat and do a small MLP)
        
        combined = month_emb + day_emb + hour_emb
        return combined  # => (batch_size, embed_dim)


# -------------------------------------------------------------------------
# 2) Spatial Positional Encoding in 2D form
# -------------------------------------------------------------------------
class SpatialPositionalEncoding2D(nn.Module):
    """
    A 2D positional encoding for an 8x9 grid.
    - row_embed has shape (8, embed_dim)
    - col_embed has shape (9, embed_dim)
    We sum row_embed[row] + col_embed[col] to get a unique embedding
    for each (row, col).
    """
    def __init__(self, grid_size=(8, 9), embed_dim=16):
        super().__init__()
        self.H, self.W = grid_size
        self.row_embed = nn.Embedding(self.H, embed_dim)
        self.col_embed = nn.Embedding(self.W, embed_dim)

    def forward(self):
        """
        Returns a 2D positional encoding of shape (embed_dim, H, W).
        Typically you'd add this to your feature map of shape (B, embed_dim, H, W).
        """
        device = self.row_embed.weight.device
        
        # row indices: 0..(H-1), col indices: 0..(W-1)
        rows = torch.arange(self.H, device=device)  # shape (H,)
        cols = torch.arange(self.W, device=device)  # shape (W,)
        
        # row_embed(rows) => (H, embed_dim)
        # col_embed(cols) => (W, embed_dim)
        row_emb = self.row_embed(rows)  # (H, embed_dim)
        col_emb = self.col_embed(cols)  # (W, embed_dim)
        
        # Sum: broadcast to (H, W, embed_dim), then transpose to (embed_dim, H, W)
        # row_emb => (H, 1, embed_dim)
        # col_emb => (1, W, embed_dim)
        # out => (H, W, embed_dim) => permute => (embed_dim, H, W)
        pos_2d = row_emb.unsqueeze(1) + col_emb.unsqueeze(0)  # (H, W, embed_dim)
        pos_2d = pos_2d.permute(2, 0, 1)  # => (embed_dim, H, W)
        return pos_2d


# -------------------------------------------------------------------------
# 3) Attention-based Pooling over the flattened tokens
# -------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """
    Learns a linear 'query' that produces a scalar attention score for each token.
    Then uses softmax to get attention weights, and outputs a weighted sum.
    """
    def __init__(self, embed_dim=16):
        super().__init__()
        self.attn_query = nn.Linear(embed_dim, 1)  # (embed_dim) -> scalar

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        returns (batch_size, embed_dim)
        """
        # attn_logits => (batch_size, seq_len, 1)
        attn_logits = self.attn_query(x)
        attn_logits = attn_logits.squeeze(-1)  # => (batch_size, seq_len)
        
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # => (batch_size, seq_len, 1)
        pooled = (x * attn_weights).sum(dim=1)  # => (batch_size, embed_dim)
        return pooled


# -------------------------------------------------------------------------
# 4) Transformer Model with "Apply 2D Pos Encoding, then Flatten"
# -------------------------------------------------------------------------
class TransformerModel2D(nn.Module):
    """
    Demonstrates how to:
      - Keep the data in (B, C, H, W) shape initially.
      - Apply a 1x1 conv (or linear) to project from 16 -> embed_dim in 2D form.
      - Add the 2D positional embeddings (embed_dim, H, W).
      - Flatten to (B, H*W, embed_dim).
      - Add the temporal embedding (month/day/hour).
      - Pass through Transformer.
      - Use attention-based pooling for a final single embedding.
    """
    def __init__(self, 
                 input_dim=16,   # raw weather feature channels
                 embed_dim=16, 
                 num_heads=4, 
                 num_layers=3, 
                 grid_size=(8,9)):
        super().__init__()
        
        self.grid_size = grid_size  # (H=8, W=9)
        self.embed_dim = embed_dim
        
        # 2D positional encoder (returns (embed_dim, H, W))
        self.spatial_pe_2d = SpatialPositionalEncoding2D(grid_size, embed_dim)
        
        # 1x1 conv to project from input_dim -> embed_dim while preserving H,W
        # If input_dim == embed_dim, you could skip this layer or make it an identity.
        # self.feature_projection = nn.Conv2d(
        #     in_channels=input_dim, 
        #     out_channels=embed_dim,
        #     kernel_size=1, 
        #     bias=False
        # )

        self.feature_projection = SimpleCNN(in_channels=input_dim, out_channels=embed_dim)

        # Embedding for (month, day, hour) => (batch_size, embed_dim)
        # self.datetime_embed = DateTimeEmbedding(embed_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention-based pooling
        self.attn_pool = AttentionPool(embed_dim)

    def forward(self, x, time_emb):
        """
        x: shape (batch_size, 16, 8, 9)
        month/day/hour: (batch_size, 3)
        """
        #minus 1 to make the month and day and hour start from 0
        # month = month - 1
        # day = day - 1
        # hour = hour - 1
        
        B, C, H, W = x.shape
        assert (H, W) == self.grid_size, "Input grid size must match the set grid_size."

        # 1) Project input features to embed_dim in 2D form:
        #    (B, 16, 8, 9) -> (B, embed_dim, 8, 9)
        x_proj = self.feature_projection(x)  # => (B, embed_dim, 8, 9)

        # 2) Get 2D positional embedding => shape (embed_dim, 8, 9)
        pos_2d = self.spatial_pe_2d()  # => (embed_dim, 8, 9)

        # 3) Broadcast and add => (B, embed_dim, 8, 9)
        #   pos_2d is (embed_dim, H, W). We add it to each batch example.
        x_proj = x_proj + pos_2d.unsqueeze(0)  # broadcast over batch

        # 4) Flatten spatial dimension => (B, embed_dim, 8*9) => (B, 72, embed_dim)
        x_proj = x_proj.view(B, self.embed_dim, -1).permute(0, 2, 1)
        # shape now: (batch_size, 72, embed_dim)

        # 5) Embed the date/time => (B, embed_dim)
        #directly use the time embedding
        # time_emb = self.datetime_embed(month, day, hour)  # => (B, embed_dim)


        # Expand to (B, 1, embed_dim) and broadcast to the 72 tokens
        time_emb_2d = time_emb.unsqueeze(1).expand(-1, x_proj.size(1), -1)  # => (B, 72, embed_dim)

        # 6) Combine the time embedding with the spatially encoded tokens
        x_proj = x_proj + time_emb_2d  # add or could use another projection

        # 7) Pass through Transformer
        out = self.transformer_encoder(x_proj)  # => (B, 72, embed_dim)

        # 8) Apply attention-based pooling => (B, embed_dim)
        pooled = self.attn_pool(out)

        return pooled

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)
    

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    
class IT(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, 
                 seq_len,
                 d_model,
                 n_heads,
                 dropout=0.1,
                 d_ff=2048,
                 activation = 'gelu',
                 e_layers = 2):
        super(IT, self).__init__()
        self.seq_len = seq_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed_type='fixed', freq='h',
                                                    dropout=dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        # print(means.shape)
        x_enc =  x_enc - means

        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        # print(enc_out[0,:,:])
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        enc_out = enc_out.mean(1)
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out # [B, D]
    
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, query, key, value):
        # query: (B*pred_hours, 1, embed_dim)
        # key, value: (B*pred_hours, 2, embed_dim)
        attn_output, _ = self.cross_attn(query, key, value)
        query = self.norm1(query + attn_output)
        ffn_output = self.ffn(query)
        out = self.norm2(query + ffn_output)
        return out


# ============================================================================
# 5) Combined Predictor Model
#    Uses cross-attention fusion (with residual connections) to fuse the weather
#    and load latent representations based on target forecast time.
# ============================================================================
class CombinedGridLoadPredictor(nn.Module):
    def __init__(self, config):
        """
        weather_encoder: instance of TransformerModel2D (weather branch)
        load_encoder: instance of ITEncoder (load branch)
        latent_dim: shared embedding dimension (e.g., 32)
        grid_size: (H, W) of the grid
        pred_hours: number of forecast hours (e.g., 3)
        datetime_embed: shared DateTimeEmbedding instance
        """
        super(CombinedGridLoadPredictor, self).__init__()
        self.weather_encoder = TransformerModel2D(
                                                input_dim=config['input_dim'], 
                                                embed_dim=config['embed_dim'],  # let’s go bigger internally
                                                num_heads=config['n_heads'], 
                                                num_layers=config['tf_num_layers'], 
                                                grid_size=config['grid_size']
                                            )
        self.load_encoder = IT(
                                seq_len=config['seq_len'],
                                d_model=config['embed_dim'],
                                n_heads=config['n_heads'],
                                dropout=config['dropout'],
                                d_ff=int(config['embed_dim'] * 2),
                            )
        
        self.datetime_embed = DateTimeEmbedding(embed_dim=config['embed_dim'])
        
        # Cross-attention fusion block.
        # For each forecast hour, we form key/value from stacked weather and load latents.
        self.cross_attn_fusion = CrossAttentionFusion(embed_dim=config['embed_dim'], n_heads=config['n_heads'])
        
        #output layer
        self.out_layer = nn.Linear(config['embed_dim'], config['pred_hours'])
    
    def forward(self, batch):
        """
        weather_data: (B, 16, 8, 9)
        weather_time_info: (B, pred_hours, 3) for the weather forecast times
        load_x_enc: (B, 168, 20)
        load_x_mark_enc: (B, 168, 3)
        target_time_info: (B, pred_hours, 3) for the downstream target times
        """
        weather_data = batch['raw_data_weather']
        pred_month = batch['label_month']
        pred_day = batch['label_day']
        pred_hour = batch['label_hour']

        ts_data = batch['raw_data_ts']
        ts_month = batch['train_month']
        ts_day = batch['train_day']
        ts_hour = batch['train_hour']

        B = weather_data.size(0)
        # External embeddings using the shared DateTimeEmbedding
        target_time_emb = self.datetime_embed(pred_month, pred_day, pred_hour) # (B, latent_dim)
        
        # Weather branch: produce weather latent per forecast hour → (B, latent_dim)
        weather_latent = self.weather_encoder(weather_data, target_time_emb)

        #create x_mark for the load encoder
        load_x_mark_enc = torch.cat([ts_month.unsqueeze(-1), ts_day.unsqueeze(-1), ts_hour.unsqueeze(-1)], dim=-1) # (B, T, 3)
        
        # Load branch: produce a single latent → (B, latent_dim); replicate for forecast hours.
        load_latent = self.load_encoder(ts_data, load_x_mark_enc, None, None)  # (B, latent_dim)
        
        # For each forecast hour, form a key-value pair by stacking the two latents.
        # keys: (B, 2, latent_dim)
        keys = torch.stack([weather_latent, load_latent], dim=1)

        # Query: use target time embedding for each forecast hour → (B, 1, latent_dim)
        query = target_time_emb.unsqueeze(1)
        
        # Apply cross-attention fusion
        fused = self.cross_attn_fusion(query, keys, keys)  # (B*pred_hours, 1, latent_dim)
        
        #get prediction
        pred = self.out_layer(fused.squeeze(1))
        
        return {'pred':pred}
