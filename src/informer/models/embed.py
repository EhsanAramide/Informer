import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding stored as a non-trainable buffer.

    Args:
        d_model: Embedding width. The reference initializer requires an even
            width because of its paired sine/cosine assignment.
        max_len: Maximum supported sequence length.
    """
    def __init__(self, d_model, max_len=5000):
        """Precompute the reference sinusoidal table."""
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Return positions ``[:x.size(1)]`` with shape ``(1, L, d_model)``."""
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    """Project values with a width-three circular temporal convolution.

    Args:
        c_in: Number of value channels.
        d_model: Embedding width.

    Circular padding means the first embedded position can depend on the last
    supplied position; caller-owned decoder construction must avoid leakage.
    """
    def __init__(self, c_in, d_model):
        """Initialize the circular convolution and Kaiming-normal weights."""
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        """Embed values from ``(B, L, c_in)`` to ``(B, L, d_model)``."""
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    """Non-trainable sinusoidal lookup table for categorical values.

    Args:
        c_in: Number of categories.
        d_model: Embedding width; the reference paired assignment expects even.
    """
    def __init__(self, c_in, d_model):
        """Create and freeze the sinusoidal category table."""
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """Look up detached embeddings for integer category indices."""
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    """Sum categorical calendar embeddings in reference column order.

    Args:
        d_model: Output width.
        embed_type: ``"fixed"`` selects sinusoidal lookups; every other value
            selects trainable :class:`torch.nn.Embedding` tables.
        freq: ``"t"`` additionally enables the quarter-hour minute bin.

    Input columns are month, day, weekday, hour, and optionally minute-bin.
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """Initialize categorical tables for the requested frequency."""
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        """Cast markers to integers and return their summed embeddings."""
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    """Linearly project continuous, frequency-dependent timestamp features.

    Args:
        d_model: Output embedding width.
        embed_type: Retained for reference signature compatibility and unused.
        freq: Key selecting the expected continuous marker width.
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """Initialize the frequency-dependent linear projection."""
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        """Project continuous marker channels to ``d_model``."""
        return self.embed(x)

class DataEmbedding(nn.Module):
    """Combine value, position, and timestamp embeddings followed by dropout.

    Args:
        c_in: Number of input value channels.
        d_model: Shared embedding width.
        embed_type: ``"timeF"`` for continuous features, ``"fixed"`` for fixed
            categorical tables, or another value for learned categorical tables.
        freq: Timestamp frequency selector used to determine marker channels.
        dropout: Dropout applied after the three components are added.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """Initialize value, position, timestamp, and dropout components."""
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """Embed aligned value and timestamp tensors of equal sequence length."""
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
