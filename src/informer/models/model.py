import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.masking import TriangularCausalMask, ProbMask
from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .embed import DataEmbedding

class Informer(nn.Module):
    """Informer encoder-decoder forecasting model.

    Args:
        enc_in: Encoder value-channel count.
        dec_in: Decoder value-channel count.
        c_out: Forecast output-channel count.
        seq_len: Retained reference argument; forward uses ``x_enc`` length.
        label_len: Retained reference argument; the caller assembles ``x_dec``.
        out_len: Number of trailing decoder positions returned.
        factor: ProbSparse sampling factor.
        d_model: Internal embedding width.
        n_heads: Attention head count.
        e_layers: Encoder block count.
        d_layers: Decoder block count.
        d_ff: Feed-forward hidden width.
        dropout: Embedding, attention, and residual dropout probability.
        attn: ``"prob"`` selects ProbSparse; any other value selects full.
        embed: ``"timeF"``, ``"fixed"``, or learned categorical embedding.
        freq: Timestamp-frequency selector for the embedding.
        activation: ``"relu"`` or, for every other value, GELU.
        output_attention: Return encoder attention diagnostics with predictions.
        distil: Insert convolutional distillation between encoder blocks.
        mix: Apply the reference decoder head/time mixing operation.
        device: Retained reference argument; it does not move parameters.

    Encoder, decoder, and forecast widths may differ. The caller owns decoder
    context and placeholder construction and must prevent future-target leakage.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        """Initialize the reference Informer architecture."""
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """Forecast the final ``out_len`` decoder positions.

        Args:
            x_enc: Historical encoder values ``(B, L_enc, enc_in)``.
            x_mark_enc: Encoder timestamp markers ``(B, L_enc, M)``.
            x_dec: Decoder context and placeholders ``(B, L_dec, dec_in)``.
            x_mark_dec: Decoder timestamp markers ``(B, L_dec, M)``.
            enc_self_mask: Optional encoder self-attention mask.
            dec_self_mask: Optional decoder self-attention mask.
            dec_enc_mask: Optional decoder-to-encoder cross-attention mask.

        Returns:
            Prediction ``(B, out_len, c_out)`` when enough decoder positions
            exist, or ``(prediction, encoder_attentions)`` when requested.
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    """Informer with independent encoders over progressively shorter suffixes.

    Arguments match :class:`Informer`, except ``e_layers`` is a list containing
    the depth of each encoder. Encoder ``i`` receives the final
    ``L_enc // (2 ** i)`` embedded positions. Diagnostics are nested by encoder.
    The ``seq_len``, ``label_len``, and ``device`` arguments retain the same
    compatibility-only behavior as :class:`Informer`.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        """Initialize the reference multi-encoder InformerStack architecture."""
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """Forecast from stacked suffix encoders using the Informer interface.

        Args:
            x_enc: Historical encoder values ``(B, L_enc, enc_in)``.
            x_mark_enc: Encoder timestamp markers ``(B, L_enc, M)``.
            x_dec: Decoder context and placeholders ``(B, L_dec, dec_in)``.
            x_mark_dec: Decoder timestamp markers ``(B, L_dec, M)``.
            enc_self_mask: Accepted but currently ignored inside EncoderStack.
            dec_self_mask: Optional decoder self-attention mask.
            dec_enc_mask: Optional decoder-to-encoder cross-attention mask.

        Returns:
            Prediction, optionally paired with nested encoder diagnostics.
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
