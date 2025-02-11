from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from torchtext.data.utils import get_tokenizer
from typing import Iterable, List

import torchtext as tt
import sentencepiece as sp

from utils.utils_sp import SOS_IDX, EOS_IDX, PAD_IDX, SRC_LN, TRG_LN

device = "cuda" if torch.cuda.is_available() else "cpu"

model_title = "transformer_nn"

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2Seq(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 num_encoder_layers: int = NUM_DECODER_LAYERS,
                 num_decoder_layers: int = NUM_DECODER_LAYERS,
                 emb_size: int = EMB_SIZE,
                 nhead: int = NHEAD,
                 dim_feedforward: int = FFN_HID_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, trg_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.trg_tok_emb = TokenEmbedding(trg_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.trg_tok_emb(trg))

        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_masks(src, trg)

        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None,
                                src_padding_mask, trg_padding_mask, src_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor):
        src_mask = torch.zeros((src.shape[0], src.shape[0]), device=device)
        return self.transformer.encoder(
                    self.positional_encoding(self.src_tok_emb(src)),
                    src_mask)

    def decode(self, trg: Tensor, encoder_state: Tensor):
        trg_mask = msa_mask(trg.size(0)).to(device)
        return self.transformer.decoder(
                    self.positional_encoding(self.trg_tok_emb(trg)),
                    encoder_state,
                    trg_mask)

    def greedy_decode(self, src, max_len=80):
        self.eval()
        with torch.no_grad():
            src = src.to(device)
            encoder_state = self.encode(src).to(device)
            ys = torch.ones(1, 1).fill_(SOS_IDX).type(torch.long).to(device)
            for i in range(max_len-1):
                out = self.decode(ys, encoder_state)
                out = out.transpose(0, 1)
                prob = self.generator(out[:, -1])
                next_word = torch.argmax(prob, dim=1)
                next_word = next_word.item()

                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                if next_word == EOS_IDX:
                    break
        return ys

    # actual function to translate input sentence into target language
    def translate(self, src_text: str, src_text2idx, trg_idx2text, max_len=None):
        self.eval()
        src = torch.tensor(src_text2idx(src_text)).view(-1, 1)
        num_tokens = src.shape[0]
        if max_len == None:
            max_len = num_tokens + 50
        trg_tokens = self.greedy_decode(src, max_len).flatten()
        return trg_idx2text(trg_tokens.tolist())


######################################################################
# During training, we need a subsequent word mask that will prevent the model from looking into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#

def msa_mask(sz):
    # masked self-attention mask
    mask = torch.triu(torch.ones((sz, sz)), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def create_masks(src, trg):
    src_seq_len = src.shape[0]
    trg_seq_len = trg.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device)
    trg_mask = msa_mask(trg_seq_len).type_as(src_mask)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1).type_as(src_mask)
    trg_padding_mask = (trg == PAD_IDX).transpose(0, 1).type_as(src_mask)
    return src_mask, trg_mask, src_padding_mask, trg_padding_mask
