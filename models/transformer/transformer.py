import torch
import math
from torch import nn
from torch.nn import functional as F
# from utils.load_vocabs import UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX
device = "cuda" if torch.cuda.is_available() else "cpu"

SOS_IDX, EOS_IDX, PAD_IDX = 1, 2, 3

# Hyperparameters
model_title = "transformer"

emb_dim = 512
hid_dim = 512

n_heads = 8 # head_dim = 512/8 = 64
n_layers = 6

block_size = 60 # maximum context length for predictions
dropout = 0.1
# ---------------


class Head(nn.Module):
    """One head of attention
    There are three possible modes:
        1. Self-Attention
        2. Masked Self-Attention
        3. Cross-Attention
    The type of attention head is regulated by `masked`
    parameter and input to forward.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        # 1. Compute query, key and value
        # For cross attention `key` and `value` come from y

        q = self.query(x) # (B, T, hs)
        k = self.key(y)   # (B, T, hs)
        v = self.value(y) # (B, T, hs)

        # 2. Compute attention scores ("affinities")
        # wei: (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # upper triangle is set to -inf to zero-out by softmax
        wei = q @ k.transpose(-2,-1) / k.shape[-1]**0.5
        if mask is not None:
            wei = wei.masked_fill(mask==0, -float("inf"))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        # out: (B, Ts, Tt) @ (B, Tt, hs) -> (B, Ts, hs)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        out = torch.cat([h(x, y, mask=mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """Linear layer followed by a non-linearity """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    """Encoder Transformer block"""
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        head_size = emb_dim // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x, mask):
        # I will implement NLDA (pre-norm) architecture
        # dropouts are placed in `sa` and `ffwd`
        x_ln = self.ln1(x)
        x = x + self.sa(x_ln, x_ln, mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_heads, n_layers):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers

        # each token directly reads off the logits for the next token from a lookup table
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        # self.position_emb = PositionalEncoding(emb_dim, block_size)
        self.position_emb = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.ModuleList([EncoderBlock(emb_dim, n_heads=n_heads)
                                      for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(emb_dim) # final layer norm
        # self.lm_head = nn.Linear(emb_dim, vocab_size)

        # self.apply(self._init_weights)

    def forward(self, input, mask):
        B, T = input.shape

        # input and targets are both (B,T,voc_sz) tensor of integers
        # x = self.token_emb(input) # (B,T,C)
        # x = self.position_emb(x) # (B,T,C)
        tok_emb = self.token_emb(input) # (B,T,C)
        pos_emb = self.position_emb(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        for b in self.blocks:
            x = b(x, mask) # (B,T,C)
        x = self.ln_final(x) # (B,T,C)
        # state = self.lm_head(x) # (B,T,vocab_size)
        return x

    def get_mask(self, tokens):
        # padding mask
        B, T = tokens.shape
        padding_mask = (tokens != PAD_IDX).float()
        padding_mask = padding_mask.unsqueeze(1) * torch.ones((B, T, T)).to(device)
        return padding_mask

class DecoderBlock(nn.Module):
    """Decoder Transformer block"""
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        head_size = emb_dim // n_heads
        self.msa = MultiHeadAttention(n_heads, head_size) # Masked Self-Attention
        self.ca = MultiHeadAttention(n_heads, head_size) # Cross-attention
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln3 = nn.LayerNorm(emb_dim)

    def forward(self, x, y, msa_mask, csa_mask):
        # x - input, y - encoder states
        # I will implement NLDA (pre-norm) architecture
        # dropouts are placed in `sa` and `ffwd`
        x_ln = self.ln1(x)
        x = x + self.msa(x_ln, x_ln, mask=msa_mask)
        x = x + self.ca(self.ln2(x), y, mask=csa_mask)
        x = x + self.ffwd(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_heads, n_layers):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        # self.position_emb = PositionalEncoding(emb_dim, block_size)
        self.position_emb = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.ModuleList([DecoderBlock(emb_dim, n_heads=n_heads)
                                      for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(emb_dim) # final layer norm
        self.lm_head = nn.Linear(emb_dim, vocab_size)

        # self.apply(self._init_weights)

    def forward(self, input, encoder_states, msa_mask, csa_mask):
        B, T = input.shape
        # input and targets are both (B,T) tensor of integers
        # x = self.token_emb(input) # (B,T,C)
        # x = self.position_emb(x) # (B,T,C)

        tok_emb = self.token_emb(input) # (B,T,C)
        pos_emb = self.position_emb(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for b in self.blocks:
            x = b(x, encoder_states, msa_mask, csa_mask) # (B,T,C)
        x = self.ln_final(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_sz, trg_vocab_sz):
        super().__init__()
        self.encoder = Encoder(src_vocab_sz, n_heads, n_layers)
        self.decoder = Decoder(trg_vocab_sz, n_heads, n_layers)

    def forward(self, src, trg):
        # src, trg: B, T
        src, trg = pad_to_equal_len(src, trg)
        src_pad_mask = self.get_padding_mask(src)
        trg_pad_mask = self.get_padding_mask(trg)

        trg_msa_mask = trg_pad_mask * self.get_msa_mask(trg)
        csa_mask = src_pad_mask * trg_pad_mask.transpose(-2, -1)

        encoder_states = self.encoder(src, src_pad_mask)
        logits = self.decoder(trg, encoder_states, trg_msa_mask, csa_mask)
        return logits
        # B, T, C = logits.shape
        # logits = logits.view(B*T, C)
        # trg = trg.view(B*T)
        # loss = F.cross_entropy(logits, trg)
        # return logits, loss

    def get_msa_mask(self, tokens):
        B, T = tokens.shape
        return torch.triu(torch.ones((B, T, T)), diagonal=1).to(device)

    def get_padding_mask(self, tokens):
        B, T = tokens.shape
        return (tokens != PAD_IDX).float().unsqueeze(1).expand((B, T, T))

    def greedy_decode(self, src, max_len=block_size):
        """
        for inference
        Args:
            src: (1,T) - input to encoder
            trg: (1,T) - input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        enc_state = self.encoder(src)
        B, T = src.shape
        assert B == 1, f"B = {B}"
        out = torch.full((B, 1), SOS_IDX).to(device)
        for i in range(max_len-1):
            logits = self.decoder(out, enc_state)
            next_idx = torch.argmax(logits[:, -1], dim=1).view(-1, 1)
            out = torch.cat([out, next_idx], dim=1)
        return out.squeeze(0)

    def translate(self, src_text: str, src_text2idx, trg_idx2text, max_len=None):
        src = torch.tensor(src_text2idx(src_text))
        src = src.unsqueeze(0)
        num_tokens = src.shape[0]
        if max_len == None:
            max_len = num_tokens + 50
        trg_tokens = self.greedy_decode(src, max_len).flatten()
        return trg_idx2text(trg_tokens.tolist()[0])

    # def translate(self, src, max_new_tokens=block_size):
    #     # src: (T)
    #     self.eval()
    #     with torch.no_grad():
    #         src = src.unsqueeze(0).to(device)
    #         out = self.apply_padding([SOS_IDX])
    #         for i in range(max_new_tokens-1):
    #             out_tensor = torch.tensor(out).to(device).unsqueeze(0)
    #             # get the predictions
    #             logits = self(src, out_tensor) # (1, T, voc_sz)
    #             logits = logits[:, i+1, :] # (1, 1, voc_sz)
    #             probs = F.softmax(logits, dim=1)
    #             idx_next = torch.argmax(probs, dim=-1)[0].item()
    #             out[i+1] = idx_next
    #             if idx_next == EOS_IDX:
    #                 break
    #     return out

    # def translate(self, src, max_new_tokens):
    #     # src: (T)
    #     out = []
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = src[:, -block_size:]
    #         # get the predictions
    #         logits = self(idx_cond)

    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)

    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)

    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         out.append(idx_next)

    #         # append sampled index to the running sequence
    #         # idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return torch.stack(out)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_model_dim, max_seq_len):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super().__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x


def pad_to_equal_len(x1, x2, pad_value=PAD_IDX):
    return (torch.nn.utils.rnn.pad_sequence([x1.t(), x2.t()],
                                             padding_value=pad_value,
                                             batch_first=True)
            .transpose(-2, -1)
            .unbind(0))