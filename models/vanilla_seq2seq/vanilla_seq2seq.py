import torch
from torch import nn
import random
from utils.load_vocabs import UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX, remove_special_tokens
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
model_title = "vanilla_seq2seq"
emb_dim = 256
hid_dim = 512
n_layers = 2
dropout = 0.5
teacher_forcing_ratio = 0.8
# ---------------

class Encoder(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.n_tokens = n_tokens

        # Define embedding, dropout and LSTM layers.
        self.embedding = nn.Embedding(n_tokens, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src):
        # src has a shape of [seq_len, batch_size]

        # Compute an embedding from src data and apply dropout.
        # embedded should have a shape of [seq_len, batch_size, emb_dim]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # Compute the RNN output values.
        # When using LSTM, hidden should be a tuple of two tensors:
        # 1) hidden state
        # 2) cell state
        # both of shape [n_layers * n_directions, batch_size, hid_dim]
        _, hidden = self.rnn(embedded)

        return hidden

class Decoder(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.n_tokens = n_tokens

        # Define embedding, dropout and LSTM layers.
        # Additionaly, Decoder will need a linear layer to predict next token.
        self.text2emb = nn.Embedding(n_tokens, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.rnn2logits = nn.Linear(hid_dim, n_tokens)

    def forward(self, input, hidden):
        # input: [batch_size]
        # hidden is a tuple of two tensors:
        # 1) hidden state: [n_layers, batch_size, hid_dim]
        # 2) cell state:   [n_layers, batch_size, hid_dim]

        # Compute an embedding from input data and apply dropout.
        # Remember, that LSTM layer expects input to have a shape of
        # [seq_len, batch_size, emb_dim], which means that we need
        input = input.unsqueeze(dim=0)     # [1, B]
        embedded = self.text2emb(input)    # [1, B, emb_dim]
        embedded = self.dropout(embedded)

        # Compute the RNN output values.
        output, hidden = self.rnn(embedded, hidden)
        # output: [1, batch_size, hid dim]
        output = output.squeeze(dim=0) # [B, hid_dim]

        # Compute logits for the next token probabilities from RNN output.
        pred = self.rnn2logits(output) # [B, n_tokens]

        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_sz, trg_vocab_sz):
        super().__init__()

        self.encoder = Encoder(src_vocab_sz)
        self.decoder = Decoder(trg_vocab_sz)

    def forward(self, src, trg):
        # src has a shape of [src_seq_len, batch_size]
        # trg has a shape of [trg_seq_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.n_tokens

        # tensor to store decoder predictions
        preds = torch.zeros(trg_len-1, batch_size, trg_vocab_size).to(device)

        # Last hidden state of the encoder is used as
        # the initial hidden state of the decoder.
        hidden = self.encoder(src)

        # First input to the decoder is the <sos> token.
        input = trg[0]

        for t in range(trg_len-1):
            pred, hidden = self.decoder(input, hidden)
            preds[t] = pred
            teacher_force = (random.random() < teacher_forcing_ratio)
            top_pred = pred.argmax(dim=1)
            input = trg[t+1] if teacher_force else top_pred

        return preds

    @torch.no_grad()
    def translate(self, src_seq, max_len=100):
        self.eval()
        src_seq = src_seq.view(-1, 1).to(device)
        hidden = self.encoder(src_seq)
        pred_tokens = [SOS_IDX]
        for _ in range(max_len):
            decoder_input = torch.tensor([pred_tokens[-1]]).to(device)
            pred, hidden = self.decoder(decoder_input, hidden)
            _, pred_token = pred.max(dim=1)
            pred_tokens.append(pred_token.item())
            if pred_token == EOS_IDX:
                break

        return pred_tokens


"""
        @torch.no_grad()
        def translate(self, src_str, src_text2idx, trg_idx2text, max_len=100):
            self.eval()
            src_idx = torch.tensor(src_text2idx(src_str))
            src_idx = src_idx.view(-1, 1).to(device)
            hidden = self.encoder(src_idx)
            pred_tokens = [SOS_IDX]
            for _ in range(max_len):
                decoder_input = torch.tensor([pred_tokens[-1]]).to(device)
                pred, hidden = self.decoder(decoder_input, hidden)
                _, pred_token = pred.max(dim=1)
                pred_tokens.append(pred_token.item())
                if pred_token == EOS_IDX:
                    break

            return trg_idx2text(pred_tokens)
"""

