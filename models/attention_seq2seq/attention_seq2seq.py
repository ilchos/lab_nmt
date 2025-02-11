import torch
from torch import nn
import random
from utils.load_vocabs import UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
model_title = "attention_seq2seq" # 25M parameters
emb_dim = 256
hid_dim = 512
encoder_layers = 1 # bidirectional
decoder_layers = 1
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
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=encoder_layers,
                           bidirectional=True)

        # fc layer for hidden states:
        # so neural network will learn to take the best
        # of the forward and backward pass
        self.fc_hidden = nn.Linear(2*hid_dim, hid_dim)
        self.fc_cell = nn.Linear(2*hid_dim, hid_dim)

    def forward(self, src):
        # encoder rnn encodes whole sequence at once
        # src: [src_len, batch_size]
        # embedded: [seq_len, batch_size, emb_dim]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # state: [src_len, batch_size, 2*hid_dim]
        # hidden, cell: [2, batch_size, hid_dim]
        states, (hidden, cell) = self.rnn(embedded)

        # concat vectors from 2 directions
        # 'unsqueeze' is used to fit pytorch's lstm convension
        # hidden, cell -> [1, batch_size, 2*hid_dim]
        hidden = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[0], cell[1]), dim=1).unsqueeze(0)

        # apply fc layers to hidden and cell
        # hidden, cell -> [1, batch_size, hid_dim]
        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)

        return states, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.n_tokens = n_tokens

        self.text2emb = nn.Embedding(n_tokens, emb_dim)
        self.dropout = nn.Dropout(dropout)
        # RNN takes input embedding and context vector of size `hid_dim`
        self.rnn = nn.LSTM(emb_dim + 3*hid_dim, hid_dim, num_layers=decoder_layers)

        # Multi-layer perceptron to calculate attention scores
        # Takes state vector from forward and backward pass of encoder (2*hid_dim)
        # and decoder hidden vector (hid_dim)
        # => 3*hid_dim
        self.mlp = nn.Sequential(
            nn.Linear(3*hid_dim, 3*hid_dim),
            nn.Tanh(),
            nn.Linear(3*hid_dim, 1)
        )

        # Softmax for calculating attention weights
        self.softmax = nn.Softmax(dim=0)

        # Linear layer to predict next token.
        self.rnn2logits = nn.Linear(hid_dim, n_tokens)


    def forward(self, input, encoder_states, hidden_tuple):
        """decoder rnn decodes 1 symbol at a time
        at each step we calculate context vectors based on encoder_state

        Args:
            input: [batch_size]
            encoder_state: [src_len, batch_size, hid_dim]
            hidden, cell: [1, batch_size, hid_dim]

        Returns:
            logits
            hidden
        """
        # 1 Calculate embedding of input
        # input: [batch_size]->[1, batch_size] - sequence of length 1
        input = input.unsqueeze(dim=0)

        # embedded: [1, batch_size, emb_dim]
        embedded = self.text2emb(input)
        embedded = self.dropout(embedded)

        # 2 Calculate context vector
        # 2.1 Concatenate hidden vector to each of encoder's state vectors
        # encoder_state: [src_len, batch_size, 2*hid_dim]
        # hidden: [1, batch_size, hid_dim]
        # hidden_repeat: [src_len, batch_size, hid_dim]
        # context_in: [src_len, batch_size, 3*hid_dim]
        hidden, cell = hidden_tuple
        hidden_repeat = hidden.repeat(encoder_states.shape[0], 1, 1)
        context_in = torch.cat((encoder_states, hidden_repeat), dim=2)

        # 2.2 Calculate attention weights
        # MLP gives scores, softmax converts scores to weights
        # attention: [src_len, batch_size, 1]
        attention = self.softmax(self.mlp(context_in))

        # 2.3 Context is weighted sum of encoder states
        # context: [1, batch_size, 3*hid_dim]
        context = torch.einsum("snk,snl->knl", attention, context_in)

        # 3 Concatenate embedding with context and pass to RNN
        # Encoder's hidden and cell states are also passed to RNN
        # rnn_input: [1, batch_size, emb_dim + 3*hid_dim]
        rnn_input = torch.cat((embedded, context), dim=2)

        # states: [1, batch_size, hid_dim]
        # hidden_dec: tuple of [1, batch_size, hid_dim]
        states, hidden_dec = self.rnn(rnn_input, hidden_tuple)
        # state -> [batch_size, hid_dim]
        states = states.squeeze(dim=0)

        # 4 Compute logits for the next token probabilities from RNN output.
        # logits: [batch_size, n_tokens]
        logits = self.rnn2logits(states)
        return logits, hidden_dec


class Seq2Seq(nn.Module):
    def __init__(self, n_tokens_src, n_tokens_trg):
        super().__init__()

        self.encoder = Encoder(n_tokens_src)
        self.decoder = Decoder(n_tokens_trg)

    def forward(self, src, trg):
        # src has a shape of [src_seq_len, batch_size]
        # trg has a shape of [trg_seq_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.n_tokens

        #
        encoder_states, hidden = self.encoder(src)

        # tensor to store decoder predictions
        preds = torch.zeros(trg_len-1, batch_size, trg_vocab_size).to(device)

        # First input to the decoder is the <sos> token.
        input = trg[0]

        for t in range(trg_len-1):
            pred, hidden = self.decoder(input, encoder_states, hidden)
            preds[t] = pred
            teacher_force = (random.random() < teacher_forcing_ratio)
            top_pred = pred.argmax(dim=1)
            input = trg[t+1] if teacher_force else top_pred

        return preds

    @torch.no_grad()
    def greedy_decode(self, src_seq, max_len=100):
        self.eval()
        encoder_states, hidden = self.encoder(src_seq)
        pred_tokens = [SOS_IDX]
        for _ in range(max_len):
            decoder_input = torch.tensor([pred_tokens[-1]]).to(device)
            pred, hidden = self.decoder(decoder_input, encoder_states, hidden)
            _, pred_token = pred.max(dim=1)
            pred_tokens.append(pred_token.item())
            if pred_token == EOS_IDX:
                break

        return pred_tokens

    def translate(self, src_text: str, src_text2idx, trg_idx2text, max_len=None):
        self.eval()
        src = torch.tensor(src_text2idx(src_text)).view(-1, 1).to(device)
        num_tokens = src.shape[0]
        if max_len == None:
            max_len = num_tokens + 50
        trg_tokens = self.greedy_decode(src, max_len)
        return trg_idx2text(trg_tokens)
