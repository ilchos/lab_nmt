import torch
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from collections import Counter

UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ('<unk>', '<sos>', '<eos>', '<pad>')

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'ru'

# token_transform = {SRC_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
#                    TRG_LANGUAGE: get_tokenizer('spacy', language='ru_core_news_sm')}
vocab_transform = dict()
# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = dict()

def load_data(path="data/data.txt"):
    with open(path, mode="r", encoding="utf8") as f:
        data = [line.lower().rstrip().split("\t") for line in f.readlines()]

    train_data, test_data, val_data = random_split(data,
                                                   [0.8, 0.15, 0.05],
                                                   generator=torch.Generator().manual_seed(19)) # same results every time
    return train_data, test_data, val_data

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add SOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# reverse transform for SRC
def reverse_transform(tokens):
    return tokens[::-1]

def yield_tokens(data_iter, language):
    language_index = {SRC_LANGUAGE: 0, TRG_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

def init_vocabs(train_data):
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_data, ln),
                                                        min_freq=2,
                                                        specials=SPECIAL_TOKENS,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)
        text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization,
                                                   vocab_transform[ln], # Numericalization
                                                   tensor_transform)    # Add SOS/EOS and create tensor

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        trg_batch.append(text_transform[TRG_LANGUAGE](trg_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
    return src_batch, trg_batch

def remove_special_tokens(tokens):
    return list(filter(lambda x: x not in SPECIAL_TOKENS[1:], tokens))
