# from train_old import SOS_IDX, EOS_IDX, SPECIAL_TOKENS
# from train_old import SRC_LN, TRG_LN
import sentencepiece as sp
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import WordPunctTokenizer
import torch
from tqdm.auto import tqdm

UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ('<unk>', '<s>', '</s>', '<pad>')

SRC_LN = "en"
TRG_LN = "ru"

def load_sp_processors():
    return {ln: sp.SentencePieceProcessor(f"./.stor/sp_{ln}.model")
            for ln in [SRC_LN, TRG_LN]}

def text2idx_sentpiece(sp_model):
    def func(text):
        return sp_model.encode(text, add_bos=True, add_eos=True)
    return func

def idx2text_sentpiece(sp_model):
    def func(idxs):
        return sp_model.decode(idxs)
    return func

def text2idx_field(field):
    def func(text):
        tokens = field.tokenize(text)
        idx = field.process([tokens]).flatten().tolist()
        return idx
    return func

def idx2text_field(field):
    vocab_sos, vocab_bos = 2, 3
    def func(idxs):
        return " ".join([field.vocab.itos[idx] for idx in idxs
                         if idx not in [vocab_sos, vocab_bos]])
    return func

# def text2idx_transformer(sp_model, block_size):
#     def func(text):
#         idxs = sp_model.encode(text)[block_size:-2]
#         idxs.insert(SOS_IDX)
#         idxs.append(EOS_IDX)
#         return idxs
#     return func

def translate(model, src_text, src_text2idx, trg_idx2text):
    src_idx = src_text2idx(src_text)
    trg_idx = model.translate(src_idx)
    trg_idx = trg_idx.tolist()
    trg_text = trg_idx2text(trg_idx)
    return trg_text

def evaluate_bleu_score(trg_corpus, translations):
    tokenizer = WordPunctTokenizer() # split into tokens
    output_tokens = [tokenizer.tokenize(text) for text in translations]
    trg_tokens = [[tokenizer.tokenize(text)] for text in trg_corpus]
    return corpus_bleu(trg_tokens, output_tokens)*100

def translate_batch(model, src_batch, src_text2idx, trg_idx2text):
    translations = [model.translate(src, src_text2idx, trg_idx2text)
                    for src, trg in tqdm(src_batch, desc="Translate", leave=False)]
    bleu_score = evaluate_bleu_score([trg for src, trg in src_batch], translations)
    translation_tuples = [(src, trg, out) for (src, trg), out in zip(src_batch,
                                                                      translations)]
    return translation_tuples, bleu_score

def translation2string(sentence_tuples):
    summary_list = [f"{i}. " + "\n".join(sent) for i, sent in enumerate(sentence_tuples)]
    result_string = "\n-------\n".join(summary_list)
    return result_string

def translation2file(fpath, sentence_tuples, epoch):
    with open(fpath, "a+", encoding="utf8") as f:
        f.write(f"EPOCH {epoch}\n" + translation2string(sentence_tuples))

def translation2writer(sentence_tuples):
    summary_list = ["\n\n".join(sent) for sent in sentence_tuples]
    result_string = "\n\n-------\n\n".join(summary_list).replace("<unk>", "UNK")
    return result_string


# def translate_batch(model, src_batch, src_text2idx, trg_idx2text):
#     translations = [translate(model, src, src_text2idx, trg_idx2text)
#                     for src, trg in tqdm(src_batch, desc="Translate", leave=False)]
#     bleu_score = evaluate_bleu_score([trg for src, trg in src_batch], translations)
#     translation_tuples = [(src, trg, out) for (src, trg), out in zip(src_batch,
#                                                                       translations)]
#     return translation_tuples, bleu_score
