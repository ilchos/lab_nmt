import pandas as pd

# poor man's data split
df = pd.read_csv("./data/data.txt", sep="\t",
                 header=None, encoding="utf-8").sample(frac=1, random_state=19)
total_len = df.shape[0]
def write_to_file(lines, fname):
    with open("./data/"+fname, "w", encoding="utf8") as f:
        f.write("\n".join(lines))

l1, l2, l3 = [ int(p*total_len) for p in [0.8, 0.15, 0.05] ]

write_to_file(df.iloc[:l1, 0].to_list(), "train.en")
write_to_file(df.iloc[:l1, 1].to_list(), "train.ru")

write_to_file(df.iloc[l1:l1+l2, 0].to_list(), "valid.en")
write_to_file(df.iloc[l1:l1+l2, 1].to_list(), "valid.ru")

write_to_file(df.iloc[-l3:, 0].to_list(), "test.en")
write_to_file(df.iloc[-l3:, 1].to_list(), "test.ru")

# %%
import sentencepiece as sp

# taken from video
def get_options(input_path, output_path):
    return dict(
        # input spec
        input=input_path,
        input_format="text",
        # output spec
        model_prefix=output_path, # output filename prefix
        # algorithm spec
        # BPE alg
        model_type="bpe",
        vocab_size=10000,
        # normalization
        normalization_rule_name="nmt_nfkc_cf", # turn off normalization
        remove_extra_whitespaces=False,
        # input_sentence_size=200000000, # max number of training sentences
        # max_sentence_length=4192, # max number of bytes per sentence
        seed_sentencepiece_size=1_000_000,
        shuffle_input_sentence=True,
        # rare word treatment
        character_coverage=0.99995,
        byte_fallback=True,
        # merge rules
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0, # the UNK token MUST exist
        bos_id=1, # the others are optional, set to -1 to turn off
        eos_id=2,
        pad_id=3,
        # systems
        num_threads=3, # use ~all system resources
    )



# %%
# train sp models
sp.SentencePieceTrainer.train(**get_options("./data/train.en", "../.stor/sp_en"))
sp.SentencePieceTrainer.train(**get_options("./data/train.ru", "../.stor/sp_ru"))

# %%
# load model
sp_proc = sp.SentencePieceProcessor()
sp_proc.load('.stor/sp_en.model')

