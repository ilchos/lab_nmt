from utils.load_vocabs import *

def translate_encoded(model, src, trg):
    # src_tokens = src.tolist()[::-1]
    src_list = src.tolist()
    trg_list = trg.tolist()
    out_list = model.translate(src)
    return tuple(remove_special_tokens(tokens)
                 for tokens in [vocab_transform[SRC_LANGUAGE].lookup_tokens(src_list),
                                vocab_transform[TRG_LANGUAGE].lookup_tokens(trg_list),
                                vocab_transform[TRG_LANGUAGE].lookup_tokens(out_list) ])

def translate_text(model, src: str, trg: str):
    src_tokens = text_transform[SRC_LANGUAGE](src)
    trg_tokens = text_transform[TRG_LANGUAGE](trg)
    return translate_encoded(model, src_tokens, trg_tokens)

def translate_batch(model, src, trg):
    result = []
    for s in range(src.shape[1]):
        result.append(translate_encoded(model, src[:, s], trg[:, s]))
    return result

def translation_summary(tokens_tuple):
    return "\n".join([" ".join(tokens) for tokens in tokens_tuple ])

def translation2writer(src, trg, out):
    summary_list = ["\n\n".join(sent) for sent in zip(src, trg, out)]
    result_string = "\n\n-------\n\n".join(summary_list).replace("<unk>", "UNK")
    return result_string