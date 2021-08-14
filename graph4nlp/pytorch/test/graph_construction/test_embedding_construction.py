import torch

from ...modules.graph_construction.embedding_construction import EmbeddingConstruction
from ...modules.utils.padding_utils import pad_2d_vals_no_size
from ...modules.utils.vocab_utils import VocabModel

if __name__ == "__main__":
    raw_text_data = [["I like nlp.", "Same here!"], ["I like graph.", "Same here!"]]

    vocab_model = VocabModel(
        raw_text_data, max_word_vocab_size=None, min_word_vocab_freq=1, word_emb_size=300
    )

    src_text_seq = list(zip(*raw_text_data))[0]
    src_idx_seq = [vocab_model.word_vocab.to_index_sequence(each) for each in src_text_seq]
    src_len = torch.LongTensor([len(each) for each in src_idx_seq])
    num_seq = torch.LongTensor([len(src_len)])
    input_tensor = torch.LongTensor(pad_2d_vals_no_size(src_idx_seq))
    print("input_tensor: {}".format(input_tensor.shape))

    emb_constructor = EmbeddingConstruction(vocab_model.word_vocab, "w2v", "bilstm", "bilstm", 128)
    emb = emb_constructor(input_tensor, src_len, num_seq)
    print("emb: {}".format(emb.shape))
