import torch

from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import EmbeddingConstruction
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from graph4nlp.pytorch.data.dataset import Text2LabelDataItem
from graph4nlp.pytorch.data.data import GraphData, to_batch
from examples.pytorch.amr_graph_construction.amr_graph_construction import AMRGraphConstruction
from graph4nlp.pytorch.data.dataset import Text2LabelDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from stanfordcorenlp import StanfordCoreNLP

if __name__ == "__main__":
    raw_text_data = [["I like nlp.", "Same here!"], ["I like graph.", "Same here!"]]

    # src_text_seq = list(zip(*raw_text_data))[0]
    # src_idx_seq = [vocab_model.word_vocab.to_index_sequence(each) for each in src_text_seq]
    # src_len = torch.LongTensor([len(each) for each in src_idx_seq])
    # num_seq = torch.LongTensor([len(src_len)])
    # input_tensor = torch.LongTensor(pad_2d_vals_no_size(src_idx_seq))
    # print("input_tensor: {}".format(input_tensor.shape))
    raw_data = (
        "We need to borrow 55% of the hammer price until we can get planning permission for restoration which will allow us to get a mortgage . I saw a nice dog and noticed he was eating a bone ."
    )

    graph = AMRGraphConstruction.static_topology(raw_data)
    data_set = Text2LabelDataItem('I like nlp.')
    data_set.graph = graph
    vocab_model = VocabModel(
        [data_set], max_word_vocab_size=None, min_word_vocab_freq=1, word_emb_size=300
    )
    emb_constructor = EmbeddingConstruction(vocab_model.in_word_vocab, False, emb_strategy="bert_bilstm_amr",hidden_size=300)
    g = Text2LabelDataset._vectorize_one_dataitem(data_set, vocab_model)
    emb = emb_constructor(to_batch([g.graph, g.graph]))
    print("emb: {}".format(emb.node_features))