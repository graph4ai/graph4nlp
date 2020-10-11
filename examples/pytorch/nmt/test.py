from graph4nlp.pytorch.modules.graph_construction import IEBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel

embedding_style = {'word_emb_type': 'w2v', 'node_edge_emb_strategy': "mean",
                           'seq_info_encode_strategy': "bilstm"}
vocab = VocabModel.build("/home/shiina/shiina/lib/graph4nlp/graph4nlp/pytorch/test/dataset/graph2seq/processed/DependencyGraph/vocab.pt")
graph_topology = IEBasedGraphConstruction(embedding_style=embedding_style, vocab=vocab.in_word_vocab)
dep = DependencyBasedGraphConstruction(embedding_style=embedding_style, vocab=vocab.in_word_vocab)
print(isinstance(graph_topology, IEBasedGraphConstruction))
print(type(graph_topology) == IEBasedGraphConstruction)

