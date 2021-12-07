from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import (
    DependencyBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import (
    EmbeddingConstruction,
)
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values

if __name__ == "__main__":
    raw_text_data = ["i like nlp", "same here", "i like graph", "same here"]

    # data_set = [Text2LabelDataItem(input_text=x, output_label=[0, 1],
    #  tokenizer=word_tokenize) for x in raw_text_data]

    # vocab_model = VocabModel(
    #     data_set=raw_text_data, max_word_vocab_size=None, min_word_vocab_freq=1, word_emb_size=300
    # )

    # src_text_seq = list(zip(*raw_text_data))[0]
    # src_idx_seq = [[0, 1, 2], [3, 4], [0, 1, 5], [3, 4]]
    # src_idx_seq = [vocab_model.in_word_vocab.to_index_sequence(each) for each in src_text_seq]
    # src_len = torch.LongTensor([len(each) for each in src_idx_seq])
    # num_seq = torch.LongTensor([len(src_len)])
    # input_tensor = torch.LongTensor(pad_2d_vals_no_size(src_idx_seq))
    # print("input_tensor: {}".format(input_tensor.shape))

    # word_vocab = Vocab()
    # word_vocab.build_vocab({'i': 2, 'like': 2, 'nlp': 1, 'same': 2, 'here':2, 'graph': 1})
    # print(word_vocab.get_vocab_size())
    # word_vocab.load_embeddings()
    # emb_constructor = EmbeddingConstruction(word_vocab,
    # single_token_item=False, emb_strategy='w2v_transformer', rnn_dropout=0.1,
    # transformer_dropout=0.1, hidden_size=300)

    # processed_data_items = []
    # dataset = Text2LabelDataset(
    #         port=9000,
    #         graph_name='constituency',
    #         dynamic_init_graph_name=None,
    #         topology_builder=None,
    #         dynamic_init_topology_builder=None,
    #         lower_case=True,
    #         tokenizer=word_tokenize,
    #         merge_strategy='tailhead',
    #         edge_strategy='homogeneous',
    #     )
    # dataset.build_vocab()
    # print("Start preparing test data")
    # for raw_sentence in raw_text_data:
    #     data_item = Text2LabelDataItem(input_text=raw_sentence, tokenizer=word_tokenize)

    #     data_item = dataset.process_data_items(data_items=[data_item])
    #     data_item = dataset._vectorize_one_dataitem(
    #         data_item[0], dataset.vocab_model, use_ie=False
    #     )
    #     processed_data_items.append(data_item)

    # collate_data = dataset.collate_fn(processed_data_items)

    dataset = JobsDataset(
        root_dir="graph4nlp/pytorch/test/dataset/jobs",
        #    port=9000,
        topology_builder=DependencyBasedGraphConstruction,
        graph_name="dependency",
        topology_subdir="DependencyGraph",
    )

    emb_constructor_transformer = EmbeddingConstruction(
        dataset.vocab_model.in_word_vocab,
        single_token_item=False,
        emb_strategy="w2v_transformer",
        rnn_dropout=0.1,
        transformer_dropout=0.1,
        hidden_size=300,
    )

    emb_constructor_bilstm = EmbeddingConstruction(
        dataset.vocab_model.in_word_vocab,
        single_token_item=False,
        emb_strategy="w2v_bilstm",
        rnn_dropout=0.1,
        transformer_dropout=0.1,
        hidden_size=300,
    )

    user_args = get_yaml_config(
        "examples/pytorch/semantic_parsing/graph2seq/config/dependency_gcn_bi_sep_demo.yaml"
    )
    args = get_basic_args(
        graph_construction_name="node_emb", graph_embedding_name="gat", decoder_name="stdrnn"
    )
    update_values(to_args=args, from_args_list=[user_args])
    graph2seq = Graph2Seq.from_args(args, dataset.vocab_model)

    batch_data = JobsDataset.collate_fn(dataset.train[0:12])

    print(batch_data["graph_data"])
    emb_bilstm = emb_constructor_bilstm(batch_data["graph_data"])
    emb_transformer = emb_constructor_transformer(batch_data["graph_data"])
    print("emb bilstm: {}".format(emb_bilstm.batch_node_features["node_feat"].shape))
    print("emb transformer: {}".format(emb_transformer.batch_node_features["node_feat"].shape))
