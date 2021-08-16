from stanfordcorenlp import StanfordCoreNLP

from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import (
    IEBasedGraphConstruction,
)


def test_ie_graph():
    # raw_data = "James went to the corner-shop. He want to buy some (eggs), " \
    #            "<milk> and bread for breakfast."
    # raw_data = ('Return to Olympus is the only album by the alternative rock band Malfunkshun . '
    #             'It was released after the band had broken up and after lead singer Andrew Wood '
    #             '( later of Mother Love Bone ) had died of a drug overdose in 1990 . ')

    raw_data = "James is on shop. He buys eggs."

    # raw_data = ('Paul Allen was born on January 21, 1953, in Seattle, Washington, '
    #             'to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside School, '
    #             'a private school in Seattle, where he befriended Bill Gates, two years younger, '
    #             'with whom he shared an enthusiasm for computers. ')
    # 'Paul and Bill used a teletype terminal at their high school, Lakeside, '
    # 'to develop their programming skills on several time-sharing computer systems.')

    # raw_data = ('Police arrested a man named Albert Boucher . As of Saturday afternoon , he '
    #             'was being held in the Warren County Regional Jail on a $5,000 bond .')

    # use '_' replace '-' ? 'fourth-degree'->'fourth_degree'
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

    # raw_data = ('Rand Paul was assaulted in his home in Bowling Green, Kentucky , on Friday , '
    #             'according to Kentucky State Police . '
    #             'State troopers responded to a call to the senator \'s '
    #             'residence at 3:21 p.m. Friday . Police arrested a man named Rene Albert '
    #             'Boucher, who they '
    #             'allege " intentionally assaulted " Paul , causing him "minor injury" . '
    #             'Boucher , 59 , of Bowling '
    #             'Green was charged with one count of fourth_degree assault . As of Saturday '
    #             'afternoon , he '
    #             'was being held in the Warren County Regional Jail on a $5,000 bond .')

    nlp_parser = StanfordCoreNLP("http://localhost", port=9000, timeout=300000)

    # vocab_model =  VocabModel([raw_data.split('.'), raw_data.split('.')],
    #                     max_word_vocab_size=None,
    #                     min_word_vocab_freq=1,
    #                     word_emb_size=300)

    output_graph0 = IEBasedGraphConstruction.topology(
        raw_data, nlp_parser, merge_strategy=None, edge_strategy=None
    )

    # embedding_styles = {'word_emb_type':'w2v', 'node_edge_emb_strategy':'mean',
    # 'seq_info_encode_strategy':'bilstm'}
    # embedding_styles = {
    #     "word_emb_type": "w2v",
    #     "node_edge_emb_strategy": "lstm",
    #     "seq_info_encode_strategy": "bilstm",
    # }
    # iegc = IEBasedGraphConstruction(vocab=vocab_model.word_vocab,
    # embedding_style=embedding_styles, use_cuda=False)
    # output_graph0 = iegc.embedding(output_graph0)

    output_graph1 = IEBasedGraphConstruction.topology(
        raw_data, nlp_parser, merge_strategy="global", edge_strategy=None
    )
    # output_graph1 = iegc.embedding(output_graph1)

    output_graph2 = IEBasedGraphConstruction.topology(
        raw_data, nlp_parser, merge_strategy=None, edge_strategy="as_node"
    )
    # output_graph2 = iegc.embedding(output_graph2)

    output_graph3 = IEBasedGraphConstruction.topology(
        raw_data, nlp_parser, merge_strategy="global", edge_strategy="as_node"
    )
    # output_graph3 = iegc.embedding(output_graph3)

    # IEBasedGraphConstruction.topology(raw_data, nlp_parser,
    #                                   merge_strategy="share_common_tokens",
    #                                   edge_strategy='as_node')

    print(output_graph0.node_attributes)
    print(output_graph0.edge_attributes)

    print(output_graph1.node_attributes)
    print(output_graph1.edge_attributes)

    print(output_graph2.node_attributes)
    print(output_graph2.edge_attributes)

    print(output_graph3.node_attributes)
    print(output_graph3.edge_attributes)
    pass


if __name__ == "__main__":
    test_ie_graph()
