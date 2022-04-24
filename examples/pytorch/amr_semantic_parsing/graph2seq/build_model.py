from graph4nlp.pytorch.models.graph2seq import Graph2Seq


def get_model(opt, vocab_model):
    print(opt)
    model = Graph2Seq.from_args(opt, vocab_model)
    return model
