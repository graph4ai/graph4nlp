from .graph_construction import get_graph_construction_args
from .graph_embedding import get_graph_embedding_args
from .prediction.generation import get_decoder_args


def get_basic_args(graph_construction_name, graph_embedding_name, decoder_name):
    graph_construction_args = get_graph_construction_args(graph_construction_name)
    graph_embedding_args = get_graph_embedding_args(graph_embedding_name)
    decoder_args = get_decoder_args(decoder_name)
    ret = {
        "graph_constrcution_args": graph_construction_args,
        "graph_embedding_args": graph_embedding_args,
        "decoder_args": decoder_args
    }
    return ret


__all__ = ["get_graph_construction_args", "get_graph_embedding_args", "get_decoder_args", "get_basic_args"]
