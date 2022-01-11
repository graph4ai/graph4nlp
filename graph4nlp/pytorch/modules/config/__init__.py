from .graph_construction import get_graph_construction_args
from .graph_embedding import get_graph_embedding_args
from .graph_initialization import get_graph_initialization_args
from .prediction.generation import get_decoder_args


def get_basic_args(graph_construction_name, graph_embedding_name, decoder_name):
    """
        It will build the template for ``Graph2X`` model.
    Parameters
    ----------
    graph_construction_name: str
        The graph construction method name. Expected in ["dependency", "constituency", \
            "node_emb", "node_emb_refined"].
    graph_embedding_name: str
        The graph embedding name. Expected in ["gcn", "gat", "graphsage", "ggnn"].
    decoder_name: str
        The decoder name. Expected in ["stdrnn", "stdtree"].

    Returns
    -------
    template_dict: dict
        The template dict.
        The structure is shown as follows:
            {
                graph_construction_args: dict,
                graph_embedding_args: dict,
                decoder_args: dict
            }
    """
    graph_construction_args = get_graph_construction_args(graph_construction_name)
    graph_initialization_args = get_graph_initialization_args()
    graph_embedding_args = get_graph_embedding_args(graph_embedding_name)
    decoder_args = get_decoder_args(decoder_name)
    ret = {
        "graph_construction_args": graph_construction_args,
        "graph_initialization_args": graph_initialization_args,
        "graph_embedding_args": graph_embedding_args,
        "decoder_args": decoder_args,
    }
    return ret


__all__ = [
    "get_graph_construction_args",
    "get_graph_initialization_args",
    "get_graph_embedding_args",
    "get_decoder_args",
    "get_basic_args",
]
