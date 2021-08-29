import copy


class Node:
    """
    Class for self-defined node data structure

    ...

    Attributes
    ----------
    word_ : (str)
        The text contained in this node.

    type_ : (int)
        Node type in graph data structure, 0 for word nodes, 1 for constituency nodes,\
             2~n can be defined by other node types.

    id_ : (int)
        Unique identifier for nodes.

    sentence_ : (int)
        To illustrate which sentence the node is related when multiple sentences included\
             in a graph.
    """

    def __init__(self, word_, type_, id_, sentence_):
        # word: this node's text
        self.word = word_

        # type: 0 for word nodes, 1 for constituency nodes  # noqa
        self.type = type_

        # id: unique identifier for every node
        self.id = id_

        self.head = False

        self.tail = False

        self.sentence = sentence_

    def __str__(self):
        return (
            self.word
            + "_type_"
            + str(self.type)
            + "_id_"
            + str(self.id)
            + "_head_"
            + str(self.head)
            + "_tail_"
            + str(self.tail)
            + "_sentence_"
            + str(self.sentence)
        )


class UtilityFunctionsForGraph:
    """
    A class for utility functions of graph operations.
    """

    @staticmethod
    def get_head_node(g, sentence_id):
        """Return the head node in a syntactic graph, to be used when we need to connect multiple\
             graphs into one merged graph."""
        for n in g.nodes():
            if (n.head is True) and (n.sentence == sentence_id):
                return n

    @staticmethod
    def get_tail_node(g, sentence_id):
        """Return the tail node in a syntactic graph, to be used when we need to connect multiple\
             graphs into one merged graph."""
        for n in g.nodes():
            if (n.tail is True) and (n.sentence == sentence_id):
                return n

    @staticmethod
    def cut_root_node(g):
        """Remove the \"Root\" node in a graph."""
        pass

    @staticmethod
    def cut_pos_node(g):
        """Remove some part-of-speech nodes in a graph, e.g., \"NN\", \"JJ\"."""
        node_arr = list(g.nodes())
        del_arr = []
        for n in node_arr:
            edge_arr = list(g.edges())
            cnt_in = 0
            cnt_out = 0
            for e in edge_arr:
                if n.id == e[0].id:
                    cnt_out += 1
                    out_ = e[1]
                if n.id == e[1].id:
                    cnt_in += 1
                    in_ = e[0]
            if cnt_in == 1 and cnt_out == 1 and out_.type == 0:
                del_arr.append((n, in_, out_))
        for d in del_arr:
            g.remove_node(d[0])
            g.add_edge(d[1], d[2])
        return g

    @staticmethod
    def cut_line_node(g):
        """Remove intermediate nodes organized like a line."""
        node_arr = list(g.nodes())

        for n in node_arr:
            edge_arr = list(g.edges())
            cnt_in = 0
            cnt_out = 0
            for e in edge_arr:
                if n.id == e[0].id:
                    cnt_out += 1
                    out_ = e[1]
                if n.id == e[1].id:
                    cnt_in += 1
                    in_ = e[0]
            if cnt_in == 1 and cnt_out == 1:
                g.remove_node(n)
                g.add_edge(in_, out_)
        return g

    @staticmethod
    def get_seq_nodes(g):
        """Return word nodes in a syntactic graph."""
        res = []
        node_arr = list(g.nodes())
        for n in node_arr:
            if n.type == 0:
                res.append(copy.deepcopy(n))
        return sorted(res, key=lambda x: x.id)

    @staticmethod
    def get_non_seq_nodes(g):
        """Return non-word nodes (like constituency tag nodes) in a syntactic graph."""
        res = []
        node_arr = list(g.nodes())
        for n in node_arr:
            if n.type != 0:
                res.append(copy.deepcopy(n))
        return sorted(res, key=lambda x: x.id)

    @staticmethod
    def get_all_text(g):
        """Return all nodes text strings in a string list."""
        seq_arr = UtilityFunctionsForGraph.get_seq_nodes(g)
        nonseq_arr = UtilityFunctionsForGraph.get_non_seq_nodes(g)
        seq = [x.word for x in seq_arr]
        nonseq = [x.word for x in nonseq_arr]
        return seq + nonseq

    @staticmethod
    def get_all_id(g):
        """Return all nodes id."""
        seq_arr = UtilityFunctionsForGraph.get_seq_nodes(g)
        nonseq_arr = UtilityFunctionsForGraph.get_non_seq_nodes(g)
        seq = [x.id for x in seq_arr]
        nonseq = [x.id for x in nonseq_arr]
        return seq + nonseq

    @staticmethod
    def get_id2word(g):
        """Return a dict for mapping from nodes id to word text."""
        res = {}
        seq_arr = UtilityFunctionsForGraph.get_seq_nodes(g)
        nonseq_arr = UtilityFunctionsForGraph.get_non_seq_nodes(g)
        for x in seq_arr:
            res[x.id] = x.word
        for x in nonseq_arr:
            res[x.id] = x.word
        return res

    @staticmethod
    def print_edges(g):
        """Display all edges text and id in a graph."""
        edge_arr = list(g.edges())
        for e in edge_arr:
            print(e[0].word, e[1].word), (e[0].id, e[1].id)

    @staticmethod
    def print_nodes(g):
        """Display all nodes in a graph."""
        nodes_arr = list(g.nodes())
        for n in nodes_arr:
            print(n)
