import operator
from queue import PriorityQueue

import torch
import torch.nn as nn

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.modules.prediction.generation.base import DecoderBase
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab


class StrategyBase(nn.Module):
    def __init__(self):
        super(StrategyBase, self).__init__()

    def generate(self, **kwargs):
        raise NotImplementedError()


class BeamSearchNode(object):
    def __init__(self, hiddenstate, enc_attn_weights_average, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.enc_attn_weights_average = enc_attn_weights_average
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


class BeamSearchStrategy(StrategyBase):
    def __init__(self, beam_size, vocab, decoder: DecoderBase, rnn_type, use_copy=False, use_coverage=False,
                 max_decoder_step=50):
        super(BeamSearchStrategy, self).__init__()
        self.rnn_type = rnn_type
        self.beam_size = beam_size
        self.decoder = decoder
        self.vocab = vocab
        self.use_copy = use_copy
        self.use_coverage = use_coverage
        self.max_decoder_step = max_decoder_step

    def generate(self, graphs: list, oov_dict=None, topk=1):
        """
            Generate sequences using beam search.
        Parameters
        ----------
        graphs: list[GraphData]
        oov_dict: Vocab
        topk: int, default=1

        Returns
        -------
        prediction: list
        """
        params = self.decoder.extract_params(graphs)
        params['tgt_seq'] = None
        params['beam_width'] = self.beam_size
        params['topk'] = topk
        params['oov_dict'] = oov_dict
        return self._beam_search(**params)

    def _beam_search(self, graph_node_embedding, graph_node_mask=None, rnn_node_embedding=None,
                     graph_level_embedding=None,
                     graph_edge_embedding=None, graph_edge_mask=None, tgt_seq=None, src_seq=None, oov_dict=None,
                     beam_width=4, topk=1):

        decoded_results = []

        target_len = self.max_decoder_step

        batch_size = graph_node_embedding.shape[0]
        decoder_input = torch.tensor([self.vocab.SOS] * batch_size).to(graph_node_embedding.device)
        decoder_state = self.decoder.get_decoder_init_state(rnn_type=self.rnn_type, batch_size=batch_size,
                                                            content=graph_level_embedding)

        for idx in range(batch_size):
            if self.rnn_type == "LSTM":
                decoder_state_one = (
                    decoder_state[0][:, idx, :].unsqueeze(1), decoder_state[1][:, idx, :].unsqueeze(1))
            else:
                decoder_state_one = decoder_state[:, idx, :].unsqueeze(1)

            graph_node_embedding_one = graph_node_embedding[idx, :, :].unsqueeze(0)
            graph_node_mask_one = graph_node_mask[idx, :].unsqueeze(0) if graph_node_mask is not None else None
            rnn_node_embedding_one = rnn_node_embedding[idx, :, :].unsqueeze(
                0) if rnn_node_embedding is not None else None

            # Start with the start of the sentence token
            decoder_input_one = decoder_input[idx].view(-1)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_state_one, [], None, decoder_input_one, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > beam_width * target_len + 10: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input_n = n.wordid
                decoder_hidden_n = n.h
                enc_attn_weights_average = n.enc_attn_weights_average

                if n.wordid.item() == self.vocab.EOS and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue
                # decoder_output, decoder_state, dec_attn_scores, coverage_vec
                decoder_input_n = self.decoder._filter_oov(decoder_input_n)
                decoder_output, decoder_hidden_n_out, dec_attn_scores, coverage_vec = \
                    self.decoder.decode_step(decoder_input=decoder_input_n, rnn_state=decoder_hidden_n,
                                             dec_input_mask=graph_node_mask_one,
                                             encoder_out=graph_node_embedding_one, rnn_emb=rnn_node_embedding_one,
                                             enc_attn_weights_average=enc_attn_weights_average,
                                             src_seq=src_seq[idx, :].unsqueeze(0) if src_seq is not None else None,
                                             oov_dict=oov_dict)

                if self.use_coverage:
                    enc_attn_weights_average = [t.clone() for t in enc_attn_weights_average]
                    enc_attn_weights_average.append(dec_attn_scores.unsqueeze(0))
                else:
                    enc_attn_weights_average = []

                decoder_output = torch.log(decoder_output + 1e-31)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden_n_out, enc_attn_weights_average, n, decoded_t,
                                          n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance[1:target_len + 1])

            decoded_results.append(utterances)
        return decoded_results

    def beam_search_for_tree_decoding(self, decoder_initial_state,
                                        decoder_initial_input,
                                        parent_state,
                                        graph_node_embedding,
                                        rnn_node_embedding=None,
                                        src_seq=None,
                                        oov_dict=None,
                                        sibling_state=None,
                                        device=None,
                                        topk=1):

        decoded_results = []

        decoder_hidden = decoder_initial_state
        decoder_input = decoder_initial_input
        form_manager = self.vocab

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, [], None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            if qsize > self.max_decoder_step: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == form_manager.get_symbol_idx(form_manager.end_token) and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
                
            # decode for one step using decoder
            prediction, decoder_hidden, _ = self.decoder.decode_step(dec_single_input=decoder_input,
                                                        dec_single_state=decoder_hidden,
                                                        memory=graph_node_embedding,
                                                        parent_state=parent_state)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(prediction, self.beam_size)
            nextnodes = []

            for new_k in range(self.beam_size):
                decoded_t = torch.tensor([indexes[0][new_k]], dtype=torch.long, device=device)

                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, [], n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_results.append(utterances)
        assert(len(decoded_results) == 1 and len(utterances) == topk)
        return decoded_results