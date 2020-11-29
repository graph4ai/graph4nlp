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


class Hypothesis(object):
    def __init__(self, tokens, log_probs, dec_state, num_non_words, enc_attn_weights, use_coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.dec_state = dec_state
        self.num_non_words = num_non_words
        self.enc_attn_weights = enc_attn_weights
        self.use_coverage = use_coverage
        if not self.use_coverage:
            self.enc_attn_weights = []

    def __repr__(self):
        return repr(self.tokens)

    def __len__(self):
        return len(self.tokens) - self.num_non_words

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.log_probs)

    def create_next(self, token, log_prob, dec_state, non_word, add_enc_attn_weights):
        if self.use_coverage:
            assert len(add_enc_attn_weights.shape) == 3
            assert add_enc_attn_weights.shape[0] == 1
            assert add_enc_attn_weights.shape[1] == 1
            enc_attn_weights_processed = [t.clone() for t in self.enc_attn_weights]
            enc_attn_weights_processed.append(add_enc_attn_weights)
        else:
            enc_attn_weights_processed = []
        return Hypothesis(tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob],
                          dec_state=dec_state,
                          num_non_words=self.num_non_words + 1 if non_word else self.num_non_words,
                          enc_attn_weights=enc_attn_weights_processed, use_coverage=self.use_coverage)



class DecoderStrategy(StrategyBase):
    def __init__(self, beam_size, vocab, decoder: DecoderBase, rnn_type, use_copy=False, use_coverage=False,
                 max_decoder_step=50):
        super(DecoderStrategy, self).__init__()
        self.rnn_type = rnn_type
        self.beam_size = beam_size
        self.decoder = decoder
        self.vocab = vocab
        self.use_copy = use_copy
        self.use_coverage = use_coverage
        self.max_decoder_step = max_decoder_step

    def generate(self, graph_list: list, oov_dict=None, topk=1):
        """
            Generate sequences using beam search.
        Parameters
        ----------
        graph_list: list[GraphData]
        oov_dict: Vocab
        topk: int, default=1

        Returns
        -------
        prediction: list
        """
        params = self.decoder.extract_params(graph_list)
        params['tgt_seq'] = None
        params['beam_size'] = self.beam_size
        params['topk'] = topk
        params['oov_dict'] = oov_dict
        return self._beam_search(**params)

    def _beam_search(self, graph_node_embedding, graph_node_mask=None, rnn_node_embedding=None,
                     graph_level_embedding=None,
                     graph_edge_embedding=None, graph_edge_mask=None, tgt_seq=None, src_seq=None, oov_dict=None,
                     beam_size=4, topk=1):

        assert 0 < topk <= beam_size
        len_in_words = False

        min_out_len = 1
        max_out_len = self.max_decoder_step + 1
        batch_size = graph_node_embedding.shape[0]

        decoder_state = self.decoder.get_decoder_init_state(rnn_type=self.rnn_type, batch_size=batch_size,
                                                            content=graph_level_embedding)

        batch_results = []
        for batch_idx in range(batch_size):
            if self.rnn_type == "lstm":
                single_decoder_state = (
                    decoder_state[0][:, batch_idx, :].unsqueeze(1), decoder_state[1][:, batch_idx, :].unsqueeze(1))
            elif self.rnn_type == "gru":
                single_decoder_state = decoder_state[:, batch_idx, :].unsqueeze(1)
            else:
                raise NotImplementedError("RNN Type {} is not implemented, expected in ``lstm`` or ``gru``"
                                          .format(self.rnn_type))

            single_graph_node_embedding = graph_node_embedding[batch_idx, :, :].unsqueeze(0).expand(beam_size, -1, -1).contiguous()
            single_graph_node_mask = graph_node_mask[batch_idx, :].unsqueeze(0).expand(beam_size, -1, -1).contiguous() if graph_node_mask is not None else None
            single_rnn_node_embedding = rnn_node_embedding[batch_idx, :, :].unsqueeze(
                0).expand(beam_size, -1, -1).contiguous() if rnn_node_embedding is not None else None

            step = 0
            results, backup_results = [], []

            hypos = [Hypothesis([self.vocab.SOS], [], single_decoder_state, 1, [], use_coverage=self.use_coverage)]

            while len(hypos) > 0 and step <= self.max_decoder_step:
                n_hypos = len(hypos)
                if n_hypos < beam_size:
                    hypos.extend(hypos[-1] for _ in range(beam_size - n_hypos)) # check deep copy

                decoder_input = torch.tensor([h.tokens[-1] for h in hypos]).to(graph_node_embedding.device)

                if self.rnn_type == "lstm":
                    single_decoder_state = (
                        torch.cat([h.dec_state[0] for h in hypos], 1), torch.cat([h.dec_state[1] for h in hypos], 1))
                elif self.rnn_type == "gru":
                    single_decoder_state = torch.cat([h.dec_state for h in hypos], 1)
                else:
                    raise NotImplementedError("RNN Type {} is not implemented, expected in ``lstm`` or ``gru``"
                                              .format(self.rnn_type))

                decoder_input = self.decoder._filter_oov(decoder_input)

                enc_tmp = []
                coverage_input = []
                if step > 0  and self.use_coverage:
                    for ii in range(step):
                        coverage_input.append(torch.cat([h.enc_attn_weights[ii] for h in hypos], dim=1))

                decoder_output, single_decoder_state, dec_attn_scores, _ = \
                    self.decoder.decode_step(decoder_input=decoder_input, rnn_state=single_decoder_state,
                                             dec_input_mask=single_graph_node_mask,
                                             encoder_out=single_graph_node_embedding, rnn_emb=single_rnn_node_embedding,
                                             enc_attn_weights_average=coverage_input,
                                             src_seq=src_seq[batch_idx, :].unsqueeze(0) if self.use_copy else None,
                                             oov_dict=oov_dict)
                decoder_output = torch.log(decoder_output + 1e-31)
                top_v, top_i = decoder_output.data.topk(beam_size)
                # print(top_v.shape, decoder_output.shape, dec_attn_scores.shape, "-----")

                new_hypos = []
                for in_idx in range(n_hypos):
                    for out_idx in range(beam_size):
                        new_tok = top_i[in_idx][out_idx].item()
                        new_prob = top_v[in_idx][out_idx].item()
                        new_enc_attn_weights = dec_attn_scores[in_idx, :].unsqueeze(0).unsqueeze(0)

                        non_word = new_tok == self.vocab.EOS  # only SOS & EOS don't count

                        if self.rnn_type == "lstm":
                            tmp_decoder_state = (
                                single_decoder_state[0][:, in_idx, :].unsqueeze(1),
                                single_decoder_state[1][:, in_idx, :].unsqueeze(1))
                        elif self.rnn_type == "gru":
                            tmp_decoder_state = single_decoder_state[:, in_idx, :].unsqueeze(1)
                        else:
                            raise NotImplementedError("RNN Type {} is not implemented, expected in ``lstm`` or ``gru``"
                                                      .format(self.rnn_type))

                        new_hypo = hypos[in_idx].create_next(token=new_tok, log_prob=new_prob,
                                                             dec_state=tmp_decoder_state, non_word=non_word,
                                                             add_enc_attn_weights=new_enc_attn_weights)
                        new_hypos.append(new_hypo)

                # Block sequences with repeated ngrams
                # block_ngram_repeats(new_hypos, block_ngram_repeat)

                # process the new hypotheses
                new_hypos = sorted(new_hypos, key=lambda h: -h.avg_log_prob)[:beam_size]
                hypos = []
                new_complete_results, new_incomplete_results = [], []
                for nh in new_hypos:
                    length = len(nh)  # Does not count SOS and EOS
                    if nh.tokens[-1] == self.vocab.EOS:  # a complete hypothesis
                        if len(new_complete_results) < beam_size and min_out_len <= length <= max_out_len:
                            new_complete_results.append(nh)
                    elif len(hypos) < beam_size and length < max_out_len:  # an incomplete hypothesis
                        hypos.append(nh)
                    elif length == max_out_len and len(new_incomplete_results) < beam_size:
                        new_incomplete_results.append(nh)
                if new_complete_results:
                    results.extend(new_complete_results)
                elif new_incomplete_results:
                    backup_results.extend(new_incomplete_results)
                step += 1
            if not results:  # if no sequence ends with EOS within desired length, fallback to sequences
                results = backup_results  # that are "truncated" at the end to max_out_len
            batch_results.append(sorted(results, key=lambda h: -h.avg_log_prob)[:topk])
        ret = torch.zeros(batch_size, topk, self.max_decoder_step).long()
        for sent_id, each in enumerate(batch_results):
            for i in range(topk):
                ids = torch.Tensor(each[i].tokens[1:])[:self.max_decoder_step]
                if ids.shape[0] < self.max_decoder_step:
                    pad = torch.zeros(self.max_decoder_step - ids.shape[0])
                    ids = torch.cat((ids, pad), dim=0)
                ret[sent_id, i, :] = ids
        return ret

    def beam_search_for_tree_decoding(self, decoder_initial_state,
                                        decoder_initial_input,
                                        parent_state,
                                        graph_node_embedding,
                                        rnn_node_embedding=None,
                                        src_seq=None,
                                        oov_dict=None,
                                        sibling_state=None,
                                        device=None,
                                        topk=1,
                                        enc_batch=None):

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
        nodes.put((-node.eval(), torch.randn(1).item(), node))
        qsize = 1

        # start beam search
        while True:
            if qsize > self.max_decoder_step: break

            # fetch the best node
            score, _nounce, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == form_manager.get_symbol_idx(form_manager.end_token) and n.prevNode != None:
                endnodes.append((score, _nounce, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
                
            # decode for one step using decoder
            prediction, decoder_hidden, _ = self.decoder.decode_step(tgt_batch_size=1, 
                                                        dec_single_input=decoder_input,
                                                        dec_single_state=decoder_hidden,
                                                        memory=graph_node_embedding,
                                                        parent_state=parent_state,
                                                        oov_dict=oov_dict,
                                                        enc_batch=enc_batch)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(prediction, self.beam_size)
            nextnodes = []

            for new_k in range(self.beam_size):
                decoded_t = torch.tensor([indexes[0][new_k]], dtype=torch.long, device=device)

                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, [], n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, torch.randn(1).item(), node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, _nounce, nn = nextnodes[i]
                nodes.put((score, _nounce, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, _, n in sorted(endnodes, key=operator.itemgetter(0)):
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