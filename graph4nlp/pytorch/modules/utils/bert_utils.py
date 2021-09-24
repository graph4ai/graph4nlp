from collections import defaultdict, namedtuple
from dataclasses import dataclass

from typing import Any, Dict, List

import numpy as np
import torch

from transformers import PreTrainedTokenizerBase

# When using the sliding window trick for long sequences,
# we take the representation of each token with maximal context.
# Take average of the BERT embeddings of these BPE sub-tokens
# as the embedding for the word.
# Take *weighted* average of the word embeddings through all layers.

def extract_bert_hidden_states(all_encoder_layers, bert_token_to_orig_map, weighted_avg=False):
  """Extract BERT hidden states.

  Parameters
  ----------
  all_encoder_layers : torch.Tensor
      All layer outputs of the BERT encoder.
  bert_token_to_orig_map : torch.Tensor
      The mapping between BERT wordpiece tokens to original tokens.
  weighted_avg : boolean
    Specify whether to compute the weighted average (with leranable weight vector) or average of
    the embeddings of all hidden layers, default: ``False``.

  Returns
  -------
  torch.Tensor
      The output BERT embeddings.
  """
  # assert all_encoder_layers.requires_grad == False
  _, max_nb_strides, seq_len, bert_dim = all_encoder_layers.shape
  _, _, _, max_doc_len = bert_token_to_orig_map.shape
  gpu = all_encoder_layers.device

  # bert_token_to_orig_map        -> NB_DOCS, MAX_NB_STRIDES, SEQ_LEN, MAX_DOC_LEN
  # bert_token_to_orig_map.unsqueeze(0).transpose(-1, -2) -> 1, NB_DOCS, MAX_NB_STRIDES, MAX_DOC_LEN, SEQ_LEN
  # all_encoder_layers                                    -> NB_DOCS, MAX_NB_STRIDES, SEQ_LEN, BERT_DIM
  # Now we have NB_DOCS = 1 (see batch process in embedding_construction.py) /

  mapping = bert_token_to_orig_map.unsqueeze(0).transpose(-1, -2)
  # mapping                  -> 1, 1, MAX_NB_STRIDES, MAX_DOC_LEN, SEQ_LEN
  # all_encoder_layers       -> 1, MAX_NB_STRIDES, SEQ_LEN, BERT_DIM

  # When mapping is sparse, it does not work (matmul supports 2-D sparse, not 5-D)
  # We'll broadcast manually
  out_features = torch.zeros(max_nb_strides, max_doc_len, bert_dim).cuda(device=gpu)
  for stride in range(mapping.shape[2]):
    out_features[stride, :, :] = torch.matmul(mapping[0][0][stride], all_encoder_layers[0][stride])

  out_features = out_features.unsqueeze(0).unsqueeze(0)
  # out_features  -> 1, 1, MAX_NB_STRIDES, MAX_DOC_LEN, BERT_DIM
  # torch.sum(bert_token_to_orig_map, (1, 2)) does not work with sparse
  # since bert_token_to_orig_map is full of 0 and 1, we have a workaround
  out_features = torch.sum(out_features, 2) / torch.clamp(torch.sparse.sum(bert_token_to_orig_map, [1, 2]).to_dense(), min=1).unsqueeze(-1).unsqueeze(0)
  result = out_features.cpu()

  # Memory Management...
  out_features.detach()
  torch.cuda.empty_cache()

  return result


@dataclass
class FastBertInputFeatures(object):
  input_ids: np.ndarray     # 2-D NUM_STRIDES, MAX_SEQ_LEN
  input_mask: np.ndarray    # 2-D NUM_STRIDES, MAX_SEQ_LEN


def fast_convert_text_to_bert(text: str,
                              bert_tokenizer: PreTrainedTokenizerBase,
                              max_seq_length: int,
                              doc_stride: int):
  """
  Helper function, but FASTer
  """
  encoding = bert_tokenizer(text=text,
                            return_tensors="np",
                            return_token_type_ids=False,
                            return_attention_mask=True,
                            verbose=False)

  def _in_strides(a: np.ndarray) -> np.ndarray:
    # extra padding for a smooth sliding window
    a_padded = np.pad(a, (0, max_seq_length - a.shape[0] % doc_stride), mode='constant', constant_values=0)
    a_strided = np.lib.stride_tricks.sliding_window_view(x=a_padded, window_shape=(max_seq_length,))[::doc_stride]
    return a_strided

  strides_input_ids, strides_input_mask = list(map(_in_strides, [encoding.input_ids[0], encoding.attention_mask[0]]))
  return FastBertInputFeatures(input_ids=strides_input_ids, input_mask=strides_input_mask)


def convert_text_to_bert_features(text, bert_tokenizer, max_seq_length, doc_stride):
  """Helper function to convert text to BERT features.
  # The convention in BERT is:
      # (a) For sequence pairs:
      #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
      #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
      # (b) For single sequences:
      #  tokens:   [CLS] the dog is hairy . [SEP]
      #  type_ids: 0   0   0   0  0     0 0
  """
  tok_to_orig_index = []
  all_doc_tokens = []
  for (i, token) in enumerate(text):
    sub_tokens = bert_tokenizer.wordpiece_tokenizer.tokenize(token.lower())
    for sub_ in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_)

  # The -2 accounts for [CLS] and [SEP]
  max_tokens_for_doc = max_seq_length - 2

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of the up to our max length with a stride of `doc_stride`.
  _DocSpan = namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    if length > max_tokens_for_doc:
      length = max_tokens_for_doc
    doc_spans.append(_DocSpan(start=start_offset, length=length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, doc_stride)

  out_features = []
  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_to_orig_map_matrix = []
    # token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                             split_token_index)
      # token_is_max_context[len(tokens)] = is_max_context
      tmp_vec = np.zeros(len(text))
      if is_max_context:
        tmp_vec[tok_to_orig_index[split_token_index]] = 1
      token_to_orig_map_matrix.append(tmp_vec)
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    curr_len = len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * curr_len

    # pad input_ids and mask to max_seq_length
    input_ids += [0] * (max_seq_length - curr_len)
    input_mask += [0] * (max_seq_length - curr_len)

    feature = BertInputFeatures(
      doc_span_index=doc_span_index,
      tokens=tokens,
      token_to_orig_map=token_to_orig_map,
      token_to_orig_map_matrix=token_to_orig_map_matrix,
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
    out_features.append(feature)

  return out_features


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


@dataclass
class BertInputFeatures(object):
  doc_span_index: int
  tokens: List[str]
  token_to_orig_map: Dict[int, int]
  token_to_orig_map_matrix: List[np.ndarray]
  input_ids: List[int]
  input_mask: List[int]
  segment_ids: List[int]
