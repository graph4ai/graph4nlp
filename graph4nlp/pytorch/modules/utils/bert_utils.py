from collections import defaultdict, namedtuple
import torch

# When using the sliding window trick for long sequences,
# we take the representation of each token with maximal context.
# Take average of the BERT embeddings of these BPE sub-tokens
# as the embedding for the word.
# Take *weighted* average of the word embeddings through all layers.

def extract_bert_hidden_states(all_encoder_layers, max_doc_len, features, weighted_avg=False):
  # assert all_encoder_layers.requires_grad == False
  num_layers, batch_size, num_chunk, max_token_len, bert_dim = all_encoder_layers.shape
  out_features = torch.Tensor(num_layers, batch_size, max_doc_len, bert_dim).fill_(0)
  device = all_encoder_layers.get_device() if all_encoder_layers.is_cuda else None
  if device is not None:
    out_features = out_features.to(device)

  token_count = []
  # Map BERT tokens to doc words
  for i, ex_feature in enumerate(features): # Example
    ex_token_count = defaultdict(int)
    for j, chunk_feature in enumerate(ex_feature): # Chunk
      for k in chunk_feature.token_is_max_context: # Token
        if chunk_feature.token_is_max_context[k]:
          doc_word_idx = chunk_feature.token_to_orig_map[k]
          out_features[:, i, doc_word_idx] += all_encoder_layers[:, i, j, k]
          ex_token_count[doc_word_idx] += 1
    token_count.append(ex_token_count)

  for i, ex_token_count in enumerate(token_count):
    for doc_word_idx, count in ex_token_count.items():
      out_features[:, i, doc_word_idx] /= count

  # Average through all layers
  if not weighted_avg:
    out_features = torch.mean(out_features, 0)
  return out_features

def convert_text_to_bert_features(text, bert_tokenizer, max_seq_length, doc_stride):
  # The convention in BERT is:
      # (a) For sequence pairs:
      #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
      #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
      # (b) For single sequences:
      #  tokens:   [CLS] the dog is hairy . [SEP]
      #  type_ids: 0   0   0   0  0     0 0

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
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                             split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    feature = BertInputFeatures(
    doc_span_index=doc_span_index,
    tokens=tokens,
    token_to_orig_map=token_to_orig_map,
    token_is_max_context=token_is_max_context,
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

class BertInputFeatures(object):
    """A single set of BERT features of data."""
    def __init__(self,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
