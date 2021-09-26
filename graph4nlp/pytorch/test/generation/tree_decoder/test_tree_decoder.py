import random
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn, optim

from ....modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder, create_mask
from ....modules.utils.tree_utils import DataLoader, Tree, to_cuda

warnings.filterwarnings("ignore")


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        enc_emb_size,
        enc_hidden_size,
        dec_hidden_size,
        dropout_input,
        dropout_output,
        pad_idx=0,
        encode_rnn_num_layer=1,
        embeddings=None,
    ):
        super(SequenceEncoder, self).__init__()
        if embeddings is None:
            self.embedding = nn.Embedding(input_size, enc_emb_size, padding_idx=pad_idx)
        else:
            self.embedding = embeddings

        self.rnn = nn.LSTM(
            enc_emb_size,
            enc_hidden_size,
            encode_rnn_num_layer,
            bias=True,
            batch_first=True,
            dropout=dropout_output,
            bidirectional=True,
        )
        self.fc = nn.Linear(enc_hidden_size * 4, dec_hidden_size)
        self.dropout = None
        if dropout_input > 0:
            self.dropout = nn.Dropout(dropout_input)

    def forward(self, input_src):
        # batch_size x src_length x emb_size
        src_emb = self.dropout(self.embedding(input_src))
        # output: [batch size, src len, hid dim * num directions],
        # hidden: a tuple of length "n layers * num directions",
        # each element in tuple is batch_size * enc_hidden_size
        output, (hn, cn) = self.rnn(src_emb)
        # this can be concatennate or add operation.
        hn_output = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        cn_output = torch.cat((cn[-2, :, :], cn[-1, :, :]), dim=1)

        # hidden = torch.tanh(self.fc(
        #     torch.cat((hn[-2, :, :], hn[-1, :, :], cn[-2, :, :], cn[-1, :, :]), dim=1)))
        return output, (hn_output, cn_output)


class AttnUnit(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type, dropout):
        super(AttnUnit, self).__init__()
        self.hidden_size = hidden_size
        self.separate_attention = attention_type != "uniform"

        if self.separate_attention == "separate_different_encoder_type":
            self.linear_att = nn.Linear(3 * self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.linear_out = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0, 2, 1), attention)

        if self.separate_attention == "separate_different_encoder_type":
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0, 2, 1), attention_2)

        if self.separate_attention == "separate_different_encoder_type":
            hid = F.tanh(
                self.linear_att(
                    torch.cat((enc_attention.squeeze(2), enc_attention_2.squeeze(2), dec_s_top), 1)
                )
            )
        else:
            hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2), dec_s_top), 1)))
        h2y_in = hid

        h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)

        return pred


def train_tree_decoder(seed, save_every):
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    """loading data"""
    data_dir = r"/home/lishucheng/Graph4AI/graph4ai/graph4nlp/pytorch/test/generation/tree_decoder/data/jobs640/"  # noqa
    src_vocab_file = r"/home/lishucheng/Graph4AI/graph4ai/graph4nlp/pytorch/test/generation/tree_decoder/data/jobs640/vocab.q.txt"  # noqa
    tgt_vocab_file = r"/home/lishucheng/Graph4AI/graph4ai/graph4nlp/pytorch/test/generation/tree_decoder/data/jobs640/vocab.f.txt"  # noqa
    data_file = r"/home/lishucheng/Graph4AI/graph4ai/graph4nlp/pytorch/test/generation/tree_decoder/data/jobs640/train.txt"  # noqa
    mode = "train"
    min_freq = 2
    max_vocab_size = 10000
    batch_size = 20
    device = None
    use_copy = True
    use_coverage = True
    train_data_loader = DataLoader(
        data_dir=data_dir,
        use_copy=use_copy,
        src_vocab_file=src_vocab_file,
        tgt_vocab_file=tgt_vocab_file,
        data_file=data_file,
        mode=mode,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        batch_size=batch_size,
        device=device,
    )
    # print(train_data_loader.share_vocab.idx2symbol)
    # print(train_data_loader.share_vocab.idx2symbol)

    """For encoder-decoder"""
    if use_copy:
        enc_emb_size = 300
        tgt_emb_size = 300
        embeddings = nn.Embedding(
            train_data_loader.share_vocab.vocab_size,
            enc_emb_size,
            padding_idx=train_data_loader.share_vocab.get_symbol_idx(
                train_data_loader.share_vocab.pad_token
            ),
        )
        input_size = train_data_loader.share_vocab.vocab_size
        output_size = train_data_loader.share_vocab.vocab_size
        enc_hidden_size = 150
        dec_hidden_size = 300
    else:
        embeddings = None
        input_size = train_data_loader.src_vocab.vocab_size
        output_size = train_data_loader.tgt_vocab.vocab_size
        enc_emb_size = 150
        tgt_emb_size = 150
        enc_hidden_size = 150
        dec_hidden_size = 300

    enc_dropout_input = 0.1
    enc_dropout_output = 0.3
    dec_dropout_input = 0.1
    # dec_dropout_input = 0
    dec_dropout_output = 0.3
    attn_dropout = 0.1
    # teacher_force_ratio = 0.3
    teacher_force_ratio = 1
    max_dec_seq_length = 220
    max_dec_tree_depth = 220

    encoder = SequenceEncoder(
        input_size=input_size,
        enc_emb_size=enc_emb_size,
        enc_hidden_size=enc_hidden_size,
        dec_hidden_size=dec_hidden_size,
        pad_idx=train_data_loader.src_vocab.get_symbol_idx(train_data_loader.src_vocab.pad_token),
        dropout_input=enc_dropout_input,
        dropout_output=enc_dropout_output,
        encode_rnn_num_layer=1,
        embeddings=embeddings,
    )

    attention_type = "uniform"

    # criterion = nn.NLLLoss(size_average=False,
    # ignore_index=train_data_loader.tgt_vocab.get_symbol_idx(
    #   train_data_loader.tgt_vocab.pad_token))
    criterion = nn.NLLLoss(size_average=False)

    attn_unit = AttnUnit(dec_hidden_size, output_size, attention_type, attn_dropout)

    if not use_copy:
        tree_decoder = StdTreeDecoder(
            attn=attn_unit,
            attn_type="uniform",
            embeddings=embeddings,
            enc_hidden_size=enc_hidden_size,
            dec_emb_size=tgt_emb_size,
            dec_hidden_size=dec_hidden_size,
            output_size=output_size,
            device=device,
            criterion=criterion,
            teacher_force_ratio=teacher_force_ratio,
            use_sibling=True,
            use_attention=True,
            use_copy=use_copy,
            use_coverage=use_coverage,
            fuse_strategy="average",
            num_layers=1,
            dropout_input=dec_dropout_input,
            dropout_output=dec_dropout_output,
            rnn_type="lstm",
            max_dec_seq_length=max_dec_seq_length,
            max_dec_tree_depth=max_dec_tree_depth,
            tgt_vocab=train_data_loader.tgt_vocab,
        )
    else:
        tree_decoder = StdTreeDecoder(
            attn=attn_unit,
            attn_type="uniform",
            embeddings=embeddings,
            enc_hidden_size=enc_hidden_size,
            dec_emb_size=tgt_emb_size,
            dec_hidden_size=dec_hidden_size,
            output_size=output_size,
            device=device,
            criterion=criterion,
            teacher_force_ratio=teacher_force_ratio,
            use_sibling=True,
            use_attention=True,
            use_copy=use_copy,
            use_coverage=use_coverage,
            fuse_strategy="average",
            num_layers=1,
            dropout_input=dec_dropout_input,
            dropout_output=dec_dropout_output,
            rnn_type="lstm",
            max_dec_seq_length=max_dec_seq_length,
            max_dec_tree_depth=max_dec_tree_depth,
            tgt_vocab=train_data_loader.tgt_vocab,
        )

    to_cuda(encoder, device)
    to_cuda(tree_decoder, device)

    init_weight = 0.08

    print("encoder initializing...")
    for name, param in encoder.named_parameters():
        print(name, param.size())
        if param.requires_grad:
            if ("embedding.weight" in name) or ("bert_embedding" in name):
                pass
            else:
                if len(param.size()) >= 2:
                    if "rnn" in name:
                        init.orthogonal_(param)
                    else:
                        init.xavier_uniform_(param, gain=1.0)
                else:
                    init.uniform_(param, -init_weight, init_weight)

    print("decoder initializing...")
    for name, param in tree_decoder.named_parameters():
        print(name, param.size())
        if param.requires_grad:
            if "rnn" in name and len(param.size()) >= 2:
                init.orthogonal_(param)
            else:
                init.uniform_(param, -init_weight, init_weight)

    max_epochs = 300
    epoch = 0
    grad_clip = 5
    optim_state = {"learningRate": 1e-3, "weight_decay": 1e-5}

    print("using adam")
    encoder_optimizer = optim.Adam(
        encoder.parameters(),
        lr=optim_state["learningRate"],
        weight_decay=optim_state["weight_decay"],
    )

    decoder_optimizer = optim.Adam(tree_decoder.parameters(), lr=optim_state["learningRate"])

    print("Starting training.")
    encoder.train()
    tree_decoder.train()

    iterations = max_epochs * train_data_loader.num_batch
    start_time = time.time()

    # best_val_acc = 0
    checkpoint_dir = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph\
        4nlp\pytorch\test\generation\tree_decoder\checkpoint_dir"

    print("Batch number per Epoch:", train_data_loader.num_batch)
    print_every = train_data_loader.num_batch

    loss_to_print = 0
    for i in range(iterations):
        if (i + 1) % train_data_loader.num_batch == 0:
            epoch += 1

        epoch = i // train_data_loader.num_batch

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        enc_batch, enc_len_batch, dec_tree_batch = train_data_loader.random_batch()

        output_, hidden_ = encoder(enc_batch)
        encode_output_dict = {
            "graph_node_embedding": None,
            "graph_node_mask": None,
            "graph_edge_embedding": None,
            "rnn_node_embedding": output_,
            "graph_level_embedding": hidden_,
            "graph_edge_mask": None,
        }
        loss = tree_decoder(encode_output_dict, dec_tree_batch, enc_batch)

        loss.backward()
        torch.nn.utils.clip_grad_value_(encoder.parameters(), grad_clip)
        torch.nn.utils.clip_grad_value_(tree_decoder.parameters(), grad_clip)
        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_to_print += loss

        if (i + 1) % print_every == 0:
            end_time = time.time()
            print(
                (
                    "{}/{}, train_loss = {}, epochs = {}, time since last print = {}".format(
                        i,
                        iterations,
                        (loss_to_print / print_every),
                        epoch,
                        (end_time - start_time) / 60,
                    )
                )
            )
            loss_to_print = 0
            start_time = time.time()

        if i == iterations - 1 or ((i + 1) % (save_every * train_data_loader.num_batch) == 0):
            print("saving model...")
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = tree_decoder
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(checkpoint, "{}/s2t".format(checkpoint_dir) + str(i))

        if loss != loss:
            print("loss is NaN.  This usually indicates a bug.")
            break


def test_tree_decoder(seed, save_every):
    """loading data"""
    data_dir = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\ \
        pytorch\test\generation\tree_decoder\data\jobs640"
    src_vocab_file = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp \
        \pytorch\test\generation\tree_decoder\data\jobs640\vocab.q.txt"
    tgt_vocab_file = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp \
        \pytorch\test\generation\tree_decoder\data\jobs640\vocab.f.txt"
    data_file = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\ \
        pytorch\test\generation\tree_decoder\data\jobs640\test.txt"
    checkpoint_dir = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\ \
        pytorch\test\generation\tree_decoder\checkpoint_dir"

    mode = "test"
    min_freq = 2
    max_vocab_size = 10000
    batch_size = 1
    device = None
    max_dec_seq_length = 220
    max_dec_tree_depth = 100
    use_copy = True
    # use_coverage = True

    test_data_loader = DataLoader(
        data_dir=data_dir,
        use_copy=use_copy,
        src_vocab_file=src_vocab_file,
        tgt_vocab_file=tgt_vocab_file,
        data_file=data_file,
        mode=mode,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        batch_size=batch_size,
        device=device,
    )
    # print(test_data_loader.share_vocab.idx2symbol)
    # print("test samples number : ", test_data_loader.num_batch)

    num_batch = 25
    # model_num = save_every * num_batch - 1
    model_num = 1199

    if use_copy:
        # enc_emb_size = 300
        # tgt_emb_size = 300
        enc_hidden_size = 150
        dec_hidden_size = 300
    else:
        # enc_emb_size = 150
        # tgt_emb_size = 150
        enc_hidden_size = 150
        dec_hidden_size = 300

    max_acc = 0
    while True:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        try:
            checkpoint = torch.load("{}/s2t".format(checkpoint_dir) + str(model_num))
            print("model : s2t" + str(model_num))
        except BaseException:
            break
        encoder = checkpoint["encoder"]
        tree_decoder = checkpoint["decoder"]

        encoder.eval()
        tree_decoder.eval()

        reference_list = []
        candidate_list = []

        data = test_data_loader.data
        for i in range(len(data)):
            x = data[i]
            reference = torch.tensor(x[1], dtype=torch.long)
            input_word_list = x[0]
            # print(test_data_loader.tgt_vocab.get_idx_symbol_for_list(reference))
            candidate = do_generate(
                use_copy,
                enc_hidden_size,
                dec_hidden_size,
                encoder,
                tree_decoder,
                input_word_list,
                test_data_loader.src_vocab,
                test_data_loader.tgt_vocab,
                device,
                max_dec_seq_length,
                max_dec_tree_depth,
            )
            candidate = [int(c) for c in candidate]
            num_left_paren = sum(
                1 for c in candidate if test_data_loader.tgt_vocab.idx2symbol[int(c)] == "("
            )
            num_right_paren = sum(
                1 for c in candidate if test_data_loader.tgt_vocab.idx2symbol[int(c)] == ")"
            )
            diff = num_left_paren - num_right_paren
            if diff > 0:
                for _ in range(diff):
                    candidate.append(test_data_loader.tgt_vocab.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]
            # ref_str = convert_to_string(reference, test_data_loader.tgt_vocab)
            # cand_str = convert_to_string(candidate, test_data_loader.tgt_vocab)
            reference_list.append(reference)
            candidate_list.append(candidate)
            # print(cand_str)

        val_acc = compute_tree_accuracy(candidate_list, reference_list, test_data_loader.tgt_vocab)
        print("ACCURACY = {}\n".format(val_acc))
        if val_acc >= max_acc:
            max_acc = val_acc
            max_index = model_num

        model_num += save_every * num_batch

    print("max accuracy:", max_acc)
    best_model = torch.load(checkpoint_dir + "s2t" + str(max_index))
    torch.save(best_model, checkpoint_dir + "best_model")


def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def do_generate(
    use_copy,
    enc_hidden_size,
    dec_hidden_size,
    encoder,
    tree_decoder,
    enc_w_list,
    word_manager,
    form_manager,
    device,
    max_dec_seq_length,
    max_dec_tree_depth,
):
    # initialize the rnn state to all zeros
    prev_c = torch.zeros((1, dec_hidden_size), requires_grad=False)
    prev_h = torch.zeros((1, dec_hidden_size), requires_grad=False)

    to_cuda(prev_c, device)
    to_cuda(prev_h, device)

    # reversed order
    enc_w_list = list(np.array(enc_w_list)[::-1])
    enc_w_list.append(word_manager.get_symbol_idx(word_manager.end_token))
    enc_w_list.insert(0, word_manager.get_symbol_idx(word_manager.start_token))
    enc_w_list = torch.tensor(enc_w_list, dtype=torch.long).unsqueeze(0)

    # print(form_manager.get_idx_symbol_for_list(enc_w_list[0]))

    enc_outputs = torch.zeros((1, enc_w_list.size(1), enc_hidden_size), requires_grad=False)
    to_cuda(enc_outputs)

    output_, (hn_, cn_) = encoder(enc_w_list)
    # print(output_.size())

    enc_outputs = output_
    prev_c = cn_
    prev_h = hn_

    # decode
    queue_decode = []
    queue_decode.append({"s": (prev_c, prev_h), "parent": 0, "child_index": 1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <= max_dec_tree_depth:
        s = queue_decode[head - 1]["s"]
        parent_h = s[1]
        t = queue_decode[head - 1]["t"]

        sibling_state = torch.zeros((1, dec_hidden_size), dtype=torch.float, requires_grad=False)
        sibling_state = to_cuda(sibling_state)

        flag_sibling = False
        for q_index in range(len(queue_decode)):
            if (
                (head <= len(queue_decode))
                and (q_index < head - 1)
                and (queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"])
                and (queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"])
            ):
                flag_sibling = True
                sibling_index = q_index
        if flag_sibling:
            sibling_state = queue_decode[sibling_index]["s"][1]

        if head == 1:
            prev_word = torch.tensor(
                [form_manager.get_symbol_idx(form_manager.start_token)], dtype=torch.long
            )
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx("(")], dtype=torch.long)

        to_cuda(prev_word, device)

        i_child = 1

        if use_copy:
            enc_context = None
            input_mask = create_mask(
                torch.LongTensor([enc_outputs.size(1)] * enc_outputs.size(0)),
                enc_outputs.size(1),
                device,
            )
            decoder_state = (s[0].unsqueeze(0), s[1].unsqueeze(0))

        while True:
            if not use_copy:
                curr_c, curr_h = tree_decoder.rnn(prev_word, s[0], s[1], parent_h, sibling_state)
                prediction = tree_decoder.attention(enc_outputs, curr_h, torch.tensor(0))
                s = (curr_c, curr_h)
                _, _prev_word = prediction.max(1)
                prev_word = _prev_word
            else:
                # print(form_manager.idx2symbol[np.array(prev_word)[0]])
                decoder_embedded = tree_decoder.embeddings(prev_word)
                pred, decoder_state, _, _, enc_context = tree_decoder.rnn(
                    parent_h,
                    sibling_state,
                    decoder_embedded,
                    decoder_state,
                    enc_outputs.transpose(0, 1),
                    None,
                    None,
                    input_mask=input_mask,
                    encoder_word_idx=enc_w_list,
                    ext_vocab_size=tree_decoder.embeddings.num_embeddings,
                    log_prob=False,
                    prev_enc_context=enc_context,
                    encoder_outputs2=output_.transpose(0, 1),
                )

                dec_next_state_1 = decoder_state[0].squeeze(0)
                dec_next_state_2 = decoder_state[1].squeeze(0)

                pred = torch.log(pred + 1e-31)
                prev_word = pred.argmax(1)

            if (
                int(prev_word[0]) == form_manager.get_symbol_idx(form_manager.end_token)
                or t.num_children >= max_dec_seq_length
            ):
                break
            elif int(prev_word[0]) == form_manager.get_symbol_idx(form_manager.non_terminal_token):
                # print("we predicted N");exit()
                queue_decode.append(
                    {
                        "s": (dec_next_state_1.clone(), dec_next_state_2.clone()),
                        "parent": head,
                        "child_index": i_child,
                        "t": Tree(),
                    }
                )
                t.add_child(int(prev_word[0]))
            else:
                t.add_child(int(prev_word[0]))
            i_child = i_child + 1
        head = head + 1
    # refine the root tree (TODO, what is this doing?)
    for i in range(len(queue_decode) - 1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"] - 1]["t"].children[cur["child_index"] - 1] = cur["t"]
    return queue_decode[0]["t"].to_list(form_manager)


def is_all_same(c1, c2):
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        return all_same
    else:
        return False


def compute_accuracy(candidate_list, reference_list, form_manager):
    if len(candidate_list) != len(reference_list):
        print(
            "candidate list has length {}, reference list has length {}\n".format(
                len(candidate_list), len(reference_list)
            )
        )

    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    for i in range(len_min):
        if is_all_same(candidate_list[i], reference_list[i]):
            c = c + 1
        else:
            pass

    return c / float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    candidate_list = []
    for i in range(len(candidate_list_)):
        candidate_list.append(
            Tree.norm_tree(candidate_list_[i], form_manager).to_list(form_manager)
        )
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(
            Tree.norm_tree(reference_list_[i], form_manager).to_list(form_manager)
        )
    return compute_accuracy(candidate_list, reference_list, form_manager)


if __name__ == "__main__":
    train_tree_decoder(1234, 2)
    test_tree_decoder(1234, 2)
