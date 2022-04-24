from graph4nlp.pytorch.modules.utils.tree_utils import Tree


def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def is_all_same(c1, c2, form_manager):
    all_same = True
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        if all_same:
            return True
    if len(c1) != len(c2) or all_same is False:
        n1 = Tree.norm_tree(Tree.deduplicate_tree(c1, form_manager), form_manager)
        n2 = Tree.norm_tree(Tree.deduplicate_tree(c2, form_manager), form_manager)
        if len(n1) == len(n2):
            all_same = True
            for j in range(len(n1)):
                if n1[j] != n2[j]:
                    all_same = False
                    break
        else:
            return False
        if all_same:
            pass
            # print(" ".join(form_manager.get_idx_symbol_for_list(c1)))
            # print(" ".join(form_manager.get_idx_symbol_for_list(c2)))
        return all_same
    raise NotImplementedError("you should not arrive here!")


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
        if is_all_same(candidate_list[i], reference_list[i], form_manager):
            c = c + 1
        else:
            pass

    return c / float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    candidate_list = []
    for i in range(len(candidate_list_)):
        candidate_list.append(candidate_list_[i])
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(reference_list_[i])
    return compute_accuracy(candidate_list, reference_list, form_manager)
