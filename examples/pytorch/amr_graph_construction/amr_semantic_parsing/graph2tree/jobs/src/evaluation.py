from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase


class ExactMatch(EvaluationMetricBase):
    def __init__(self):
        super(ExactMatch, self).__init__()

    def calculate_scores(self, ground_truth, predict):
        correct = 0
        assert len(ground_truth) == len(predict)
        for gt, pred in zip(ground_truth, predict):
            gt = gt.strip()
            pred = pred.strip()
            if gt == pred:
                correct += 1.0
        return correct / len(ground_truth)


def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def get_split_comma(input_str):
    input_str = input_str.replace(",", " , ")
    input_list = [item.strip() for item in input_str.split()]
    ref_char = "$"
    for index in range(len(input_list)):
        if input_list[index] == ",":
            if input_list[:index].count("(") == input_list[:index].count(")"):
                if input_list[index + 1 :].count("(") == input_list[index + 1 :].count(")"):
                    if input_list[index] == ref_char:
                        raise RuntimeError
                    else:
                        input_list[index] = ref_char
    new_str = " ".join(input_list).split("$")
    result_set = set()
    for str_ in new_str:
        result_set.add(str_.strip())
    return result_set


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
        d1 = " ".join([form_manager.get_idx_symbol(x) for x in c1])
        d2 = " ".join([form_manager.get_idx_symbol(x) for x in c2])
        if get_split_comma(d1) == get_split_comma(d2):
            return True
        return False
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
