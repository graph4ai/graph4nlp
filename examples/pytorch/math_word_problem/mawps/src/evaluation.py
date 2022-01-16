from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase

import sympy
from sympy.parsing.sympy_parser import parse_expr


class SolutionMatch(EvaluationMetricBase):
    def __init__(self):
        super(SolutionMatch, self).__init__()

    def calculate_scores(self, ground_truth, predict):
        correct = 0
        assert len(ground_truth) == len(predict)
        for gt, pred in zip(ground_truth, predict):
            gt = gt.strip()
            pred = pred.strip()

            result = True
            c1 = gt
            c2 = pred
            if ("=" not in c1) or ("=" not in c2):
                result = False
            else:
                try:
                    s1 = c1.split("=")
                    s2 = c2.split("=")
                    eq1 = []
                    eq2 = []
                    x = sympy.Symbol("x")
                    eq1.append(parse_expr(s1[0]))
                    eq1.append(parse_expr(s1[1]))
                    eq2.append(parse_expr(s2[0]))
                    eq2.append(parse_expr(s2[1]))
                    res1 = sympy.solve(sympy.Eq(eq1[0], eq1[1]), x)
                    res2 = sympy.solve(sympy.Eq(eq2[0], eq2[1]), x)

                    if not res1 or not res2:
                        result = False
                    result = res1[0] == res2[0]
                except BaseException:
                    result = False
            if result:
                correct += 1.0
        return correct / len(ground_truth)


def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def is_all_same(c1, c2, form_manager):
    all_same = False
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
    if all_same is False:
        if is_solution_same(c1, c2, form_manager):
            return True
        return False
    else:
        return True


def is_solution_same(i1, i2, form_manager):
    c1 = " ".join([form_manager.get_idx_symbol(x) for x in i1])
    c2 = " ".join([form_manager.get_idx_symbol(x) for x in i2])
    if ("=" not in c1) or ("=" not in c2):
        return False
    elif (form_manager.unk_token in c1) or (form_manager.unk_token in c2):
        return False
    else:
        try:
            s1 = c1.split("=")
            s2 = c2.split("=")
            eq1 = []
            eq2 = []
            x = sympy.Symbol("x")
            eq1.append(parse_expr(s1[0]))
            eq1.append(parse_expr(s1[1]))
            eq2.append(parse_expr(s2[0]))
            eq2.append(parse_expr(s2[1]))
            res1 = sympy.solve(sympy.Eq(eq1[0], eq1[1]), x)
            res2 = sympy.solve(sympy.Eq(eq2[0], eq2[1]), x)

            if not res1 or not res2:
                return False
            if res1[0] == res2[0]:
                # print("Excution_true: ", c1, '\t', c2)
                pass
            return res1[0] == res2[0]

        except BaseException:
            # print("Excution_error: ", c1, '\t', c2)
            pass
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
