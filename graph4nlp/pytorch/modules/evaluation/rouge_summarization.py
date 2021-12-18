import os
import re
import shutil

from graph4nlp.pytorch.modules.evaluation.base import EvaluationMetricBase

import pyrouge


class SummarizationRouge(EvaluationMetricBase):
    def __init__(self):
        super(SummarizationRouge, self).__init__()
        self.rouge = pyrouge.Rouge155()

    def calculate_scores(self, ground_truth, predict):
        """
            The standard rouge calculation function for summerization. It will compute \
                the rouge scores using pyrouge tools.
            Note that each instance in ``ground_truth`` or ``predict`` is a list of sentences.
        Parameters
        ----------
        ground_truth: list[list[string]]
            The ground truth (correct) target values. It is a list of lists.
        predict: list[list[string]]
            The predicted target values. It is a list of lists.
        Returns
        -------
        score: float
            The final bleu score
        """
        if os.path.exists("pred"):
            shutil.rmtree("pred")
        if os.path.exists("gt"):
            shutil.rmtree("gt")
        os.makedirs("pred", exist_ok=False)
        os.makedirs("gt", exist_ok=False)
        cnt = 0
        for g, p in zip(ground_truth, predict):
            self.dump_tmp(g, os.path.join("gt", str(cnt) + ".txt"))
            self.dump_tmp(p, os.path.join("pred", str(cnt) + ".txt"))
            cnt += 1
        self.rouge.model_filename_pattern = "#ID#.txt"
        self.rouge.system_filename_pattern = r"(\d+).txt"
        self.rouge.model_dir = "pred"
        self.rouge.system_dir = "gt"
        rouge_results = self.rouge.convert_and_evaluate()
        dic = self.rouge.output_to_dict(rouge_results)
        return dic

    def make_html_safe(self, s):
        """Replace any angled brackets in string s to avoid interfering with HTML \
            attention visualizer."""
        s = s.replace("<", "&lt;")
        s = s.replace(">", "&gt;")
        return s

    def dump_tmp(self, content, path):
        output = "\n".join(content)
        output = self.make_html_safe(output)

        with open(path, "w") as f:
            f.write(output)


def read_file(path):
    with open(path, "r") as f:
        content = f.readlines()
    ret = []
    for line in content:
        if line.strip() == "":
            continue
        if line.split()[-1] != "</t>":
            line += " . </t>"
        sents = re.findall("<t>(.*?)</t>", line, flags=re.S | re.M)
        sen = [sent.strip() for sent in sents]
        ret.append(sen)
    return ret


if __name__ == "__main__":
    gt_file = "/home/shiina/shiina/lib/graph4nlp/cnn_tgt_output.txt"
    pred_file = "/home/shiina/shiina/lib/graph4nlp/cnn_pred_output.txt"
    gt_collect = read_file(gt_file)
    pred_collect = read_file(pred_file)
    assert len(gt_collect) == len(pred_collect)

    metric = SummarizationRouge().calculate_scores(ground_truth=gt_collect, predict=pred_collect)
    print(metric)
