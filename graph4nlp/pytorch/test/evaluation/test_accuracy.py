import random
import unittest
import numpy as np
import torch
import tqdm
from sklearn import metrics

from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy


"""
    strict unittest by comparing the results with sklearn
"""


class MyTestCase(unittest.TestCase):
    def test_special_case(self):
        ground_truth = np.array([0, 1, 3, 1, 1])
        predict = np.array([0, 1, 3, 3, 1])
        mcm_sklearn = metrics.multilabel_confusion_matrix(ground_truth, predict)
        accuracy = Accuracy(metrics=["precision", "recall", "F1", "accuracy"])
        mcm_graph4ai = accuracy._calculate_confusion_matrix(
            ground_truth=ground_truth, predict=predict
        )

        self.assertEqual(True, (mcm_graph4ai == mcm_sklearn).all())
        print("mcm test passed")

        precision_sklearn, recall_sklearn, f1_sklearn, _ = metrics.precision_recall_fscore_support(
            y_true=ground_truth, y_pred=predict, average="micro"
        )

        accuracy_sklearn = metrics.accuracy_score(y_true=ground_truth, y_pred=predict)

        (
            precision_graph4ai,
            recall_graph4ai,
            f1_graph4ai,
            accuracy_graph4ai,
        ) = accuracy.calculate_scores(
            ground_truth=torch.from_numpy(ground_truth),
            predict=torch.from_numpy(predict),
            average="micro",
        )
        self.assertEqual(precision_graph4ai, precision_sklearn)
        self.assertEqual(recall_graph4ai, recall_sklearn)
        self.assertEqual(f1_graph4ai, f1_sklearn)
        self.assertEqual(accuracy_graph4ai, accuracy_sklearn)

        print("micro average special case test passed")

        accuracy_sklearn = metrics.accuracy_score(y_true=ground_truth, y_pred=predict)

        precision_sklearn, recall_sklearn, f1_sklearn, _ = metrics.precision_recall_fscore_support(
            y_true=ground_truth, y_pred=predict, average="macro"
        )

        (
            precision_graph4ai,
            recall_graph4ai,
            f1_graph4ai,
            accuracy_graph4ai,
        ) = accuracy.calculate_scores(
            ground_truth=torch.from_numpy(ground_truth),
            predict=torch.from_numpy(predict),
            average="macro",
        )
        self.assertEqual(precision_graph4ai, precision_sklearn)
        self.assertEqual(recall_graph4ai, recall_sklearn)
        self.assertEqual(f1_graph4ai, f1_sklearn)
        self.assertEqual(accuracy_graph4ai, accuracy_sklearn)
        print("macro average special case test passed")
        accuracy_sklearn = metrics.accuracy_score(y_true=ground_truth, y_pred=predict)
        precision_sklearn, recall_sklearn, f1_sklearn, _ = metrics.precision_recall_fscore_support(
            y_true=ground_truth, y_pred=predict, average="weighted"
        )

        (
            precision_graph4ai,
            recall_graph4ai,
            f1_graph4ai,
            accuracy_graph4ai,
        ) = accuracy.calculate_scores(
            ground_truth=torch.from_numpy(ground_truth),
            predict=torch.from_numpy(predict),
            average="weighted",
        )
        self.assertEqual(precision_graph4ai, precision_sklearn)
        self.assertEqual(recall_graph4ai, recall_sklearn)
        self.assertEqual(f1_graph4ai, f1_sklearn)
        self.assertEqual(accuracy_graph4ai, accuracy_sklearn)
        print("weighted average special case test passed")

    def test_random(self):
        accuracy = Accuracy(metrics=["accuracy", "precision", "recall", "F1"])
        for _ in tqdm.tqdm(range(100)):
            length = random.randint(2, 1000000)
            n_labels = random.randint(2, 100000)
            ground_truth = []
            predict = []
            for _ in range(length):
                ground_truth.append(random.randint(0, n_labels - 1))
                predict.append(random.randint(0, n_labels - 1))
            ground_truth = np.array(ground_truth)
            predict = np.array(predict)
            average = random.choice(["weighted", "macro", "micro"])
            accuracy_sklearn = metrics.accuracy_score(y_true=ground_truth, y_pred=predict)
            (
                precision_sklearn,
                recall_sklearn,
                f1_sklearn,
                _,
            ) = metrics.precision_recall_fscore_support(
                y_true=ground_truth, y_pred=predict, average=average
            )

            (
                accuracy_graph4ai,
                precision_graph4ai,
                recall_graph4ai,
                f1_graph4ai,
            ) = accuracy.calculate_scores(
                ground_truth=torch.from_numpy(ground_truth),
                predict=torch.from_numpy(predict),
                average=average,
            )
            self.assertEqual(precision_graph4ai, precision_sklearn)
            self.assertEqual(recall_graph4ai, recall_sklearn)
            self.assertEqual(f1_graph4ai, f1_sklearn)
            self.assertEqual(accuracy_graph4ai, accuracy_sklearn)
        print("random test 1 passed")

        for _ in tqdm.tqdm(range(100)):
            length = random.randint(2, 1000000)
            n_labels = random.randint(2, 100000)
            replace_ratio = random.random() / 1.5
            ground_truth = []
            predict = []
            for _ in range(length):
                ground_truth.append(random.randint(0, n_labels - 1))
                if random.random() < replace_ratio:
                    predict.append(random.randint(0, n_labels - 1))
                else:
                    predict.append(ground_truth[-1])
            ground_truth = np.array(ground_truth)
            predict = np.array(predict)
            average = random.choice(["weighted", "macro", "micro"])
            accuracy_sklearn = metrics.accuracy_score(y_true=ground_truth, y_pred=predict)
            (
                precision_sklearn,
                recall_sklearn,
                f1_sklearn,
                _,
            ) = metrics.precision_recall_fscore_support(
                y_true=ground_truth, y_pred=predict, average=average
            )

            (
                accuracy_graph4ai,
                precision_graph4ai,
                recall_graph4ai,
                f1_graph4ai,
            ) = accuracy.calculate_scores(
                ground_truth=torch.from_numpy(ground_truth),
                predict=torch.from_numpy(predict),
                average=average,
            )
            self.assertEqual(precision_graph4ai, precision_sklearn)
            self.assertEqual(recall_graph4ai, recall_sklearn)
            self.assertEqual(f1_graph4ai, f1_sklearn)
            self.assertEqual(accuracy_graph4ai, accuracy_sklearn)
        print("random test 2 passed")


if __name__ == "__main__":
    unittest.main()
