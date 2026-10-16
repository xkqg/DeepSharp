# Prints scikit-learn's F1, ROC AUC, mean absolute percentage error and Brier score for the twenty rows the new-measure tests
# divide in time - t from 1 to 20: the first ten to train on, the next five to choose, the last five to test - each beside the
# same measure of predicting, for every row, the training rows' average answer: new-measures.txt, read by NewMeasureTests.
#
#   python new-measures.py > new-measures.txt      (scikit-learn 1.9.1, numpy 2.5.3)
#
# The F1 and the AUC of several labels are taken label by label and averaged (average='macro'), each label a column of noughts
# and ones; a part where a label is all one class has no AUC, as scikit-learn says. The Brier score of one answer is
# brier_score_loss of its chances, and of labels of which a row holds one, of the row's chances across them, which add up to one.

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, mean_absolute_percentage_error, roc_auc_score

T = np.arange(1, 21)
PARTS = {"train": slice(0, 10), "validation": slice(10, 15), "test": slice(15, 20)}


def likely(t):
    return [(t * 5 % 7) / 10.0, (t * 3 % 5) / 10.0, (t * 2 % 9) / 10.0]


def shares(t):
    row = np.array(likely(t)) + 0.1
    return list(row / row.sum())


def show(name, values):
    print(name + " = " + repr([float(value) for value in values]))


def each_part(name, measure, actual, predicted):
    average = actual[PARTS["train"]].mean(axis=0)
    show(name + " model", [measure(actual[part], predicted[part]) for part in PARTS.values()])
    show(name + " average", [measure(actual[part], np.tile(average, (len(actual[part]), 1))) for part in PARTS.values()])


# One label: t % 2, the chance said for it the first of likely(t).
one = np.array([[t % 2] for t in T], dtype=float)
said = np.array([[likely(t)[0]] for t in T])
each_part("one f1", lambda a, p: f1_score(a[:, 0], p[:, 0] >= 0.5), one, said)
each_part("one auc", lambda a, p: roc_auc_score(a[:, 0], p[:, 0]), one, said)
each_part("one brier", lambda a, p: brier_score_loss(a[:, 0], p[:, 0]), one, said)

# Labels of which a row holds one: a, b or c by t * 7 % 3, the chances likely(t) shifted and made to add up to one.
held = np.array([[1.0 if t * 7 % 3 == k else 0.0 for k in range(3)] for t in T])
chances = np.array([shares(t) for t in T])


def likeliest(p):
    named = np.zeros_like(p)
    named[np.arange(len(p)), p.argmax(axis=1)] = 1
    return named


each_part("one-of f1", lambda a, p: f1_score(a, likeliest(p), average="macro", zero_division=0), held, chances)
each_part("one-of auc", lambda a, p: roc_auc_score(a, p, average="macro"), held, chances)
each_part("one-of brier", lambda a, p: brier_score_loss(a.argmax(axis=1), p, labels=[0, 1, 2]), held, chances)

# Labels of which a row holds any number: t % 2, t % 3 == 0 and t % 5 == 0, the chances likely(t).
many = np.array([[t % 2, 1.0 if t % 3 == 0 else 0.0, 1.0 if t % 5 == 0 else 0.0] for t in T])
said_many = np.array([likely(t) for t in T])
each_part("any f1", lambda a, p: f1_score(a, p >= 0.5, average="macro", zero_division=0), many, said_many)

# An amount: 2 + t * 3 % 7, predicted as 2.5 + t % 4.
amount = np.array([[2 + t * 3 % 7] for t in T], dtype=float)
guess = np.array([[2.5 + t % 4] for t in T], dtype=float)
each_part("amount mape", mean_absolute_percentage_error, amount, guess)
