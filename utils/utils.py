#!/usr/bin/python
# coding: utf-8

import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=2323):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return "%2d hr %02d min" % (hr, min)
    elif mode == "sec":
        t = int(t)
        min = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (min, sec)
    else:
        raise NotImplementedError


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_roc_auc_score(data, y_true, y_pred):
    score = roc_auc_score(data[y_true], data[y_pred])
    print("Roc auc score of valid offline_data is {:.4f}".format(score))
    return score


def write2file_bak(output_path, probs):
    with open(output_path, "w", encoding="utf-8") as outf:
        for i in range(0, len(probs), 2):
            if probs[i] > probs[i + 1]:
                print("B", file=outf)
            else:
                print("C", file=outf)


def write2file(output_path, probs):
    with open(output_path, "w", encoding="utf-8") as outf:
        for i in range(0, len(probs)):
            if probs[i] <= 0:
                print("B", file=outf)
            else:
                print("C", file=outf)
