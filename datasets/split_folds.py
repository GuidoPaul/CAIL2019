#!/usr/bin/python
# coding: utf-8

import os
import json
import numpy as np
from sklearn.model_selection import KFold


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_folds(data_path, fold_dir):
    ensure_dir(fold_dir)
    with open(data_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            x = json.loads(line)
            lines.append(x)
        lines = np.array(lines)

    kf = KFold(n_splits=10, random_state=2019, shuffle=True)
    for i, (idx_train, idx_valid) in enumerate(kf.split(lines)):
        l_train, l_valid = lines[idx_train], lines[idx_valid]
        print(len(l_train), len(l_valid))
        with open(f"{fold_dir}/fold{i}_train.txt", "w", encoding="utf-8") as f:
            for line in l_train:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")
        with open(f"{fold_dir}/fold{i}_valid.txt", "w", encoding="utf-8") as f:
            for line in l_valid:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    split_folds("SCM_5k.json", "bigfolds")
