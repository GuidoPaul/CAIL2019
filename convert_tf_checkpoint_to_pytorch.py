#!/usr/bin/python
# coding: utf-8

import shutil

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

BERT_MODEL_PATH = "models/chinese_L-12_H-768_A-12/"

if __name__ == "__main__":
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        BERT_MODEL_PATH + "bert_model.ckpt",
        BERT_MODEL_PATH + "bert_config.json",
        "models/pytorch_pretrain/pytorch_model.bin",
    )
    shutil.copyfile(
        BERT_MODEL_PATH + "bert_config.json",
        "models/pytorch_pretrain/bert_config.json",
    )
    shutil.copyfile(
        BERT_MODEL_PATH + "vocab.txt", "models/pytorch_pretrain/vocab.txt"
    )
