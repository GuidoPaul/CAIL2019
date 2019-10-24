#!/usr/bin/python
# coding: utf-8

import json
import re

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset

from processing import do_feature_engineering

STOPWORDS_PATH = "datasets/stopwords.txt"


class InputExample(object):
    """A single set of training/test triplet data example."""

    def __init__(self, guid, text_a, text_b, text_c):
        """init InputExample class.

        Parameters
        ----------
        guid : str
            Unique id for the example.
        text_a : str
            The untokenized text of the first sequence.
        text_b : str
            The untokenized text of the second sequence.
        text_c : str
            The untokenized text of the third sequence.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c

    def __str__(self):
        return str(
            f"text_a: {self.text_a}\n"
            f"text_b: {self.text_b}\n"
            f"text_c: {self.text_c}"
        )


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, segment_ids, input_mask):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask

    def __str__(self):
        return str(
            f"input_ids: {self.input_ids}\n"
            f"segment_ids: {self.segment_ids}\n"
            f"input_mask: {self.input_mask}"
        )


def remove_punctuation(line):
    # rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    line = rule.sub("", line)
    return line


class CAILDataset(Dataset):
    def __init__(
        self,
        data_path,
        max_seq_len,
        vocab_path,
        # tfidf_a_df,
        # tfidf_b_df,
        # tfidf_c_df,
        fts_flag=False,
        mode="test",
    ):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.vocab_path = vocab_path
        # self.exft_a_df = tfidf_a_df
        # self.exft_b_df = tfidf_b_df
        # self.exft_c_df = tfidf_c_df
        self.fts_flag = fts_flag
        self.mode = mode
        self.reset()

    def reset(self):
        self.tokenizer = BertTokenizer(vocab_file=self.vocab_path)
        self.build_examples()

    def read_data(self):
        print(self.data_path)
        xlist = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                # xlist.append((x["A"], x["B"], x["C"]))
                if self.mode == "train" or self.mode == "valid":
                    if i % 2 == 0:
                        xlist.append((x["A"], x["B"], x["C"]))
                    else:
                        xlist.append((x["A"], x["C"], x["B"]))
                else:
                    xlist.append((x["A"], x["B"], x["C"]))
        return xlist

    def build_examples(self):
        xlist = self.read_data()
        self.examples = []
        list_text_a = []
        list_text_b = []
        list_text_c = []
        for idx, x in enumerate(xlist):
            guid = "%s-%d" % (self.mode, idx)
            text_a = x[0]
            text_b = x[1]
            text_c = x[2]
            example = InputExample(
                guid=guid, text_a=text_a, text_b=text_b, text_c=text_c
            )
            self.examples.append(example)
            list_text_a.append(text_a)
            list_text_b.append(text_b)
            list_text_c.append(text_c)
        if self.fts_flag:
            self.exft_a_df = self.build_ex_features(list_text_a)
            self.exft_b_df = self.build_ex_features(list_text_b)
            self.exft_c_df = self.build_ex_features(list_text_c)
            self.exft_a_df.fillna(0, inplace=True)
            self.exft_b_df.fillna(0, inplace=True)
            self.exft_c_df.fillna(0, inplace=True)

    def build_features(self, example):
        max_seq_len = self.max_seq_len - 2

        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = self.tokenizer.tokenize(example.text_b)
        tokens_c = self.tokenizer.tokenize(example.text_c)

        if len(tokens_a) > max_seq_len:
            tokens_a = tokens_a[-max_seq_len:]
        if len(tokens_b) > max_seq_len:
            tokens_b = tokens_b[-max_seq_len:]
        if len(tokens_c) > max_seq_len:
            tokens_c = tokens_c[-max_seq_len:]

        input_ids_a = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokens_a + ["[SEP]"]
        )
        input_ids_b = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokens_b + ["[SEP]"]
        )
        input_ids_c = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokens_c + ["[SEP]"]
        )
        input_mask_a = [1] * len(input_ids_a)
        input_mask_b = [1] * len(input_ids_b)
        input_mask_c = [1] * len(input_ids_c)
        segment_ids_a = [0] * len(input_ids_a)
        segment_ids_b = [0] * len(input_ids_b)
        segment_ids_c = [0] * len(input_ids_c)

        padding_a = [0] * (max_seq_len - len(tokens_a))
        padding_b = [0] * (max_seq_len - len(tokens_b))
        padding_c = [0] * (max_seq_len - len(tokens_c))

        input_ids_a += padding_a
        segment_ids_a += padding_a
        input_mask_a += padding_a
        input_ids_b += padding_b
        segment_ids_b += padding_b
        input_mask_b += padding_b
        input_ids_c += padding_c
        segment_ids_c += padding_c
        input_mask_c += padding_c

        feature_a = InputFeature(
            input_ids=input_ids_a,
            segment_ids=segment_ids_a,
            input_mask=input_mask_a,
        )
        feature_b = InputFeature(
            input_ids=input_ids_b,
            segment_ids=segment_ids_b,
            input_mask=input_mask_b,
        )
        feature_c = InputFeature(
            input_ids=input_ids_c,
            segment_ids=segment_ids_c,
            input_mask=input_mask_c,
        )
        return feature_a, feature_b, feature_c

    def build_ex_features(self, list_text):
        return do_feature_engineering(list_text)

    def _preprocess_op(self, index):
        example = self.examples[index]
        if self.mode == "train" or self.mode == "valid":
            if index % 2 == 0:
                op = 1
            else:
                op = -1
        else:
            op = 1
        feature_a, feature_b, feature_c = self.build_features(example)
        return (
            op,
            np.array(feature_a.input_ids, dtype=np.int64),
            np.array(feature_a.segment_ids, dtype=np.int64),
            np.array(feature_a.input_mask, dtype=np.int64),
            np.array(feature_b.input_ids, dtype=np.int64),
            np.array(feature_b.segment_ids, dtype=np.int64),
            np.array(feature_b.input_mask, dtype=np.int64),
            np.array(feature_c.input_ids, dtype=np.int64),
            np.array(feature_c.segment_ids, dtype=np.int64),
            np.array(feature_c.input_mask, dtype=np.int64),
        )

    def _exft_preprocess_op(self, index):
        example = self.examples[index]
        if self.mode == "train" or self.mode == "valid":
            if index % 2 == 0:
                op = 1
            else:
                op = -1
        else:
            op = 1
        feature_a, feature_b, feature_c = self.build_features(example)
        return (
            op,
            np.array(feature_a.input_ids, dtype=np.int64),
            np.array(feature_a.segment_ids, dtype=np.int64),
            np.array(feature_a.input_mask, dtype=np.int64),
            np.array(feature_b.input_ids, dtype=np.int64),
            np.array(feature_b.segment_ids, dtype=np.int64),
            np.array(feature_b.input_mask, dtype=np.int64),
            np.array(feature_c.input_ids, dtype=np.int64),
            np.array(feature_c.segment_ids, dtype=np.int64),
            np.array(feature_c.input_mask, dtype=np.int64),
            torch.tensor(self.exft_a_df.iloc[index], dtype=torch.float32),
            torch.tensor(self.exft_b_df.iloc[index], dtype=torch.float32),
            torch.tensor(self.exft_c_df.iloc[index], dtype=torch.float32),
        )

    def __getitem__(self, index):
        if self.fts_flag:
            return self._exft_preprocess_op(index)
        else:
            return self._preprocess_op(index)

    def __len__(self):
        return len(self.examples)
