#!/usr/bin/python
# coding: utf-8

import os
import warnings
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from datasets.cail_dataset import CAILDataset
from loss.loss import TripletLoss_op
from models.net import (
    BertEmbedding2ForTripletNet,
    BertEmbeddingForTripletNet,
    BertForTripletNet,
    BertFtsForTripletNet,
)
from train.tester import Tester
from utils.logger import init_logger
from utils.utils import seed_everything, time_to_str, write2file


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ARCH = "bert"
SEED = 2323
FOLD_ID = 2

TEST_PATH = "/input/input.txt"
OUTPUT_PATH = "/output/output.txt"
# TEST_PATH = "datasets/input.txt"  #
# TEST_PATH = "datasets/SCM_5k.json"
TEST_PATH = f"datasets/bigfolds/fold{FOLD_ID}_valid.txt"  #
OUTPUT_PATH = "output/output.txt"  #
LOG_DIR = "output/logs"

MAX_SEQ_LENGTH = 445
BATCH_SIZE = 16

seed_everything(SEED)
logger = init_logger(log_name=ARCH, log_dir=LOG_DIR)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("---------- Bert Eval ... ----------")
start_time = timer()

# bert_config.json, pytorch_model.bin vocab.txt in chpts
BERT_MODEL_PATH = "output/ckpts6920"
BERT_VOCAB_PATH = "output/ckpts6920/vocab.txt"

test_dataset = CAILDataset(
    data_path=TEST_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
    seed=SEED,
    mode="test",
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

model = BertForTripletNet.from_pretrained(BERT_MODEL_PATH)
model = model.to(device)

tester = Tester(
    model=model,
    test_loader=test_loader,
    device=device,
    criterion=TripletLoss_op(),
    logger=logger,
)

# tester.eval()
test_probs_6920 = tester.test()

# -------------

BERT_MODEL_PATH = "output/ckpts6833"
BERT_VOCAB_PATH = "output/ckpts6833/vocab.txt"

model = BertForTripletNet.from_pretrained(BERT_MODEL_PATH)
model = model.to(device)

tester = Tester(
    model=model,
    test_loader=test_loader,
    device=device,
    criterion=TripletLoss_op(),
    logger=logger,
)

test_probs_6833 = tester.test()

# -------------

BERT_MODEL_PATH = "output/ckpts443"
BERT_VOCAB_PATH = "output/ckpts443/vocab.txt"

model = BertEmbeddingForTripletNet.from_pretrained(BERT_MODEL_PATH)
model = model.to(device)

tester = Tester(
    model=model,
    test_loader=test_loader,
    device=device,
    criterion=TripletLoss_op(),
    logger=logger,
)

test_probs_443 = tester.test()

# -------------

BERT_MODEL_PATH = "output/ckpts439"
BERT_VOCAB_PATH = "output/ckpts439/vocab.txt"

model = BertEmbedding2ForTripletNet.from_pretrained(BERT_MODEL_PATH)
model = model.to(device)

tester = Tester(
    model=model,
    test_loader=test_loader,
    device=device,
    criterion=TripletLoss_op(),
    logger=logger,
)

test_probs_439 = tester.test()

# -------------

BERT_MODEL_PATH = "output/ckptskz"
BERT_VOCAB_PATH = "output/ckptskz/vocab.txt"

test_dataset = CAILDataset(
    data_path=TEST_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
    fts_flag=True,
    seed=SEED,
    mode="test",
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

model = BertFtsForTripletNet.from_pretrained(BERT_MODEL_PATH)
model = model.to(device)

tester = Tester(
    model=model,
    test_loader=test_loader,
    device=device,
    criterion=TripletLoss_op(),
    fts_flag=True,
    logger=logger,
)

# tester.eval()
test_probs_kz = tester.test()

# -------------

test_probs = []
correct = 0
for i in range(len(test_probs_kz)):
    # probs = test_probs_6833[i] + test_probs_6920[i] + test_probs_kz[i]
    probs = (
        test_probs_6833[i]
        + test_probs_6920[i]
        + test_probs_kz[i]
        + test_probs_443[i]
        + test_probs_439[i]
    )
    test_probs.append(probs)
    if probs <= 0:
        correct += 1

print(correct)
print(correct / len(test_probs))

write2file(OUTPUT_PATH, test_probs)

print(f'Took {time_to_str((timer() - start_time), "sec")}')
