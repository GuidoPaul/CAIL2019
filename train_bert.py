#!/usr/bin/python
# coding: utf-8

import os
import warnings
from timeit import default_timer as timer

import torch
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import DataLoader

from callback.modelcheckpoint import ModelCheckpoint
from datasets.cail_dataset import CAILDataset
from loss.loss import TripletLoss_op
from models.net2 import BertForTripletNet as BertForTripletNet
from train.tester import Tester
from train.trainer import Trainer
from utils.logger import init_logger
from utils.utils import seed_everything, time_to_str


warnings.filterwarnings("ignore")


ARCH = "bert"
SEED = 2323
FOLD_ID = 2
MULTI_GPUS = [0, 1, 2, 3]
RESUME = False


LOG_DIR = "output/logs"
CHECKPOINT_DIR = "output/ckpts/"
TRAIN_PATH = f"datasets/bigfolds/fold{FOLD_ID}_train.txt"
VALID_PATH = f"datasets/bigfolds/fold{FOLD_ID}_valid.txt"
BERT_MODEL_PATH = "models/ms"
BERT_VOCAB_PATH = "models/ms/vocab.txt"
BEST_MODEL_NAME = "{arch}_best.pth"
EPOCH_MODEL_NAME = "{arch}_{epoch}_{val_loss}.pth"

MAX_SEQ_LENGTH = 445
BATCH_SIZE = 12
NUM_EPOCHS = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_PROPORTION = 0.05


seed_everything(SEED)
logger = init_logger(log_name=ARCH, log_dir=LOG_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, MULTI_GPUS))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_dataset = CAILDataset(
    data_path=TRAIN_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
    mode="train",
)
valid_dataset = CAILDataset(
    data_path=VALID_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
    mode="valid",
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

logger.info("---------- Bert Train ... ----------")
start_time = timer()

seed_everything(SEED)

model = BertForTripletNet.from_pretrained(BERT_MODEL_PATH)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
num_train_optimization_steps = (
    int(len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
)
optimizer = BertAdam(
    optimizer_grouped_parameters,
    lr=LEARNING_RATE,
    warmup=WARMUP_PROPORTION,
    t_total=num_train_optimization_steps,
)

model_checkpoint = ModelCheckpoint(
    checkpoint_dir=CHECKPOINT_DIR,
    mode="min",
    monitor="val_loss",
    save_best_only=False,
    best_model_name=BEST_MODEL_NAME,
    epoch_model_name=EPOCH_MODEL_NAME,
    arch=ARCH,
    logger=logger,
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    device=device,
    n_gpus=len(MULTI_GPUS),
    criterion=TripletLoss_op(),
    fts_flag=False,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    model_checkpoint=model_checkpoint,
    logger=logger,
    resume=RESUME,
)

trainer.summary()
trainer.train()

logger.info(f'Took {time_to_str((timer() - start_time), "sec")}')

print("---------- Bert Eval ... ----------")
start_time = timer()

BERT_MODEL_PATH = "output/ckpts"
BERT_VOCAB_PATH = "output/ckpts/vocab.txt"

test_dataset = CAILDataset(
    data_path=VALID_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
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

tester.eval()

print(f'Took {time_to_str((timer() - start_time), "sec")}')
