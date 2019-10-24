#!/usr/bin/python
# coding: utf-8

"""
@Author: shenglin.bsl
@Date: 2019-09-04 17:47:37
@Last Modified by:   shenglin.bsl
@Last Modified time: 2019-09-04 17:47:37
"""
import os
import warnings

import torch
from pytorch_transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

from datasets.cail_dataset import CAILDataset
from utils.logginger import init_logger
from utils.utils import seed_everything
from viz_utils.model_utils import get_normalized_attention, get_tokenized_text
from viz_utils.visualization_utils import visualize_attention

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

ARCH = "bert"
SEED = 2323
FOLD_ID = 2

MAX_SEQ_LENGTH = 445
BATCH_SIZE = 10
NUM_EXAMPLES = 10

TEST_PATH = f"datasets/bigfolds/fold{FOLD_ID}_valid.txt"
BERT_MODEL_PATH = "output/ckpts"
BERT_VOCAB_PATH = "output/ckpts/vocab.txt"

LOG_DIR = "output/bertviz/"
OUTPUT_IMAGE_DIR = "output/bertviz/"

seed_everything(SEED)
logger = init_logger(log_name=ARCH, log_dir=LOG_DIR)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

model = BertModel.from_pretrained(BERT_MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model = model.to(device)

batch_data = next(iter(test_loader))
op = batch_data[0].tolist()
input_ids_a = batch_data[1].tolist()
token_type_ids_a = batch_data[2].tolist()
attention_mask_a = batch_data[3].tolist()
input_ids_b = batch_data[4].tolist()
token_type_ids_b = batch_data[5].tolist()
attention_mask_b = batch_data[6].tolist()
input_ids_c = batch_data[7].tolist()
token_type_ids_c = batch_data[8].tolist()
attention_mask_c = batch_data[9].tolist()

for i in range(NUM_EXAMPLES):
    tokenized_text_a = get_tokenized_text(tokenizer, input_ids_a[i])
    tokenized_text_b = get_tokenized_text(tokenizer, input_ids_b[i])
    tokenized_text_c = get_tokenized_text(tokenizer, input_ids_c[i])
    attention_weights_a = get_normalized_attention(
        model,
        input_ids_a[i],
        token_type_ids_a[i],
        attention_mask_a[i],
        method="last_layer_heads_average",
        normalization_method="min-max",
        device=device,
        logger=logger,
    )
    attention_weights_b = get_normalized_attention(
        model,
        input_ids_b[i],
        token_type_ids_b[i],
        attention_mask_b[i],
        method="last_layer_heads_average",
        normalization_method="min-max",
        device=device,
        logger=logger,
    )
    attention_weights_c = get_normalized_attention(
        model,
        input_ids_c[i],
        token_type_ids_c[i],
        attention_mask_c[i],
        method="last_layer_heads_average",
        normalization_method="min-max",
        device=device,
        logger=logger,
    )

    tokens_and_weights_a = []
    for index, token in enumerate(tokenized_text_a):
        tokens_and_weights_a.append((token, attention_weights_a[index].item()))

    tokens_and_weights_b = []
    for index, token in enumerate(tokenized_text_b):
        tokens_and_weights_b.append((token, attention_weights_b[index].item()))

    tokens_and_weights_c = []
    for index, token in enumerate(tokenized_text_c):
        tokens_and_weights_c.append((token, attention_weights_c[index].item()))

    output_path = os.path.join(OUTPUT_IMAGE_DIR, f"bert_viz_{i}.png")

    all_key_words = visualize_attention(
        tokens_and_weights_a,
        tokens_and_weights_b,
        tokens_and_weights_c,
        output_path,
    )

    print("-----")
    with open("all_key_words.txt", "w", encoding="utf-8") as outf:
        for word in all_key_words:
            print(word, file=outf)
