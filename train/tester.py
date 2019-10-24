#!/usr/bin/python
# coding: utf-8

import numpy as np
import torch

from utils.utils import AverageMeter


class Tester(object):
    def __init__(
        self,
        model,
        test_loader,
        device,
        criterion,
        fts_flag=False,
        logger=None,
        verbose=1,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.fts_flag = fts_flag
        self.logger = logger
        self.verbose = verbose

    def eval(self):
        test_loss = AverageMeter()
        test_probs = []

        for step, batch in enumerate(self.test_loader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            batch_size = batch[1].size(0)

            with torch.no_grad():
                op = batch[0]
                inputs = {
                    "input_ids_a": batch[1],
                    "token_type_ids_a": batch[2],
                    "attention_mask_a": batch[3],
                    "input_ids_b": batch[4],
                    "token_type_ids_b": batch[5],
                    "attention_mask_b": batch[6],
                    "input_ids_c": batch[7],
                    "token_type_ids_c": batch[8],
                    "attention_mask_c": batch[9],
                }
                if self.fts_flag:
                    inputs.update(
                        {"x_a": batch[10], "x_b": batch[11], "x_c": batch[12]}
                    )
                anchor, positive, negative = self.model(**inputs)

                loss = self.criterion(op.float(), anchor, positive, negative)
                test_loss.update(loss.item(), batch_size)

            anchor = anchor.to("cpu").numpy()
            positive = positive.to("cpu").numpy()
            negative = negative.to("cpu").numpy()

            pos_dist = np.sqrt(
                np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
            )
            neg_dist = np.sqrt(
                np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
            )
            probs = pos_dist - neg_dist
            test_probs.append(probs)
        test_probs = np.concatenate(test_probs)

        correct = test_probs[np.where(test_probs <= 0)].shape[0]
        self.logger.info(
            f"min: {np.min(test_probs):.4f} "
            f"max: {np.max(test_probs):.4f} "
            f"avg: {np.average(test_probs):.4f} "
            f"loss: {test_loss.avg:.4f} "
            f"acc: {correct}, {float(correct / len(test_probs)):.4f}"
        )

    def test(self):
        test_probs = []

        for step, batch in enumerate(self.test_loader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                # op = batch[0]
                inputs = {
                    "input_ids_a": batch[1],
                    "token_type_ids_a": batch[2],
                    "attention_mask_a": batch[3],
                    "input_ids_b": batch[4],
                    "token_type_ids_b": batch[5],
                    "attention_mask_b": batch[6],
                    "input_ids_c": batch[7],
                    "token_type_ids_c": batch[8],
                    "attention_mask_c": batch[9],
                }
                if self.fts_flag:
                    inputs.update(
                        {"x_a": batch[10], "x_b": batch[11], "x_c": batch[12]}
                    )
                anchor, positive, negative = self.model(**inputs)

            anchor = anchor.to("cpu").numpy()
            positive = positive.to("cpu").numpy()
            negative = negative.to("cpu").numpy()

            pos_dist = np.sqrt(
                np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
            )
            neg_dist = np.sqrt(
                np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
            )
            probs = pos_dist - neg_dist
            test_probs.append(probs)
        test_probs = np.concatenate(test_probs)

        num_of_B = test_probs[np.where(test_probs <= 0)].shape[0]  # <=
        self.logger.info(
            f"min: {np.min(test_probs):.4f} "
            f"max: {np.max(test_probs):.4f} "
            f"avg: {np.average(test_probs):.4f} "
            f"num_of_B: {num_of_B}"
        )

        return test_probs
