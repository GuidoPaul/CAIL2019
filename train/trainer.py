#!/usr/bin/python
# coding: utf-8

from timeit import default_timer as timer

import numpy as np
import torch

from apex import amp
from train.train_utils import evaluate, model_device
from utils.utils import AverageMeter, seed_everything, time_to_str


class Trainer(object):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        batch_size,
        num_epochs,
        device,
        n_gpus,
        criterion,
        fts_flag,
        gradient_accumulation_steps,
        model_checkpoint,
        logger,
        resume,
        verbose=1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.n_gpus = n_gpus
        self.criterion = criterion
        self.fts_flag = fts_flag
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model_checkpoint = model_checkpoint
        self.logger = logger
        self.resume = resume
        self.verbose = verbose
        self._reset()

    def _reset(self):
        self.start_epoch = 0
        self.global_step = 0

        self.model = self.model.to(self.device)

        if self.resume:
            resume_list = self.model_checkpoint.restore(
                self.model, self.optimizer
            )

            self.model = resume_list[0]
            self.optimizer = resume_list[1]
            self.start_epoch = resume_list[2]
            self.model_checkpoint.best = resume_list[3]
            self.logger.info(
                f"Checkpoint (epoch {self.start_epoch}, best {self.model_checkpoint.best}) loaded"
            )
            self.global_step = self.start_epoch * len(self.train_loader)

        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level="O1", verbosity=0
        )
        if self.n_gpus > 1:
            self.model, self.device = model_device(
                self.model, self.n_gpus, self.logger
            )

    def summary(self):
        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(
            "trainable parameters: {:4}M".format(params / 1000 / 1000)
        )
        # self.logger.info(self.model)

    def _save_info(self, epoch, val_loss):
        if self.n_gpus > 1:
            state = {
                "epoch": epoch,
                "arch": self.model_checkpoint.arch,
                "state_dict": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_loss": round(val_loss, 4),
            }
        else:
            state = {
                "epoch": epoch,
                "arch": self.model_checkpoint.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_loss": round(val_loss, 4),
            }
        return state

    def _valid_epoch(self):
        valid_loss = AverageMeter()
        valid_probs = []

        for step, batch in enumerate(self.valid_loader):
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

                # loss = self.criterion(anchor, positive, negative)
                loss = self.criterion(op.float(), anchor, positive, negative)
                valid_loss.update(loss.item(), batch_size)

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
            # probs = (op.to("cpu").numpy() * (pos_dist - neg_dist)).diagonal()
            valid_probs.append(probs)
        valid_probs = np.concatenate(valid_probs)

        valid_log = {"val_loss": valid_loss.avg, "val_probs": valid_probs}

        return valid_log

    def _train_epoch(self, start_time):
        train_loss = AverageMeter()

        for step, batch in enumerate(self.train_loader):
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            batch_size = batch[1].size(0)

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
            # anchor, positive, negative = self.model(**inputs)
            outputs = self.model(**inputs)

            if type(outputs) not in (tuple, list):  # tuple
                outputs = (outputs,)

            loss = self.criterion(op.float(), *outputs)
            train_loss.update(loss.item(), batch_size)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            if (step + 1) % 20 == 0:
                rate = self.optimizer.get_lr()
                now_epoch = (
                    self.global_step
                    * self.batch_size
                    / len(self.train_loader.dataset)
                )
                self.logger.info(
                    f"{rate[0]:.7f} "
                    f"{self.global_step / 1000:5.2f} "
                    f"{now_epoch:6.2f}  | "
                    f"{train_loss.avg:.4f}            | "
                    f'{time_to_str((timer() - start_time), "sec")}  '
                    f"{torch.cuda.memory_allocated() // 1024 ** 2}"
                )

        train_log = {"loss": train_loss.avg}
        return train_log

    def train(self):
        self.logger.info("     rate  step  epoch  |   loss  val_loss  |  time")
        self.logger.info("-" * 68)

        min_loss = np.Inf

        start_time = timer()
        for epoch in range(self.start_epoch, self.num_epochs):
            seed_everything(epoch * 1000 + epoch)

            train_log = self._train_epoch(start_time)
            valid_log = self._valid_epoch()
            logs = dict(train_log, **valid_log)

            rate = self.optimizer.get_lr()
            now_epoch = (
                self.global_step
                * self.batch_size
                / len(self.train_loader.dataset)
            )

            asterisk = " "
            if logs["val_loss"] < min_loss:
                min_loss = logs["val_loss"]
                asterisk = "*"

            self.logger.info(
                f"{rate[0]:.7f} "
                f"{self.global_step / 1000:5.2f} "
                f"{now_epoch:6.2f}  | "
                f'{logs["loss"]:.4f}    '
                f'{logs["val_loss"]:.4f} {asterisk}| '
                f'{time_to_str((timer() - start_time), "sec")}  '
                f"{torch.cuda.memory_allocated() // 1024 ** 2}"
            )

            valid_probs = logs["val_probs"]
            correct = evaluate(valid_probs)
            self.logger.info(
                f"min: {np.min(valid_probs):.4f} "
                f"max: {np.max(valid_probs):.4f} "
                f"avg: {np.average(valid_probs):.4f} "
                f"acc: {correct}, {float(correct / len(valid_probs)):.4f}"
            )

            if self.model_checkpoint:
                state = self._save_info(epoch, val_loss=logs["val_loss"])
                self.model_checkpoint.step(state=state)
