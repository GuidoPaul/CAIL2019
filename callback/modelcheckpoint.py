#!/usr/bin/python
# coding: utf-8

import glob
import os

import numpy as np
import torch

from utils.utils import ensure_dir


class ModelCheckpoint(object):
    def __init__(
        self,
        checkpoint_dir,
        monitor,
        logger,
        arch,
        save_best_only=True,
        best_model_name=None,
        epoch_model_name=None,
        mode="min",
        epoch_freq=1,
        best=None,
    ):
        self.monitor = monitor
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.epoch_freq = epoch_freq
        self.arch = arch
        self.logger = logger
        self.best_model_name = best_model_name
        self.epoch_model_name = epoch_model_name
        self.use = "on_epoch_end"
        self.default_model_name = "pytorch_model.bin"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf

        if best:
            self.best = best

        ensure_dir(self.checkpoint_dir)

    def step(self, state):
        if self.save_best_only:
            # save best model
            current = state[self.monitor]
            if self.monitor_op(current, self.best):
                self.logger.info(
                    f"Epoch {state['epoch']}: {self.monitor} improved from {self.best:.4f} to {current:.4f}"
                )
                self.best = current
                state["best"] = self.best
                best_path = os.path.join(
                    self.checkpoint_dir,
                    self.best_model_name.format(arch=self.arch),
                )
                torch.save(state, best_path)

                best_bin_path = os.path.join(
                    self.checkpoint_dir, self.default_model_name
                )
                torch.save(state["state_dict"], best_bin_path)
        else:
            # save some epoch model
            epoch_path = os.path.join(
                self.checkpoint_dir,
                self.epoch_model_name.format(
                    arch=self.arch,
                    epoch=state["epoch"],
                    val_loss=state[self.monitor],
                ),
            )
            if state["epoch"] % self.epoch_freq == 0:
                self.logger.info(f"Epoch {state['epoch']}: save model to disk.")
                torch.save(state, epoch_path)

                epoch_bin_path = os.path.join(
                    self.checkpoint_dir, self.default_model_name
                )
                torch.save(state["state_dict"], epoch_bin_path)

    def restore(self, model, optimizer=None):
        if self.save_best_only:
            best_path = os.path.join(
                self.checkpoint_dir, self.best_model_name.format(arch=self.arch)
            )

            checkpoint = torch.load(best_path)
            start_epoch = checkpoint["epoch"] + 1
            best = checkpoint["best"]

            if model:
                model.load_state_dict(checkpoint["state_dict"])
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])

            return [model, optimizer, start_epoch, best]
        else:
            epoch_path = sorted(
                glob.glob(self.checkpoint_dir + "*.pth"), key=os.path.getctime
            )[-1]

            checkpoint = torch.load(epoch_path)
            start_epoch = checkpoint["epoch"] + 1
            portion = os.path.splitext(epoch_path)[0]
            best = float(portion.split("_")[-1])

            if model:
                model.load_state_dict(checkpoint["state_dict"])
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])

            return [model, optimizer, start_epoch, best]
