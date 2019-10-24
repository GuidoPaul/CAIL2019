#!/usr/bin/python
# coding: utf-8

import os

import numpy as np
import torch


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate(probs):
    correct = 0
    for i, prob in enumerate(probs):
        if i % 2 == 0 and prob <= 0:
            correct += 1
        if i % 2 == 1 and prob > 0:
            correct += 1
    return correct


def load_state(model, model_path):
    if os.path.exists(model_path):
        state = torch.load(model_path)
        epoch = state["epoch"]
        model.load_state_dict(state["model"])
        print(f"Restore model, epoch: {epoch}")
        return model, epoch
    else:
        print(f"Not found {model_path} model")
        return model, 1


def save_state(model, epoch, model_path):
    torch.save({"model": model.state_dict(), "epoch": epoch}, str(model_path))


def prepare_device(n_gpu_use, logger):
    if isinstance(n_gpu_use, int):
        n_gpu_use = range(n_gpu_use)
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        msg = "Warning: There's no GPU available on this machine, training will be performed on CPU."
        logger.warning(msg)
        n_gpu_use = range(0)
    if len(n_gpu_use) > n_gpu:
        msg = (
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, "
            f"but only {n_gpu} are available on this machine."
        )
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(
        "cuda:%d" % n_gpu_use[0] if len(n_gpu_use) > 0 else "cpu"
    )
    device_ids = n_gpu_use
    print(f"device_ids: {device_ids}, n_gpu: {n_gpu}")
    return device, device_ids


def model_device(model, n_gpu, logger):
    device, device_ids = prepare_device(n_gpu, logger)
    if len(device_ids) > 1:
        logger.info(f"current {len(device_ids)} GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])
    model = model.to(device)
    return model, device
