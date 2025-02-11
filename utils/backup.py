import torch
from datetime import datetime
import inspect

def save_checkpoint(fpath, epoch, model, optimizer, scheduler=None):
    checkpoint = dict(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
    )
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    checkpoint["epoch"] = epoch
    torch.save(checkpoint, fpath)


def load_checkpoint(fpath, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint["epoch"]
