from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

from resnet import ResNet18, ResNet34, ResNet50


class ModelUtils():
    def __init__(self) -> None:
        pass

    def get_model(self, model_name:str, dset:str, device:torch.device, device_ids:list) -> DataParallel:
        #TODO: add support for loading checkpointed models
        channels = 3 if dset=="cifar10" else 1
        model_name = model_name.lower()
        if model_name == "resnet18":
            model = ResNet18(channels)
        elif model_name == "resnet34":
            model = ResNet34(channels)
        elif model_name == "resnet50":
            model = ResNet50(channels)
        else:
            raise ValueError(f"Model name {model_name} not supported")
        model = DataParallel(model.to(device), device_ids=device_ids)
        return model

    def train(self, model:nn.Module, optim, dloader, loss_fn, device: torch.device, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            position = kwargs.get("position", 0)
            output = model(data, position=position)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1) # type: ignore
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are the same
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(dloader.dataset)
        return train_loss, acc

    def test(self, model, dloader, loss_fn, device, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dloader:
                data, target = data.to(device), target.to(device)
                position = kwargs.get("position", 0)
                output = model(data, position=position)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                # view_as() is used to make sure the shape of pred and target are the same
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(dloader.dataset)
        return test_loss, acc

    def save_model(self, model, path):
        if type(model) == DataParallel:
            model_ = model.module
        else:
            model_ = model
        torch.save(model_.state_dict(), path)

    def move_to_device(self, items: List[Tuple[torch.Tensor, torch.Tensor]],
                       device: torch.device) -> list:
        # Expects a list of tuples with each tupe containing two tensors
        return [[item[0].to(device), item[1].to(device)] for item in items]