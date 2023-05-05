from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import os

from resnet import ResNet18, ResNet34, ResNet50, ResNet101


class ModelUtils():
    def __init__(self) -> None:
        pass

    def adjust_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
        if epoch < 80:
            lr = 0.1
        elif epoch < 120:
            lr = 0.01
        else:
            lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def get_model(model_name:str, dset:str, device:torch.device, device_ids:list, **kwargs) -> nn.Module:
        #TODO: add support for loading checkpointed models
        channels = 3 if dset in ["cifar10", "cifar100"] else 1
        model_name = model_name.lower()
        if model_name == "resnet18":
            model = ResNet18(channels, **kwargs)
        elif model_name == "resnet34":
            model = ResNet34(channels, **kwargs)
        elif model_name == "resnet50":
            model = ResNet50(channels, **kwargs)
        elif model_name == "resnet101":
            model = ResNet101(channels, **kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported")
        model = model.to(device)
        #model = DataParallel(model.to(device), device_ids=device_ids)
        return model

    def train(self, model:nn.Module, optim, dloader, loss_fn, device: torch.device, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            if "extra_batch" in kwargs:
                data = data.view(data.size(0) * data.size(1), *data.size()[2:])
                target = target.view(target.size(0) * target.size(1), *target.size()[2:])
            total_samples += data.size(0)
            optim.zero_grad()
            # check if epoch is passed as a keyword argument
            # if so, call adjust_learning_rate
            if "epoch" in kwargs:
                self.adjust_learning_rate(optim, kwargs["epoch"])

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
    
    def train_fedprox(self, model:nn.Module, global_model, mu, optim, dloader, loss_fn, device: torch.device, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            # check if epoch is passed as a keyword argument
            # if so, call adjust_learning_rate
            if "epoch" in kwargs:
                self.adjust_learning_rate(optim, kwargs["epoch"])
            position = kwargs.get("position", 0)
            output = model(data, position=position)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1) # type: ignore
            loss = loss_fn(output, target)
            
            # for fedprox
            fed_prox_reg = 0.0
            global_weight_collector = list(global_model.parameters())
            for param_index, param in enumerate(model.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg
            
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
    
    
    def train_fedcon(self, model:nn.Module, gloabl_model, previous_nets, cos, mu, temperature, optim, dloader, loss_fn, device: torch.device, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            # check if epoch is passed as a keyword argument
            # if so, call adjust_learning_rate
            if "epoch" in kwargs:
                self.adjust_learning_rate(optim, kwargs["epoch"])

            position = kwargs.get("position", 0)
            output, feat = model(data, position=position, out_feature=True)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1) # type: ignore
            loss = loss_fn(output, target)
            
            _, feat_g = gloabl_model(data, position=position, out_feature=True)
            posi = cos(feat, feat_g)
            logits = posi.reshape(-1,1)
            for previous_net in previous_nets:
                _, feat_p = previous_net(data, position=position, out_feature=True)
                nega = cos(feat, feat_p)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
            logits /= temperature
            labels = torch.zeros(data.size(0)).long()
            loss += mu * loss_fn(logits, labels.to(device))

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

    def load_model(self, model, path, device):
        if type(model) == DataParallel:
            model_ = model.module
        else:
            model_ = model
        wts = torch.load(path, map_location=torch.device('cpu'))
        for key in wts:
            wts[key] = wts[key].to(device)
        model_.load_state_dict(wts)

    def move_to_device(self, items: List[Tuple[torch.Tensor, torch.Tensor]],
                       device: torch.device) -> list:
        # Expects a list of tuples with each tupe containing two tensors
        return [[item[0].to(device), item[1].to(device)] for item in items]