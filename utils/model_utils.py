from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import models

MODEL_DICT = {
    # https://github.com/polo5/ZeroShotKnowledgeTransfer
    'wrn16_1': models.wresnet.wrn_16_1,
    'wrn16_2': models.wresnet.wrn_16_2,
    'wrn40_1': models.wresnet.wrn_40_1,
    'wrn40_2': models.wresnet.wrn_40_2,

    # https://github.com/HobbitLong/RepDistiller
    'resnet8': models.resnet_tiny.resnet8,
    'resnet20': models.resnet_tiny.resnet20,
    'resnet32': models.resnet_tiny.resnet32,
    'resnet56': models.resnet_tiny.resnet56,
    'resnet110': models.resnet_tiny.resnet110,
    'resnet8x4': models.resnet_tiny.resnet8x4,
    'resnet32x4': models.resnet_tiny.resnet32x4,
    'vgg8': models.vgg.vgg8_bn,
    'vgg11': models.vgg.vgg11_bn,
    'vgg13': models.vgg.vgg13_bn,
    'shufflenetv2': models.shufflenetv2.shuffle_v2,
    'mobilenetv2': models.mobilenetv2.mobilenet_v2,
    
    # https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL
    'resnet50':  models.resnet.resnet50,
    'resnet18':  models.resnet.resnet18,
    'resnet34':  models.resnet.resnet34,
}

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
        channels = 3 if dset in ["cifar10", "cifar100" ,"svhn"] else 1
        model_name = model_name.lower()
        num_cls = kwargs.get("num_classes", 10)
        if channels==1:
            model = models.resnet.resnet18(channels, **kwargs)
        elif model_name in MODEL_DICT:
            model = MODEL_DICT[model_name](**kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported")
        print(f"Model {model_name} loading on device {device}")
        model = model.to(device)
        print(f"Model {model_name} loaded on device {device}")
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
            if position==0:
                output = model(data)
            else:
                output = model(data, position=position)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1) # type: ignore
            if len(target.size()) > 1 and target.size(1) == 1:
                target = target.squeeze(dim=1)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are the same
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / total_samples
        return train_loss, acc

    def soft_loss(self, preds, y):
        'Function for the soft cross entrohpy objective'
        logprobs = nn.functional.log_softmax(preds, dim = -1)
        loss = -(y * logprobs).sum() / preds.shape[0]
        return loss
    
    def train_avgkd(self, model:nn.Module, optim, dloader, loss_fn, device: torch.device, agents_prediction, **kwargs) -> Tuple[float, float]:
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
            if position==0:
                output = model(data)
            else:
                output = model(data, position=position)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1) # type: ignore
            if len(target.size()) > 1 and target.size(1) == 1:
                target = target.squeeze(dim=1)
            agents_prediction_batch = agents_prediction[batch_idx]
            agents_prediction_batch = agents_prediction_batch.to(device)
            # print('agents_prediction_batch', batch_idx, agents_prediction_batch)
            # print('output', batch_idx, output)
            # print(batch_idx, agents_prediction_batch)
            loss = self.soft_loss(output, agents_prediction_batch)
            
            # loss = loss_fn(output, torch.argmax(agents_prediction_batch, dim=1))
            # print(batch_idx, 'agents prediction', torch.argmax(agents_prediction_batch, dim=1))
            # print(batch_idx, 'target', target)
            # loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are the same
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / total_samples
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
        total_samples = 0
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
            total_samples += data.size(0)
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
        acc = correct / total_samples
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
                if position==0:
                    output = model(data)
                else:
                    output = model(data, position=position)
                if len(target.size()) > 1 and target.size(1) == 1:
                    target = target.squeeze(dim=1)
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