from glob import glob
import jmespath, importlib, os
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Subset
from utils.config_utils import load_config
from utils.data_utils import get_dataset
from utils.log_utils import LogUtils, check_and_create_path
from models.resnet import resnet18, resnet34, resnet50
from PIL import Image

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)
    
class BaseDataset(data.Dataset):
    """docstring for BaseDataset"""
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.format = config["format"]
        self.set_filepaths(config["path"])
        self.device = config["device"]
        self.train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        self.dset = []
        for path in self.filepaths:
            inp, out = self.load_batch(path)
            inp = (inp.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
            list_of_tuples = list(zip(inp, out))
            self.dset += list_of_tuples

    def set_filepaths(self, path):
        filepaths = path + "/*{}".format(self.format)
        self.filepaths = glob(filepaths)

    def load_batch(self, filepath):
        inp, out = torch.load(filepath, map_location=self.device)
        
        return inp, out

    def __getitem__(self, index):
        inp, out = self.dset[index]
        inp = Image.fromarray(inp.transpose(1, 2, 0))
        return self.train_transform(inp), out

    def __len__(self):
        return len(self.dset)
    
def get_model(model_name:str, dset:str, device:torch.device, device_ids:list, num_channels, num_classes) -> nn.Module:
        #TODO: add support for loading checkpointed models
        model_name = model_name.lower()
        if model_name == "resnet18":
            model = resnet18(num_channels, num_classes)
        elif model_name == "resnet34":
            model = resnet34(num_channels, num_classes)
        elif model_name == "resnet50":
            model = resnet50(num_channels, num_classes)
        else:
            raise ValueError(f"Model name {model_name} not supported")
        print(f"Model {model_name} loading on device {device}")
        model = model.to(device)
        #model = DataParallel(model.to(device), device_ids=device_ids)
        return model
    
def test(model, config, dloader, loss_fn, device, **kwargs):
        """TODO: generate docstring
        """
        model.eval()
        test_loss = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for data, target in dloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # output = nn.functional.log_softmax(output, dim=1)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                count += len(output)
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(dloader.dataset)
        return test_loss/count, acc
    
def train(model:nn.Module, config, optim, dloader, loss_fn, device: torch.device, **kwargs):
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        count = 0
        loss_metric = KLDiv(reduction='sum')
        for batch_idx, (data, target) in enumerate(dloader):
            position = config["position"]
            data, target = next(iter(dloader))
            data, target = data.to(device), target.to(device)
            output = model(data, position=position)
            # if kwargs.get("apply_softmax", False):
            #     output = nn.functional.log_softmax(output, dim=1) # type: ignore
            loss = loss_fn(output, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            train_loss += loss_metric(output, target).item()
            count += len(output)
            pred = output.argmax(dim=1, keepdim=True)
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        # acc = correct / (len(dloader.dataset) * 256)
        acc = correct / (220 * 256)
        return train_loss/count, acc
    
    
def run(config, model, optim, scheduler, train_loader, test_loader, loss_fn_ce, loss_fn_kl, log_utils):
    best_test_acc = 0
    # test_acc_l112_ is node 3
    test_acc_list, tr_acc_list = [], []
    for epoch in range(config["epochs"]):
        test_loss, test_acc = test(model, config, test_loader, loss_fn_ce, device)
        test_acc_list.append(test_acc)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
        tr_loss, tr_acc = train(model, config, optim, train_loader, loss_fn_kl, device, apply_softmax=True)
        tr_acc_list.append(tr_acc)
        print("epoch {} test_loss: {:.4f}, test_acc: {:.4f}, tr_loss: {:.4f}, tr_acc: {:.4f}, lr: {:.4f}".format(epoch, test_loss, test_acc, tr_loss, tr_acc, optim.param_groups[0]['lr']))
        # log_utils.log_console("epoch {} test_loss: {:.4f}, test_acc: {:.4f}, tr_loss: {:.4f}, tr_acc: {:.4f}".format(epoch, test_loss, test_acc, tr_loss, tr_acc, optim.param_groups[0]['lr']))
        # log_utils.log_tb("test_loss", test_loss, epoch)
        # log_utils.log_tb("test_acc", test_acc, epoch)
        # log_utils.log_tb("train_loss", tr_loss, epoch)
        # log_utils.log_tb("train_acc", tr_acc, epoch)
        if epoch>40:
            scheduler.step()
        
    print("Best test accuracy: {:.4f}", best_test_acc)
    # log_utils.log_console(f"Best test accuracy: {best_test_acc}")
    
    
    
#load confid
config_path = "./configs/test_config.py"
path = '.'.join(config_path.split('.')[1].split('/')[1:])
print(path)
config = importlib.import_module(path).current_config

config['load_existing'] = config.get('load_existing') or False
config["log_path"] = config["log_path"] + "_train_" + str(config["train_size"])
print(config["log_path"])
check_and_create_path(config["log_path"])
log_utils = LogUtils(config)
log_utils.log_console("Config: {}".format(config))
log_utils.log_console(config["log_path"])
log_utils = None

device_ids = config["device_ids"]
device = torch.device(f'cuda:{device_ids[0]}')

dream_data_config = {
    "format": config["format"],
    "path": config["train_path"],
    "device": device
}
dream_dset = BaseDataset(dream_data_config)
print('Dream set len', len(dream_dset))
# log_utils.log_console(f'Dream set len {len(dream_dset)}')
indices = np.random.permutation(len(dream_dset))[:config["train_size"]]
dream_dset = Subset(dream_dset, indices)
train_loader = DataLoader(dream_dset, batch_size=256)

dset_obj = get_dataset(config["dset"], config["dpath"])
test_dset = dset_obj.test_dset
test_loader = DataLoader(test_dset, batch_size=config["test_batch_size"])

model = get_model(config["model"], config["dset"], device, device_ids, dset_obj.num_channels, dset_obj.NUM_CLS)
# optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
optim = torch.optim.SGD(model.parameters(), 0.2, weight_decay=1e-4,
                                momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 200, eta_min=2e-4)
loss_fn_ce = torch.nn.CrossEntropyLoss(reduction='sum')
loss_fn_kl = KLDiv(T=20)

run(config, model, optim, scheduler, train_loader, test_loader,loss_fn_ce, loss_fn_kl, log_utils)


