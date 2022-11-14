import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import Subset
from torch.utils.data import DataLoader


class WebObj():
    def __init__(self, config, obj) -> None:
        """ The purpose of this class is to bootstrap the objects for the whole distributed training
        setup
        """
        self.num_clients, self.samples_per_client = config["num_clients"], config["samples_per_client"]
        self.device, self.device_ids = obj["device"], obj["device_ids"]
        train_dataset, test_dataset = obj["dset_obj"].train_dset, obj["dset_obj"].test_dset
        batch_size, lr = config["batch_size"], config["model_lr"]
        
        # train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        indices = np.random.permutation(len(train_dataset))

        optim = torch.optim.Adam
        self.c_models = []
        self.c_optims = []
        self.c_dsets = []
        self.c_dloaders = []

        for i in range(self.num_clients):
            model = models.resnet34()
            if config["load_existing"]:
                model = self.load_weights(config["results_path"], model, i)
            c_model = nn.DataParallel(model.to(self.device), device_ids=self.device_ids)
            c_optim = optim(c_model.parameters(), lr=lr)
            c_idx = indices[i*self.samples_per_client: (i+1)*self.samples_per_client]
            c_dset = Subset(train_dataset, c_idx)
            c_dloader = DataLoader(c_dset, batch_size=64*len(self.device_ids), shuffle=True)

            self.c_models.append(c_model)
            self.c_optims.append(c_optim)
            self.c_dsets.append(c_dset)
            self.c_dloaders.append(c_dloader)
            print(f"Client {i} initialized")

    def load_weights(self, model_dir, model, client_num):
        wts = torch.load("{}/saved_models/c{}.pt".format(model_dir, client_num))
        model.load_state_dict(wts)
        print(f"successfully loaded checkpoint for client {client_num}")
        return model