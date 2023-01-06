import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from resnet import ResNet34

from torch.utils.data import Subset
from torch.utils.data import DataLoader

from utils.data_utils import cifar_extr_noniid


def load_weights(model_dir: str, model: nn.Module, client_num: int):
    wts = torch.load("{}/saved_models/c{}.pt".format(model_dir, client_num))
    model.load_state_dict(wts)
    print(f"successfully loaded checkpoint for client {client_num}")
    return model


class ServerObj():
    def __init__(self, config, obj, rank) -> None:
        self.num_clients, self.samples_per_client = config["num_clients"], config["samples_per_client"]
        self.device, self.device_id = obj["device"], obj["device_id"]
        test_dataset = obj["dset_obj"].test_dset
        batch_size = config["batch_size"]

        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.model = ResNet34().to(self.device)


class ClientObj():
    def __init__(self, config, obj, rank) -> None:
        self.num_clients, self.samples_per_client = config["num_clients"], config["samples_per_client"]
        self.device, self.device_id = obj["device"], obj["device_id"]
        train_dataset, test_dataset = obj["dset_obj"].train_dset, obj["dset_obj"].test_dset
        batch_size, lr = config["batch_size"], config["model_lr"]
        
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        indices = np.random.permutation(len(train_dataset))

        optim = torch.optim.Adam
        self.model = ResNet34()
        self.model = DataParallel(self.model.to(self.device), device_ids=self.device_id)
        self.optim = optim(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        # rank-1 because rank 0 is the server
        self.c_dset = Subset(train_dataset, indices[(rank-1)*self.samples_per_client:rank*self.samples_per_client])
        self.c_dloader = DataLoader(self.c_dset, batch_size=batch_size)


# class WebObj():
#     def __init__(self, config, obj, rank) -> None:
#         """ The purpose of this class is to bootstrap the objects for the whole distributed training
#         setup
#         """
#         self.num_clients, self.samples_per_client = config["num_clients"], config["samples_per_client"]
#         self.device, self.device_id = obj["device"], obj["device_id"]
#         train_dataset, test_dataset = obj["dset_obj"].train_dset, obj["dset_obj"].test_dset
#         batch_size, lr = config["batch_size"], config["model_lr"]
        
#         # train_loader = DataLoader(train_dataset, batch_size=batch_size)
#         self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
#         indices = np.random.permutation(len(train_dataset))

#         optim = torch.optim.Adam
#         self.c_models = []
#         self.c_optims = []
#         self.c_dsets = []
#         self.c_dloaders = []

#         for i in range(self.num_clients):
#             model = ResNet34()
#             if config["load_existing"]:
#                 model = load_weights(config["results_path"], model, i)
#             c_model = nn.DataParallel(model.to(self.device), device_ids=self.device_ids)
#             c_optim = optim(c_model.parameters(), lr=lr)
#             if config["exp_type"].startswith("non_iid"):
#                 if i == 0:
#                     # only need to call this func once since it returns all user_groups
#                     user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset,
#                                                                             config["num_clients"], config["class_per_client"],
#                                                                             config["samples_per_client"], rate_unbalance=1)
#                 c_dset = Subset(train_dataset, user_groups_train[i].astype(int))
#             else:
#                 c_idx = indices[i*self.samples_per_client: (i+1)*self.samples_per_client]
#                 c_dset = Subset(train_dataset, c_idx)
            
#             c_dloader = DataLoader(c_dset, batch_size=batch_size*len(self.device_ids), shuffle=True)

#             self.c_models.append(c_model)
#             self.c_optims.append(c_optim)
#             self.c_dsets.append(c_dset)
#             self.c_dloaders.append(c_dloader)
#             print(f"Client {i} initialized")