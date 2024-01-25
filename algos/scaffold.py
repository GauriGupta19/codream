from collections import OrderedDict
import copy
from typing import Any, Dict, List
from torch import Tensor
import torch.nn as nn
import torch
from torch.optim import Optimizer

from algos.base_class import BaseClient, BaseServer


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    COV = 0 # Used to signal that the client is done with the current round
    START = 1 # Used to signal by the server to start the current round
    WTS = 2 # Used to send the updates from the server to the clients


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                dp = p.grad.data - ci.data + c.data
                p.data.sub_(group['lr'] * dp)

        return loss


class SCAFFOLDClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        # create a dictionary to store the client controls
        self.c_i = OrderedDict()
        for (name, param) in self.model.named_parameters():
            self.c_i[name] = torch.zeros_like(param.data)
        self.optim = ScaffoldOptimizer(self.model.parameters(), lr=self.config["lr_client"], weight_decay=0.0001)
    
    def local_train(self, model: nn.Module, optim, dloader, loss_fn, device, c, c_i):
        """
        Train the model locally
        """
        model.train()
        for i, (x, y) in enumerate(dloader):
            optim.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optim.step(c, c_i)

    def get_grads(self, model, dloader, loss_fn, device):
        # takes a pass through all the records in the dloader and accumulates the gradients
        for i, (x, y) in enumerate(dloader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()

        scaling_factor = 1 / len(dloader)
        return {k: scaling_factor * v.grad for k, v in model.named_parameters()}
 

    def get_weights(self) -> Dict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.state_dict()

    def set_weights(self, weights: OrderedDict[str, Tensor]):
        """
        Set the model weights
        """
        self.model.load_state_dict(weights)

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]

        for round in range(start_epochs, total_epochs):
            x, c = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.START)
            assert(list(c.keys()) == list(self.c_i.keys()))
            # put x and c on the device
            for k, v in x.items():
                x[k] = v.to(self.device)
            for k, v in c.items():
                c[k] = v.to(self.device)
            
            y_i = copy.deepcopy(x)
            self.model.load_state_dict(y_i)
            grad_x = self.get_grads(self.model, self.dloader, self.loss_fn, self.device)
            for i in range(self.config["local_runs"]):
                # new_c, new_c_i = OrderedDict(), OrderedDict()
                # for k, v in self.model.named_parameters():
                #     new_c[k] = c[k]
                #     new_c_i[k] = self.c_i[k]
                self.local_train(self.model, self.optim, self.dloader, self.loss_fn, self.device, c, self.c_i)

            # for every parameter in the model, compute the local pseudo gradient
            # and update the client control
            K = len(self.dloader) * self.config["batch_size"]
            local_pseudo_grad, c_i_plus, c_i_delta = OrderedDict(), OrderedDict(), OrderedDict()
            for k, v in self.model.state_dict().items():
                local_pseudo_grad[k] = v - x[k]
                # Option 1 of Scaffold paper
                try:
                    c_i_plus[k] = grad_x[k].detach()
                    c_i_delta[k] = c_i_plus[k] - self.c_i[k]
                except KeyError:
                    pass
                # Option 2 of Scaffold paper - unstable but efficient
                # c_i_plus[k] = c[k] - self.c_i[k] + (1 / (K * self.config["lr_client"])) * (-1 * local_pseudo_grad[k])

            test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)

            self.comm_utils.send_signal(dest=self.server_node, data=local_pseudo_grad, tag=self.tag.WTS)
            self.comm_utils.send_signal(dest=self.server_node, data=c_i_delta, tag=self.tag.COV)
            self.c_i = c_i_plus

class SCAFFOLDServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.c = OrderedDict()
        for (name, param) in self.model.named_parameters():
            self.c[name] = torch.zeros_like(param.data)
        
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)

    def avg(self, generic_tensors: List[OrderedDict[str, Tensor]]):
        # All models are sampled currently at every round
        # Each model is assumed to have equal amount of data and hence
        # coeff is same for everyone
        num_clients = len(generic_tensors)
        coeff = 1 / num_clients
        avgd_tensors = OrderedDict()
        first_model = generic_tensors[0]

        for client_num in range(num_clients):
            local_tensors = generic_tensors[client_num]
            for key in first_model.keys():
                if client_num == 0:
                    avgd_tensors[key] = coeff * local_tensors[key].detach().to(self.device)
                else:
                    avgd_tensors[key] += coeff * local_tensors[key].detach().to(self.device)
        return avgd_tensors

    def test(self) -> tuple[float, float]:
        """
        Test the model on the server
        """
        test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return test_loss, acc

    def update_weights(self, delta_x: OrderedDict[str, Tensor]):
        """
        Update the model weights
        """
        # print('%.6f' % delta_x['param1'].item(), "\tSERVER WEIGHT UPDATE")
        state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            state_dict[k] = v + self.config["lr_server"] * delta_x[k]

        self.model.load_state_dict(state_dict)
        # print('%.6f' % self.model.state_dict()['param1'].item(), "\tSERVER WEIGHT NEW")

    def update_covariates(self, delta_c: OrderedDict[str, Tensor]):
        """
        Update the covariates
        """
        # print('%.6f' % delta_c['param1'].item(), "\tSERVER CONTROL UPDATE")

        for k, v in self.c.items():
            self.c[k] = v + delta_c[k]
        
        # print('%.6f' % self.c['param1'].item(), "\tSERVER CONTROL NEW")

    def get_covariates(self) -> Dict[str, Tensor]:
        """
        Get the covariates
        """
        return self.c
    
    def get_weights(self) -> Dict[str, Tensor]:
        """
        Get the model weights
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def single_round(self):
        """
        Runs the whole training procedure
        """
        updates = (self.get_weights(), self.get_covariates())
        for client_node in self.clients:
            self.comm_utils.send_signal(dest=client_node, data=updates, tag=self.tag.START)
        self.log_utils.log_console("Server waiting for all clients to finish")
        local_updates_wts = self.comm_utils.wait_for_all_clients(self.clients, self.tag.WTS)
        local_updates_cov = self.comm_utils.wait_for_all_clients(self.clients, self.tag.COV)
        self.log_utils.log_console("Server received all clients done signal")
        agg_delta_x = self.avg(local_updates_wts)
        agg_delta_c = self.avg(local_updates_cov)
        self.update_weights(agg_delta_x)
        self.update_covariates(agg_delta_c)

    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()
            test_loss, acc = self.test()
            print(f"round {round} loss: {test_loss}")
            self.log_utils.log_tb(f"test_acc/clients", acc, round)
            self.log_utils.log_console("round: {} test_acc:{:.4f}".format(
                round, acc
            ))
            self.log_utils.log_console("Round {} done".format(round))
