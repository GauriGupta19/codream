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

        # print(len(self.param_groups[0]['params']))
        # print(len(server_controls.keys()))
        # print(len(client_controls.keys()))
        # raise Exception("stop here")

        for group in self.param_groups:
            # print(group['params'])
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print(client_controls.values())
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    print("what is going on?", p.shape, c.shape, ci.shape)
                    continue
                dp = p.grad.data - ci.data + c.data
                p.data.sub_(group['lr'] * dp)

            # p = group['params'][0]
            # p.data.sub_(group['lr'] * (p.grad.data + server_controls[group['name']].data - client_controls[group['name']].data))

        return loss


class SCAFFOLDClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        # create a dictionary to store the client controls
        self.c_i = OrderedDict()
        for n, p in self.model.named_parameters():
            self.c_i[n] = torch.zeros_like(p)
        # print(list(self.model.named_parameters()))
        # raise Exception("stop here")
        # print("data loader length", len(self.dloader))
        # self.optim = torch.optim.SGD(self.model.parameters(), lr=self.config["lr_client"], weight_decay=1e-4)
        self.optim = ScaffoldOptimizer(self.model.parameters(), lr=self.config["lr_client"], weight_decay=0.0001)

        # named_dicts = [
        #     {
        #         'params': [param],
        #         'lr': self.config['lr_client'],
        #         'name': name,
        #     } for (name, param) in self.model.named_parameters()
        # ]
        # self.optim = ScaffoldOptimizer(named_dicts, lr=self.config["lr_client"], weight_decay=0.0001)
    
    def local_train(self, model, optim, dloader, loss_fn, device, c, c_i):
        """
        Train the model locally
        """
        model.train()
        for i, (x, y) in enumerate(dloader):
            optim.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            # print("Shape of x,y, y_hat: ", x.shape, y.shape, y_hat.shape)
            loss = loss_fn(y_hat, y)
            loss.backward()
            # for param, server_control, client_control in zip(self.model.parameters(), c.values(), c_i.values()):
            #     param.grad += server_control - client_control
            optim.step({k: torch.zeros_like(c1).to(device) for k, c1 in c.items()}, {k: torch.zeros_like(c_i1).to(device) for k, c_i1 in c_i.items()})
            # optim.step(c, c_i)
            # optim.step()
            if i % (len(dloader) // 4) == 0:
                # test_loss, acc = self.model_utils.test(self.model,
                #                                self._test_loader,
                #                                self.loss_fn,
                #                                self.device)
                # print(f"\tClient {self.node_id} round {i}", 
                #   f"Loss: {test_loss}",
                #   f"Acc: {acc}",
                #   )
                pass

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
            # print(f"\tClient {self.node_id}", 
            #       "\n\t\treceived x", '%.6f' % x['param1'].item(),
            #       "\n\t\treceived c", '%.6f' % c['param1'].item()
            # )

            # put x and c on the device
            for k, v in x.items():
                x[k] = v.to(self.device)
            for k, v in c.items():
                c[k] = v.to(self.device)
            y_i = copy.deepcopy(x)
            self.model.load_state_dict(y_i)
            self.local_train(self.model, self.optim, self.dloader, self.loss_fn, self.device, c, self.c_i)
            # for every parameter in the model, compute the local pseudo gradient
            # and update the client control
            K = len(self.dloader)
            local_pseudo_grad, c_i_plus, c_i_delta = OrderedDict(), OrderedDict(), OrderedDict()
            for k, v in self.model.named_parameters():
                local_pseudo_grad[k] = v - x[k]
                c_i_plus[k] = c[k] - self.c_i[k] + (1 / (K * self.config["lr_client"])) * (-1 * local_pseudo_grad[k])
                c_i_delta[k] = c_i_plus[k] - self.c_i[k]
            
            # print (f"\tClient {self.node_id}", "\t\t After Training")
            # print(f"\tClient {self.node_id}", "\t\t", '%.6f' % c_i_plus['param1'].item(), "c_i_plus")
            # print(f"\tClient {self.node_id}", "\t\t", '%.6f' % self.model.state_dict()['param1'].item(), "y")

            # test_loss, acc = self.model_utils.test(self.model,
            #                                    self._test_loader,
            #                                    self.loss_fn,
            #                                    self.device)
            # print(f"\tClient {self.node_id}", 
            #       "\n\t\tc_i_plus", '%.6f' % c_i_plus['param1'].item(), 
            #       "\n\t\tc_i_delta", '%.6f' % c_i_delta['param1'].item(), 
            #       "\n\t\ty_i", '%.6f' % self.model.state_dict()['param1'].item(),
            #       "\n\t\ttest loss", test_loss, 
            #       "\n\t\tacc", acc)
                  
            self.comm_utils.send_signal(dest=self.server_node, data=local_pseudo_grad, tag=self.tag.WTS)
            self.comm_utils.send_signal(dest=self.server_node, data=c_i_delta, tag=self.tag.COV)
            self.c_i = c_i_plus

# TODO: test smaller model, test degenerate to fedavg, test other implementations with resnet, look over implementations thoroughly

class SCAFFOLDServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.c = OrderedDict()
        for n, p in self.model.named_parameters():
            self.c[n] = torch.zeros_like(p)

        # print(self.c.keys()) 'lin2.weight'
        # raise Exception("stop here")
        
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
        state_dict = self.model.state_dict()
        for k, v in self.model.named_parameters():
            state_dict[k] = v + self.config["lr_server"] * delta_x[k]


        # print("\tTEST WEIGHT", torch.linalg.norm(delta_x['lin2.weight']).item())
        # print("\tMODEL UPDATED?", torch.linalg.norm(self.model.state_dict()['lin2.weight']).item())
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
        c_temp = OrderedDict()
        for k, _ in self.model.named_parameters():
            c_temp[k] = copy.deepcopy(self.c[k]).to("cpu")
        return c_temp
    
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
