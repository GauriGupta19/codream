from collections import OrderedDict, defaultdict
from typing import Any, Dict, List
import torch
from torch import Tensor
from torch import no_grad, cuda
import torch.nn.functional as F
import torch.nn as nn

from algos.base_class import BaseClient, BaseServer


def put_on_cpu(wts):
    for k, v in wts.items():
        wts[k] = v.to("cpu")
    return wts

class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    DONE = 0 # Used to signal that the client is done with the current round
    START = 1 # Used to signal by the server to start the current round
    UPDATES = 2 # Used to send the updates from the server to the clients
    CLIENT_STATS = 3 # Used by the client to send the stats to the server


class AvgKDClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        self.communication_round = 0

    def local_train(self, local_round):
        """
        Train the model locally
        """
        self.model = self.model.to(self.device)
        avg_loss = self.model_utils.train(self.model, self.optim,
                                          self.dloader, self.loss_fn,
                                          self.device)
        if local_round == self.config["local_runs"] - 1:
            print("Client {} finished training locally with loss {} at local round {}".format(self.node_id, avg_loss, local_round))
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)

    def train_avgKD(self, labels, local_round):
        """
        Train the model on averaged labels received from server
        """
        self.model = self.model.to(self.device)
        avg_loss = self.model_utils.train_avgkd(self.model, self.optim,
                                          self.dloader, self.loss_fn,
                                          self.device, labels)

        if local_round == self.config["local_runs"] - 1:
            print("Client {} finished training avgKD with loss {} at local round {}".format(self.node_id, avg_loss, local_round))
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    
    def local_test(self, **kwargs):
        """
        Test the model locally
        """
        self.model = self.model.to(self.device)
        test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)
        # pass
        return acc

    def get_representation(self) -> tuple:
        """
        Share the dataloader and model
        """
        # print('devices', self.dloader.device, self.model.device)
        if self.communication_round == 0:
            self.communication_round = 1
            return (self.model.to('cpu'), self.dloader)
        else:
            return (self.model.to('cpu'),)

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        # print('total_epochs', total_epochs)
        for round in range(start_epochs, total_epochs+1):
            print('round', round)
            # self.log_utils.logging.info("Client waiting for semaphore from {}".format(self.server_node))
            # print("Client waiting for semaphore from {}".format(self.server_node))
            if round == 0:
                warmup_rounds = 50
                for i in range(warmup_rounds):
                # for i in range(self.config["local_runs"]):
                    self.local_train(i)
            else:
                self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.START)
                # self.log_utils.logging.info("Client received semaphore from {}".format(self.server_node))

                repr = {self.node_id: self.get_representation()}
                # self.log_utils.logging.info("Client {} sending done signal to {}".format(self.node_id, self.server_node))
                self.comm_utils.send_signal(dest=self.server_node, data=repr, tag=self.tag.DONE)
                # self.log_utils.logging.info("Client {} waiting to get new model from {}".format(self.node_id, self.server_node))
                labels_per_batch = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.UPDATES)
                # print(self.node_id, labels_per_batch)
                # self.log_utils.logging.info("Client {} received new model from {}".format(self.node_id, self.server_node))

                for i in range(self.config["local_runs"]):
                    # self.local_train()
                    self.local_train(i)

                for i in range(self.config["local_runs"]):
                    # self.local_train()
                    self.train_avgKD(labels_per_batch, i)
                
                # if round == total_epochs-1:
                #     self.comm_utils.send_signal(dest=self.server_node, data=repr, tag=self.tag.DONE)
                
            test_acc = self.local_test()
            self.comm_utils.send_signal(dest=self.server_node, data=test_acc, tag=self.tag.CLIENT_STATS)
            # self.log_utils.logging.info("Round {} done".format(round))


class AvgKDServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
        self.client_dataloaders = {}

    def update_stats(self, stats: List[List[float]], round: int):
        """
        Updates the statistics from all the clients
        the reason the server is doing it because
        the server only has the access to the log file as of now
        Args:
            stats (List[List[float]]): List of statistics of the clients
        """
        for client, stat in enumerate(stats):
            if stat is not None:
                print(f"test_acc/client{client}", stat, round)
                self.log_utils.log_tb(f"test_acc/client{client}", stat, round)

    def single_round(self):
        """
        Runs the whole training procedure
        """
        for client_node in self.clients:
            self.log_utils.log_console("Server sending semaphore from {} to {}".format(self.node_id,
                                                                                    client_node))
            self.comm_utils.send_signal(dest=client_node, data=None, tag=self.tag.START)
        self.log_utils.log_console("Server waiting for all clients to finish")
        reprs = self.comm_utils.wait_for_all_clients(self.clients, self.tag.DONE)
        self.log_utils.log_console("Server received all clients done signal")

        # print('reprs from all clients: ', reprs)
        client_models = {}

        for client_repr in reprs:
            client_node_id = list(client_repr.keys())[0]
            curr_model = client_repr[client_node_id][0]
            client_models[client_node_id] = curr_model
            
            # update self.client_dataloaders only once in the first round
            if len(client_repr[client_node_id]) == 2:
                curr_dataloader = client_repr[client_node_id][1]
                self.client_dataloaders[client_node_id] = curr_dataloader
            
        client_output_all = defaultdict(lambda: defaultdict(list))
        for client_node_1 in self.clients:
            model1 = client_models[client_node_1]
            model1.to(self.device)
            model1.eval()
            for client_node_2 in self.clients:
                if client_node_1 != client_node_2:
                    dloader2 = self.client_dataloaders[client_node_2]
                    for batch_idx, (data, target) in enumerate(dloader2):
                        data = data.to(self.device)

                        # add client 2's (whose dataloader we're working on) target labels one hot encoding only once to the outputs
                        if len(client_output_all[client_node_2][batch_idx]) == 0:
                            target_labels_onehot = F.one_hot(target, num_classes = self.dset_obj.NUM_CLS)
                            client_output_all[client_node_2][batch_idx].append(target_labels_onehot)

                        with no_grad():
                            output = model1(data)
                        output = output.to('cpu')
                        output_softmax = F.softmax(output, dim=-1)
                        client_output_all[client_node_2][batch_idx].append(output_softmax)

        client_new_labels = defaultdict(lambda: defaultdict(Tensor))
        # take average of all 4 (3 other clients models outputs and 1 ground label) for each batch
        for client_node in self.clients:
            for batch in client_output_all[client_node].keys():
                outputs_client_batch = client_output_all[client_node][batch]

                assert len(outputs_client_batch) == len(self.clients)
                
                client_new_labels[client_node][batch] = torch.mean(torch.stack(outputs_client_batch), dim=0)
                # client_new_labels[client_node][batch] = torchlist[0]
                # client_new_labels[client_node][batch] = ((torchlist[1] + torchlist[2] + torchlist[3]) / 3 + torchlist[0]) / 2 
            
        for client_node in self.clients:
            # MODIFY sent signal to labels
            self.comm_utils.send_signal(client_node,
                                        client_new_labels[client_node],
                                        self.tag.UPDATES)
        local_test_acc = self.comm_utils.wait_for_all_clients(self.clients, self.tag.CLIENT_STATS)
        self.update_stats(local_test_acc, self.round)

    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        # server used for 1 less round than clients because of the local training round
        for round in range(start_epochs, total_epochs):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()
            
            self.log_utils.log_console("Round {} done".format(round))
