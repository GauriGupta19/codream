from collections import OrderedDict
from typing import Any, Dict, List
from torch import Tensor
from torch import no_grad
import torch.nn as nn

from algos.base_class import BaseClient, BaseServer


def put_on_cpu(wts):
    for k, v in wts.items():
        wts[k] = v.to("cpu")
    return wts

def repr_on_cpu()


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

    def local_train(self):
        """
        Train the model locally
        """
        avg_loss = self.model_utils.train(self.model, self.optim,
                                          self.dloader, self.loss_fn,
                                          self.device)
        
        # 1. trial code for doing inference here, but we need to add this in server
        # 2. write a new train function for avgkd, and replace the target with new predicted labels that you get from 
        # 3. make dict while communicating
        # make sure that target and predicted labels average have same dimensions, 256X and 256X1 shouldn't be there

        # all_predicted = []
        # for batch_idx, (data, target) in enumerate(self.dloader):
        #     data, target = data.to(self.device), target.to(self.device)
            
        #     output = self.model(data)
        #     pred = output.argmax(dim=1, keepdim=True)
        #     print(batch_idx, type(pred), pred.shape, pred)
        #     print(type(target), target.shape, target)
        #     if len(target.size()) > 1:
        #         target = target.argmax(dim=1, keepdim=True)
        #     correct += pred.eq(target.view_as(pred)).sum().item()
        # acc = correct / total_samples
        # return train_loss, acc

        print("Client {} finished training with loss {}".format(self.node_id, avg_loss))
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    
    def local_test(self, **kwargs):
        """
        Test the model locally
        """
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
        return (self.dloader, self.model)

    # def set_labels(self, representation: Dict[str, Tensor]):
    #     """
    #     Set the model weights
    #     """
    #     self.model.load_state_dict(representation)

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            # self.log_utils.logging.info("Client waiting for semaphore from {}".format(self.server_node))
            # print("Client waiting for semaphore from {}".format(self.server_node))
            self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.START)
            # self.log_utils.logging.info("Client received semaphore from {}".format(self.server_node))
            for i in range(self.config["local_runs"]):
                self.local_train()
            repr = put_on_cpu(self.get_representation())
            # self.log_utils.logging.info("Client {} sending done signal to {}".format(self.node_id, self.server_node))
            self.comm_utils.send_signal(dest=self.server_node, data=repr, tag=self.tag.DONE)
            # self.log_utils.logging.info("Client {} waiting to get new model from {}".format(self.node_id, self.server_node))
            repr = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.UPDATES)
            # self.log_utils.logging.info("Client {} received new model from {}".format(self.node_id, self.server_node))
            self.set_representation(repr)
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
                self.log_utils.log_tb(f"test_acc/client{client}", stat, round)

    def fed_avg(self, model_wts: List[OrderedDict[str, Tensor]]):
        # All models are sampled currently at every round
        # Each model is assumed to have equal amount of data and hence
        # coeff is same for everyone
        num_clients = len(model_wts)
        coeff = 1 / num_clients
        avgd_wts = OrderedDict()
        first_model = model_wts[0]

        for client_num in range(num_clients):
            local_wts = model_wts[client_num]
            for key in first_model.keys():
                if client_num == 0:
                    avgd_wts[key] = coeff * local_wts[key].to(self.device)
                else:
                    avgd_wts[key] += coeff * local_wts[key].to(self.device)
        return avgd_wts

    def aggregate(self, representation_list: List[OrderedDict[str, Tensor]]):
        """
        Aggregate the model weights
        """
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

    def set_representation(self, representation):
        """
        Set the model
        """
        # put it on cpu first due to supercloud incompatibility
        self.model.load_state_dict(representation)

    def test(self) -> float:
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
        return acc

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
        avg_wts = self.aggregate(reprs)
        avg_wts = put_on_cpu(avg_wts)
        self.set_representation(avg_wts)
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node,
                                        avg_wts,
                                        self.tag.UPDATES)
        local_test_acc = self.comm_utils.wait_for_all_clients(self.clients, self.tag.CLIENT_STATS)
        self.update_stats(local_test_acc, self.round)

    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()
            acc = self.test()
            self.log_utils.log_tb(f"test_acc", acc, round)
            self.log_utils.log_console("round: {} test_acc:{:.4f}".format(
                round, acc
            ))
            self.log_utils.log_console("round: {} Best test_acc:{:.4f}".format(
                round, self.best_acc
            ))
            self.log_utils.log_console("Round {} done".format(round))
