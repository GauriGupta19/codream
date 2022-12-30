from collections import OrderedDict
from typing import Any, Dict, List
import torch.nn as nn

from algos.base_class import BaseClient, BaseServer


class FedAvgClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        # self.set_parameters()

    def local_train(self):
        """
        Train the model locally
        """
        avg_loss = self.model_utils.train(self.model, self.optim,
                                          self.dloader, self.loss_fn,
                                          self.device)
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    
    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        pass

    def get_representation(self) -> Dict[str, Any]:
        """
        Share the model weights
        """
        return self.model.module.state_dict()

    def set_representation(self, representation: Dict[str, Any]):
        """
        Set the model weights
        """
        self.model.module.load_state_dict(representation)

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            # self.log_utils.logging.info("Client waiting for semaphore from {}".format(self.server_node))
            self.comm_utils.wait_for_signal(src=self.server_node, tag=self.TAG_START)
            # self.log_utils.logging.info("Client received semaphore from {}".format(self.server_node))
            self.local_train()
            self.local_test()
            repr = self.get_representation()
            # self.log_utils.logging.info("Client {} sending done signal to {}".format(self.node_id, self.server_node))
            self.comm_utils.send_signal(dest=self.server_node, data=repr, tag=self.TAG_DONE)
            # self.log_utils.logging.info("Client {} waiting to get new model from {}".format(self.node_id, self.server_node))
            repr = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.TAG_SET_REP)
            # self.log_utils.logging.info("Client {} received new model from {}".format(self.node_id, self.server_node))
            self.set_representation(repr)
            # self.log_utils.logging.info("Round {} done".format(round))


class FedAvgServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)

    def fed_avg(self, model_wts: List[Dict[str, Any]]):
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
                    avgd_wts[key] = coeff * local_wts[key]
                else:
                    avgd_wts[key] += coeff * local_wts[key]
        return avgd_wts

    def aggregate(self, representation_list: List[Dict[str, Any]]):
        """
        Aggregate the model weights
        """
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

    def set_representation(self, representation):
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node,
                                        representation,
                                        self.TAG_SET_REP)

    def test(self):
        """
        Test the model on the server
        """
        test_loss, acc = self.model_utils.test(self._test_loader,
                                               self.model,
                                               self.loss_fn,
                                               self.device)
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)

    def single_round(self):
        """
        Runs the whole training procedure
        """
        for client_node in self.clients:
            self.log_utils.log_console("Server sending semaphore from {} to {}".format(self.node_id,
                                                                                    client_node))
            self.comm_utils.send_signal(dest=client_node, data=None, tag=self.TAG_START)
        self.log_utils.log_console("Server waiting for all clients to finish")
        reprs = self.comm_utils.wait_for_all_clients(self.clients, self.TAG_DONE)
        self.log_utils.log_console("Server received all clients done signal")
        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)

    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs, total_epochs = self.config["start_epochs"], self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()
            acc = self.test()
            self.log_utils.log_tb(f"test_acc/clients", acc, round)
            self.log_utils.log_console("round: {} test_acc:{:.4f}".format(
                round, acc
            ))
            self.log_utils.log_console("Round {} done".format(round))
