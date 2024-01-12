from collections import OrderedDict
from typing import Any, Dict, List
from torch import Tensor
import torch.nn as nn

from algos.base_class import BaseClient, BaseServer

class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    DONE = 0 # Used to signal that the client is done with the current round
    START = 1 # Used to signal by the server to start the current round
    UPDATES = 2 # Used to send the updates from the server to the clients
    CLIENT_STATS = 3 # Used by the client to send the stats to the server


class IsolatedClient(BaseClient):
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
        print("Client {} finished training with loss {}".format(self.node_id, avg_loss))
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    
    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)
        # pass
        return acc

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
            test_acc = self.local_test()
            self.comm_utils.send_signal(dest=self.server_node, data=test_acc, tag=self.tag.CLIENT_STATS)
            # self.log_utils.logging.info("Round {} done".format(round))

class IsolatedServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol

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

    def run_protocol(self):
        self.log_utils.log_console("Starting independent training")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
            for client_node in self.clients:
                self.log_utils.log_console("Server sending semaphore from {} to {}".format(self.node_id,
                                                                                        client_node))
                self.comm_utils.send_signal(dest=client_node, data=None, tag=self.tag.START)
            local_test_acc = self.comm_utils.wait_for_all_clients(self.clients, self.tag.CLIENT_STATS)
            self.update_stats(local_test_acc, self.round)
            self.log_utils.log_console("Round {} done".format(round))
