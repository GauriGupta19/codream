import torch
import numpy as np
import copy

from typing import Any, Dict, List, Tuple
from torch import Tensor
import torch.nn as nn

from algos.base_class import BaseClient, BaseServer

# Based on https://github.com/TsingZ0/PFLlib


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """

    DONE = 0  # Used to signal that the client is done with the current round
    START = 1  # Used to signal by the server to start the current round
    UPDATES = 2  # Used to send the updates from the server to the clients, include classifier + generator param updates
    SEND_CLS_CNT = 3  # Used to send self.class_counts from clients to server
    SEND_QUAL_LAB = 4  # Used after server receives all class_counts and has computed qualified labels, for distributing qualified labels back to clients
    ACK_QUAL_LAB = 5  # Used after clients receive and sets qualified labels from server, signals that training is ready to commence


def put_on_cpu(wts):
    for k, v in wts.items():
        wts[k] = v.to("cpu")
    return wts


class FedGenClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        for param in self.model.parameters():
            param.requires_grad = False
        self.mu = 1
        self.qualified_labels = list()  # TODO figure out when this field is filled
        self.batch_size = self.config["batch_size"]
        assert self.generator is not None

    def local_train(self, **kwargs):
        """
        trains for one round
        where loss is sum of predefined loss function and loss from generator

        calls model gen
        """
        assert self.generator is not None
        assert len(self.qualified_labels) > 0

        avg_loss = self.model_utils.train_fedgen(
            self.model,
            self.optim,
            self.dloader,
            self.loss_fn,
            self.device,
            self.qualified_labels,
            self.batch_size,
            self.generator,
        )
        print("Client {} finished training with loss {}".format(self.node_id, avg_loss))

    def local_test(self, **kwargs):
        pass

    def get_representation(self) -> Dict[str, Tensor]:
        """
        Share the model weights
        """

        return self.model.state_dict()

    def set_representation(
        self,
        model_representation: Dict[str, Tensor],
        gen_model_representation: Dict[str, Tensor],
    ):
        """
        Helper function that is invoked when reciving updates from server
        Note: Not implementing local feature extractor at this moment
        """
        assert self.generator is not None

        self.generator.load_state_dict(gen_model_representation)
        self.model.load_state_dict(model_representation)
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()

    def set_generator(self, generator_init: nn.Module):
        """first step in run_protocol, sets generator from server input"""
        self.generator = generator_init

    # def get_num_classes(self) -> Tensor:
    #     """helper funcion that counts how many entries per class"""
    #     if self.num_classes == None:
    #         print(f"node {self.node_id} does not have any num_classes assigned!")
    #         print(f"self.class_counts is {self.class_counts}")

    #     assert self.num_classes > 0

    #     sample_per_class = torch.zeros(self.num_classes)

    #     # iterate through trainloader
    #     for x, y in self.dloader:
    #         for yy in y:
    #             print(yy)
    #             sample_per_class[yy.item()] += 1

    #     return sample_per_class

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]

        # First send self.class_counts to server
        self.comm_utils.send_signal(
            dest=self.server_node, data=self.class_counts, tag=self.tag.SEND_CLS_CNT
        )

        # then wait for qualified labels
        qualified_labels = self.comm_utils.wait_for_signal(
            src=self.server_node, tag=self.tag.SEND_QUAL_LAB
        )
        # set qualified labels
        self.qualified_labels = qualified_labels
        print(
            f"client {self.node_id} received qualified labels {qualified_labels} from server!"
        )

        # then send ack to server to signal that training is ready to start
        self.comm_utils.send_signal(
            dest=self.server_node, data=None, tag=self.tag.ACK_QUAL_LAB
        )

        # then start regular training
        for round in range(start_epochs, total_epochs):
            self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.START)
            for i in range(self.config["local_runs"]):
                print("training locally")
                self.local_train()

            # then during normal trianing wait for repr from server, then send updats back to server
            # NOTE client does not send generator information to server
            # classifier_param = self.get_representation()
            classifier = self.model

            # self.log_utils.logging.info("Client {} sending done signal to {}".format(self.node_id, self.server_node))
            # self.comm_utils.send_signal(
            #     dest=self.server_node,
            #     data=(classifier_param, self.config["samples_per_client"]),
            #     tag=self.tag.DONE,
            # )

            self.comm_utils.send_signal(
                dest=self.server_node,
                data=(classifier, self.config["samples_per_client"]),
                tag=self.tag.DONE,
            )
            # self.log_utils.logging.info("Client {} waiting to get new model from {}".format(self.node_id, self.server_node))
            repr = self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.UPDATES
            )
            # self.log_utils.logging.info("Client {} received new model from {}".format(self.node_id, self.server_node))
            # TODO repr is a nested list, where first element is classifier and second is generator
            # TODO confirm what server sends to client, and what client sends to server
            self.set_representation(repr[0], repr[1])
            # self.log_utils.logging.info("Round {} done".format(round))


class FedGenServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

        self.qualified_labels = list()
        self.batch_size = config["batch_size"]

        # Hard coded values taken from official implementation
        self.generative_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-2,
            amsgrad=False,
        )
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98
        )

        assert self.generator is not None

    def test(self) -> float:
        """
        Test the classifier model on the server

        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def receive_models(self, reprs) -> Tuple[List[int], List[nn.Module]]:
        """
        step called by single_round, used to gather clients' models for

            1. weight computation
            2. generator training
        """
        all_models = list()
        all_weights = list()
        tot_samples = 0
        for client_input in reprs:
            # TODO verify that this is the correct data structure
            # IE list of lists of model weights OR list of model weights
            client_model, client_num_samples = client_input[0], client_input[1]
            tot_samples += client_num_samples
            all_weights.append(client_num_samples)
            all_models.append(client_model)

        for i, w in enumerate(all_weights):
            all_weights[i] = w / tot_samples

        return all_weights, all_models

    def train_generator(self, weights: List[int], models: List[nn.Module]):
        """
        step called by single round after calling receive_models
        Used to train and update generater
        """
        assert self.generator is not None

        self.generator.train()

        # NOTE taken from fedGen code where local generator is trained for
        # ensemble_epoch / n_teacher_iters = 50/5 = 10
        for _ in range(10):
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generator(labels).to(self.device)

            logits = 0
            for w, model in zip(weights, models):
                # assert model.device == z.device
                model = model.to(self.device)
                model.eval()
                logits += model(z) * w

            self.generative_optimizer.zero_grad()
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.generative_optimizer.step()

        self.generative_lr_scheduler.step()

    def aggregate_parameters(self, all_models: List[nn.Module], all_weights: List[int]):
        """helper function that updates the new parameter for classifier training"""
        assert len(all_models) > 0

        self.model = copy.deepcopy(all_models[0])
        for param in self.model.parameters():
            param.data.zero_()

        # NOTE this procedure should modify the classifier
        for w, client_model in zip(all_weights, all_models):
            for server_param, client_param in zip(
                self.model.parameters(), client_model.parameters()
            ):
                server_param.data += w * client_param.data.clone()

    def get_representation(self, model: nn.Module) -> Dict[str, Tensor]:
        """
        Share the model weights
        """

        return model.state_dict()

    def set_representation(self):
        """
        Helper function called by single round

        Sets updated model and distribute it to client
        where client receives (classifier, generator)
        """
        # classifier = put_on_cpu(self.model)
        # generator = put_on_cpu(self.generator)
        classifier, generator = self.get_representation(
            self.model
        ), self.get_representation(self.generator)
        repr_to_client = (classifier, generator)

        for client_node in self.clients:
            self.comm_utils.send_signal(
                dest=client_node, data=repr_to_client, tag=self.tag.UPDATES
            )

    def single_round(self):
        """
        Runs regular trianing procedure for a single round

        In a single round, server
            1. sends classifier and generator to clients
            2. take weighted average of classifier updates as new global update
            3. train generative model
        """
        for client_node in self.clients:
            self.log_utils.log_console(
                "Server sending semaphore from {} to {}".format(
                    self.node_id, client_node
                )
            )
            self.comm_utils.send_signal(dest=client_node, data=None, tag=self.tag.START)
        self.log_utils.log_console("Server waiting for all clients to finish")
        reprs = self.comm_utils.wait_for_all_clients(
            self.clients, self.tag.DONE
        )  # should receive list where each entry is tuple of (client_i model update, client_i num samples)

        all_weights, all_models = self.receive_models(reprs)
        self.train_generator(all_weights, all_models)
        self.aggregate_parameters(all_models, all_weights)

        self.set_representation()  # distribute classifier and generator updates back to clients

    def run_protocol(self):
        assert self.generator is not None

        self.log_utils.log_console("Starting iid clients fedgen")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]

        # first wait for clients to send class_counts
        self.log_utils.log_console("Server waiting for clients to send class counts\n")
        class_counts_all_clients = self.comm_utils.wait_for_all_clients(
            self.clients, self.tag.SEND_CLS_CNT
        )
        # using class_counts to compute qualified labels
        assert len(class_counts_all_clients) == len(self.clients)
        assert len(self.qualified_labels) == 0

        print(
            "received class count distribution from all clients: ",
            class_counts_all_clients,
        )

        self.num_classes = len(class_counts_all_clients[0])

        # qualified labels is just the sum of client labels that meet the minimum label amount
        # in our case for simplification the min label count is 0
        for client_idx in range(len(class_counts_all_clients)):
            client_label_count = class_counts_all_clients[client_idx]
            for class_id in range(self.num_classes):
                self.qualified_labels.extend(
                    [class_id for _ in range(int(client_label_count[class_id]))]
                )

        # send qualified labels back to clients
        for client_node in self.clients:
            self.log_utils.log_console(
                "Server sending qualified labels from {} to {}".format(
                    self.node_id, client_node
                )
            )
            self.comm_utils.send_signal(
                dest=client_node, data=self.qualified_labels, tag=self.tag.SEND_QUAL_LAB
            )
        """
        gen_weights = put_on_cpu(self.generator.state_dict())

        for client_node in self.clients:
            self.log_utils.log_console(
                "Server sending generator from {} to {}".format(
                    self.node_id, client_node
                )
            )
            self.comm_utils.send_signal(
                dest=client_node, data=gen_weights, tag=self.tag.SEND_GEN
            )
        """
        # block until all clients ack back
        self.log_utils.log_console(
            "Server waiting for all clients to respond to qualified labels \n"
        )
        self.comm_utils.wait_for_all_clients(self.clients, self.tag.ACK_QUAL_LAB)

        # Now regular training can commence
        for round in range(start_epochs, total_epochs):
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()
            self.log_utils.log_tb(f"test_acc/clients", acc, round)
            self.log_utils.log_console("round: {} test_acc:{:.4f}".format(round, acc))
            self.log_utils.log_console(
                "round: {} Best test_acc:{:.4f}".format(round, self.best_acc)
            )
            self.log_utils.log_console("Round {} done".format(round))
