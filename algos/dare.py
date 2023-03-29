import pdb
from typing import List, Tuple
import torch
from algos.algos import synthesize_representations, synthesize_representations_collaborative
from algos.base_class import BaseClient, BaseServer
from algos.modules import kl_loss_fn
from torch.utils.data import TensorDataset, DataLoader


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    DONE_WARMUP = 0
    START_WARMUP = 1 # Used to signal by the server to start the warmup rounds
    START_GEN_REPS = 2 # Used to signal by the server to start generating representations
    REPS_DONE = 3 # Used to signal by the client to the server that it has finished generating representations
    REPS_READY = 4 # Used to signal by the server to the client that it can start using the representations from other clients
    CLIENT_STATS = 5 # Used by the client to send the stats to the server


class DAREClient(BaseClient):
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config
        self.position = config["position"]
        self.z_j, self.y_j = [], []

    def local_warmup(self):
        warmup = self.config["warmup"]
        # Do local warmup rounds
        acc = 0.
        for round in range(warmup):
            # Do local warmup rounds
            loss, acc = self.model_utils.train(self.model,
                                               self.optim,
                                               self.dloader,
                                               self.loss_fn,
                                               self.device,
                                               epoch=round)
            print("Local Warmup -- loss {}, acc {}, round {}".format(loss, acc, round))

        test_loss, test_acc = self.model_utils.test(self.model,
                                                    self._test_loader,
                                                    self.loss_fn,
                                                    self.device)
        # self.log_utils.log_console("Warmup round done for node {}".format(self.node_id))
        print("Warmup round done for node {}, train_acc {}, test_acc {}".format(self.node_id,
                                                                                acc,
                                                                                test_acc))
        # save the model if the test loss is lower than the best loss
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.model_utils.save_model(self.model, self.config["saved_models"] + f"user{self.node_id}.pt")

    def select_representations(self, reps: List[Tuple[torch.Tensor, torch.Tensor]], k: int) -> DataLoader:
        """Returns a dataloader that consists of the top k representations

        Args:
            reps (List[torch.Tensor]): List of representations
            k (int): Number of top representations to return

        Returns:
            torch.Tensor: Indices of the top k representations
        """
        # kl_divs = []
        # for rep in reps:
        #     # i represents the student and j represents a potential teacher
        #     # rep[0] is the representation and rep[1] is the label
        #     z_j = rep[0]
        #     y_j = rep[1]
        #     y_i = self.model(z_j, position=self.position)
        #     # turn it into probability distribution
        #     y_i = torch.nn.functional.softmax(y_i, dim=1)
        #     # TODO: make sure this is performing sum reduction and not mean reduction
        #     kl_divs.append(kl_loss_fn(y_i, y_j))

        # kl_divs = torch.tensor(kl_divs)
        # top_k_ind = kl_divs.argsort(descending=True)[:k]
        # self.log_utils.log_console("Top k indices: {} for node_{}".format(top_k, self.node_id))
        # z_j, y_j = [], []
        # for ind in top_k_ind:
        # Currently we are only sharing a single batch of representations
        self.z_j.append(reps[0][0])
        self.y_j.append(reps[0][1])
        # only store latest n items in the z_j and y_j lists
        n = 20
        if len(self.z_j) > n:
            self.z_j.pop(0)
            self.y_j.pop(0)
        r_j = TensorDataset(torch.stack(self.z_j).flatten(0, 1),
                            torch.stack(self.y_j).flatten(0, 1))
        dloader_reps = DataLoader(r_j,
                                  batch_size=self.config["distill_batch_size"])
        print(self.z_j[0].shape)
        return dloader_reps
    
    def generate_rep(self, reps, labels, first_time):
        inv_algo = self.config.get("inversion_algo", "send_reps")
        if inv_algo =="send_reps":
            bs = self.config["distill_batch_size"]
            self.config["inp_shape"][0] = bs
            labels = next(iter(self.dloader))[1][:bs].to(self.device)
            reps = torch.randn(self.config["inp_shape"]).to(self.device)
            obj = {
                "model": self.model,
                "orig_img": reps,
                "target_label": labels,
                "steps": self.config["first_time_steps"] if first_time else self.config["steps"],
            }
            reps = synthesize_representations(self.config, obj)
            logits = self.model(reps, position=self.position).detach()
            logit_probs = torch.nn.functional.log_softmax(logits, dim=1) # type: ignore
            return reps, logit_probs
        elif inv_algo  == "send_model":
            return self.model.module.state_dict()

    def run_protocol(self):
        # Wait for the server to signal to start local warmup rounds
        self.comm_utils.wait_for_signal(src=0, tag=self.tag.START_WARMUP)
        # self.log_utils.log_console("Starting local warmup rounds")
        if not self.config["load_existing"]:
            self.local_warmup()
        else:
            print("skipping local warmup because checkpoints are loaded")
            test_loss, test_acc = self.model_utils.test(self.model,
                                                    self._test_loader,
                                                    self.loss_fn,
                                                    self.device)
            print("test_acc {}".format(test_acc))
        # self.log_utils.log_console("Local warmup rounds done")
        # Signal to the server that the local warmup rounds are done
        self.comm_utils.send_signal(dest=self.server_node,
                                    data=None,
                                    tag=self.tag.DONE_WARMUP)


        for round in range(self.config["epochs"]):
            # Wait for the server to signal to start the protocol
            self.comm_utils.wait_for_signal(src=self.server_node,
                                            tag=self.tag.START_GEN_REPS)
            bs = self.config["distill_batch_size"]
            self.config["inp_shape"][0] = bs
            labels = next(iter(self.dloader))[1][:bs].to(self.device)
            rep = torch.randn(self.config["inp_shape"]).to(self.device)
            rep = self.generate_rep(rep, labels, round==self.config["warmup"])
            # Send the representations to the server
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=rep,
                                        tag=self.tag.REPS_DONE)
            
            # self.utils.logger.log_image(rep, f"client{self.node_id-1}", epoch)
            # self.log_utils.log_console("Round {} done".format(round))
            # Wait for the server to send the representations
            reps = self.comm_utils.wait_for_signal(src=self.server_node,
                                                   tag=self.tag.REPS_READY)

            # move the representations to the device
            # reps = self.model_utils.move_to_device(reps, self.device)

            # Select the top k representations
            # dloader_reps = self.select_representations(reps, self.config["top_k"])

            # for epoch in range(self.config["distill_epochs"]):
            #     # Train the model using the selected representations
            #     student_loss, student_acc = self.model_utils.train(self.model,
            #                                                     self.optim,
            #                                                     dloader_reps,
            #                                                     kl_loss_fn,
            #                                                     self.device,
            #                                                     apply_softmax=True,
            #                                                     position=self.position)
            # print("student_loss: {}, student_acc: {}".format(student_loss, student_acc))
            # # Train the model on the local data
            # tr_loss, tr_acc = self.model_utils.train(self.model,
            #                                          self.optim,
            #                                          self.dloader,
            #                                          self.loss_fn,
            #                                          self.device)
            # print("tr_loss: {}, tr_acc: {}".format(tr_loss, tr_acc))
            # # Test the model on the test data
            # test_loss, test_acc = self.model_utils.test(self.model,
            #                                             self._test_loader,
            #                                             self.loss_fn,
            #                                             self.device)
            # print("test_loss: {}, test_acc: {}".format(test_loss, test_acc))
            # self.log_utils.log_console("Round {} done for node_{}".format(round, self.node_id))
            current_stats = None
            # current_stats = [student_loss, student_acc,
            #                  tr_loss, tr_acc,
            #                  test_loss, test_acc]
            # send it to the server

            self.comm_utils.send_signal(dest=self.server_node,
                                        data=current_stats,
                                        tag=self.tag.CLIENT_STATS)

            # save the model if the test loss is lower than the best loss
            # if test_loss < self.best_loss:
            #     self.best_loss = test_loss
            #     self.model_utils.save_model(self.model, self.config["saved_models"] + f"user{self.node_id}.pt")

            print("Round {} done for node_{}".format(round, self.node_id))


class DAREServer(BaseServer):
    """
    This is a relay server for the orchestration of the clients
    and not actually a server.
    """
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config

    # def set_parameters(self, config):
    #     # this function basically overrides the set_parameters function
    #     # of the BaseServer class
    #     pass

    # def setup_cuda(self, config):
    #     # this function basically overrides the setup_cuda function
    #     # of the BaseServer class because the server is only for relay
    #     pass

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
                self.log_utils.log_tb(f"student_loss/client{client}", stat[0], round)
                self.log_utils.log_tb(f"student_acc/client{client}", stat[1], round)
                self.log_utils.log_tb(f"train_loss/client{client}", stat[2], round)
                self.log_utils.log_tb(f"train_acc/client{client}", stat[3], round)
                self.log_utils.log_tb(f"test_loss/client{client}", stat[4], round)
                self.log_utils.log_tb(f"test_acc/client{client}", stat[5], round)
                self.log_utils.log_console(f"Round {round} test accuracy for client {client}: {stat[5]}")

    def single_round(self):
        # Get all the students to start synthesizing representations
        for client in self.clients:
            self.comm_utils.send_signal(dest=client, data=None, tag=self.tag.START_GEN_REPS)
        # Wait for the students to finish synthesizing representations
        self.log_utils.log_console("Waiting for students to finish generating representations")
        reps = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
        self.log_utils.log_console("Sending representations to all students")
        inv_algo = self.config.get("inversion_algo", "send_reps")
        if inv_algo == "send_model":
            bs = self.config["distill_batch_size"]
            self.config["inp_shape"][0] = bs
            obj = {
                "model_wts": reps,
                "orig_img": torch.randn(self.config["inp_shape"]).to(self.device),
                "steps": self.config["steps"], "device": self.device, "device_ids": self.device_ids,
            }
            img_batch, label_batch = synthesize_representations_collaborative(self.config, obj)
            reps = [(img_batch, label_batch)]
            for client in self.clients:
                self.comm_utils.send_signal(dest=client, data=reps, tag=self.tag.REPS_READY)
        else:
            # Send representation from every student to every other student
            for client in self.clients:
                # We want to send all the representations to a given client except the one from the client itself
                # We subtract 1 from the client id because the client ids start from 1 while the list indices start from 0
                client_rep = reps[:client - 1] + reps[client:]
                self.comm_utils.send_signal(dest=client, data=client_rep, tag=self.tag.REPS_READY)

        #TODO: Remove this later but for now we will train a model on the representations
        # Move the representations to the device
        # self.log_utils.log_console("Training the server model on the representations")
        # save reps to files numbered by the current round in pt format
        for i, rep in enumerate(reps):
            self.log_utils.log_tensor_to_disk(rep, f"node_{i}", self.round)
        # Create a dataloader for the server using reps
        # for rep in reps:
        #     self.z_list.append(rep[0].to(self.device))
        #     self.y_list.append(rep[1].to(self.device))
        # r_j = TensorDataset(torch.stack(self.z_list).flatten(0, 1),
        #                     torch.stack(self.y_list).flatten(0, 1))
        # dloader_reps = DataLoader(r_j,
        #                           batch_size=self.config["distill_batch_size"])
        # for epoch in range(self.config["distill_epochs"]):
        #     tr_loss, tr_acc = self.model_utils.train(self.model,
        #                         self.optim,
        #                         dloader_reps,
        #                         kl_loss_fn,
        #                         self.device,
        #                         apply_softmax=True)
        #     self.log_utils.log_console(f"Server train accuracy: {tr_acc}, Server train loss: {tr_loss}")
        # loss, acc = self.model_utils.test(self.model,
        #                                 self._test_loader,
        #                                 self.loss_fn,
        #                                 self.device)
        # self.model_utils.save_model(self.model, self.config["saved_models"] + f"user{self.node_id}.pt")
        # self.log_utils.log_console(f"Server test accuracy: {acc}")
        # self.log_utils.log_console(f"Server test loss: {loss}")
        # self.log_utils.log_tb("server_loss", loss, self.round)
        # self.log_utils.log_tb("server_acc", acc, self.round)
        
        # Log the representations as images by iterating over each client
        for client, rep in enumerate(reps):
            # Only store first three channel and 64 images for a 8x8 grid
            imgs = rep[0][:64, :3]
            self.log_utils.log_image(imgs, f"client{client+1}", self.round)
        # Wait for the students to finish training
        self.log_utils.log_console("Waiting for students to finish training")
        stats = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.CLIENT_STATS)
        # Update the stats
        return stats

    def run_protocol(self):
        self.log_utils.log_console("Starting Server")
        # Get all the students to start local warmup rounds
        for client in self.clients:
            self.comm_utils.send_signal(dest=client, data=None, tag=self.tag.START_WARMUP)
        # Wait for the students to finish warmup rounds
        self.log_utils.log_console("Waiting for students to finish warmup rounds")
        self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.DONE_WARMUP)
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
            stats = self.single_round()
            self.update_stats(stats, round)
            self.log_utils.log_console("Round {} done".format(round))