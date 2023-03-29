import pdb
from typing import List, Tuple
import torch
from algos.algos import synthesize_representations, synthesize_representations_collaborative, synthesize_representations_collaborative_parallel
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


class DistillRepsClient(BaseClient):
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

    def generate_rep(self):
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
                "steps": self.config["steps"],
            }
            reps = synthesize_representations(self.config, obj)
            logits = self.model(reps, position=self.position).detach()
            logit_probs = torch.nn.functional.log_softmax(logits, dim=1) # type: ignore
            return reps, logit_probs
        elif inv_algo.startswith("send_model"):
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
            bs = self.config["distill_batch_size"]
            self.config["inp_shape"][0] = bs
            rep = torch.randn(self.config["inp_shape"]).to(self.device)

        self.comm_utils.send_signal(dest=self.server_node,
                                    data=None,
                                    tag=self.tag.DONE_WARMUP)
        for round in range(self.config["epochs"]):
            # Wait for the server to signal to start the protocol
            self.comm_utils.wait_for_signal(src=self.server_node,
                                            tag=self.tag.START_GEN_REPS)
            rep = self.generate_rep()
            # Send the representations to the server
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=rep,
                                        tag=self.tag.REPS_DONE)
            
            # self.utils.logger.log_image(rep, f"client{self.node_id-1}", epoch)
            # self.log_utils.log_console("Round {} done".format(round))
            # Wait for the server to send the representations
            reps = self.comm_utils.wait_for_signal(src=self.server_node,
                                                    tag=self.tag.REPS_READY)


class DistillRepsServer(BaseServer):
    """
    This is a relay server for the orchestration of the clients
    and not actually a server.
    """
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config

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
            img_batch, label_batch = synthesize_representations_collaborative_parallel(self.config, obj)
            reps = [(img_batch, label_batch)]
            for client in self.clients:
                self.comm_utils.send_signal(dest=client, data=reps, tag=self.tag.REPS_READY)
        elif inv_algo == "send_model_centralized":
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

        for i, rep in enumerate(reps):
            self.log_utils.log_tensor_to_disk(rep, f"node_{i}", self.round)
        
        # Log the representations as images by iterating over each client
        for client, rep in enumerate(reps):
            # Only store first three channel and 64 images for a 8x8 grid
            imgs = rep[0][:64, :3]
            self.log_utils.log_image(imgs, f"client{client+1}", self.round)

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
            self.single_round()
            self.log_utils.log_console("Round {} done".format(round))