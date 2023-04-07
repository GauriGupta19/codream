import math
import pdb
import time
from typing import List, Tuple
import torch
import torch.nn as nn
from algos.algos import synthesize_representations, synthesize_representations_collaborative, synthesize_representations_collaborative_parallel
from algos.base_class import BaseClient, BaseServer
from algos.modules import kl_loss_fn, total_variation_loss, DeepInversionFeatureHook
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
    FINAL_REPS = 6 # Used by the clients to send the output of the final layer to the server
    START_DISTILL = 7 # Used by the server to signal the clients to start distillation


class DistillRepsClient(BaseClient):
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config
        self.position = config["position"]
        self.z_j, self.y_j = [], []
        self.set_algo_params()

    def set_algo_params(self):
        self.inversion_algo = self.config.get("inversion_algo", "send_reps")
        self.data_lr = self.config["data_lr"]
        self.global_steps = self.config["global_steps"]
        self.local_steps = self.config["local_steps"]
        self.alpha_preds = self.config["alpha_preds"]
        self.alpha_tv = self.config["alpha_tv"]
        self.alpha_l2 = self.config["alpha_l2"]
        self.alpha_f = self.config["alpha_f"]
        self.distill_batch_size = self.config["distill_batch_size"]
        self.distill_epochs = self.config["distill_epochs"]
        self.warmup = self.config["warmup"]
        self.first_time_steps = self.config["first_time_steps"]
        self.loss_r_feature_layers = []
        self.EPS = 1e-8
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    def local_warmup(self):
        # Do local warmup rounds
        acc = 0.
        for round in range(self.warmup):
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

    def generate_rep(self, reps: torch.Tensor):
        inputs = reps.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        acts = self.model(inputs, position=self.position)
        probs = torch.softmax(acts, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + self.EPS), dim=1).mean()
        loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(self.loss_r_feature_layers) if hasattr(model, "r_feature")])
        loss = self.alpha_preds * entropy + self.alpha_tv * total_variation_loss(inputs).to(entropy.device) +\
                self.alpha_l2 * torch.linalg.norm(inputs).to(entropy.device) + self.alpha_f * loss_r_feature
        loss.backward()
        return inputs.grad
            
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
            for g_step in range(self.config["global_steps"]):
                # Wait for the server to send the latest representations
                reps = self.comm_utils.wait_for_signal(src=self.server_node,
                                                      tag=self.tag.START_GEN_REPS)
                reps = reps.to(self.device)
                for l_step in range(self.config["local_steps"]):
                    grads = self.generate_rep(reps)
                # Send the locally updated representations to the server
                self.comm_utils.send_signal(dest=self.server_node,
                                            data=grads.to("cpu"),
                                            tag=self.tag.REPS_DONE)
            # Wait for the server to send the latest representations
            reps = self.comm_utils.wait_for_signal(src = self.server_node,
                                                   tag = self.tag.START_DISTILL)
            reps = reps.to(self.device)
            # send the output of the last layer to the server
            out = self.model(reps, position=self.position).to("cpu")
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=out,
                                        tag=self.tag.FINAL_REPS)


class DistillRepsServer(BaseServer):
    """
    This is a relay server for the orchestration of the clients
    and not actually a server.
    """
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config
        bs = self.config["distill_batch_size"]
        self.config["inp_shape"][0] = bs

    def adam_update(self, grad):
        betas = self.optimizer.param_groups[0]['betas']
        # access the optimizer's state
        state = self.optimizer.state[self.reps]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(self.reps.data)
        if 'exp_avg_sq' not in state:
            state['exp_avg_sq'] = torch.zeros_like(self.reps.data)
        if 'step' not in state:
            state['step'] = 1
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        bias_correction1 = 1 - betas[0] ** state['step']
        bias_correction2 = 1 - betas[1] ** state['step']
        exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
        exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.optimizer.param_groups[0]['eps'])
        step_size = self.optimizer.param_groups[0]['lr'] / bias_correction1
        self.reps.data.addcdiv_(exp_avg, denom, value=-step_size)
        self.optimizer.state[self.reps]['step'] += 1

    def single_round(self):
        start = time.time()
        g_steps = self.config["global_steps"]
        for g_step in range(g_steps):
            for client in self.clients:
                self.comm_utils.send_signal(dest=client, data=self.reps, tag=self.tag.START_GEN_REPS)
            grads = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
            grads = torch.stack(grads)
            self.adam_update(grads.mean(dim=0))
            if g_step % 500 == 0 or g_step== (g_steps - 1):
                print(f"{g_step}/{g_steps}", time.time() - start)
        for client in self.clients:
            self.comm_utils.send_signal(dest=client, data=self.reps, tag=self.tag.START_DISTILL)
        acts = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.FINAL_REPS)
        acts = torch.stack(acts)
        acts = acts.mean(dim=0)
        acts = torch.log_softmax(acts, dim=1)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        return self.reps.detach(), acts.detach()

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
            self.reps = torch.randn(self.config["inp_shape"]).to(self.device)
            self.optimizer = torch.optim.Adam([self.reps], lr=self.config["data_lr"])
            inp, out = self.single_round()
            self.log_utils.log_tensor_to_disk((inp, out), f"node", round)
            # Only store first three channel and 64 images for a 8x8 grid
            imgs = self.reps[:64, :3]
            self.log_utils.log_image(imgs, f"reps", round)
            self.log_utils.log_console("Round {} done".format(round))