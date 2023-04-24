import math
import pdb
import time
from typing import List, Tuple
import torch
import torch.nn as nn
from algos.algos import synthesize_representations, synthesize_representations_collaborative
from algos.base_class import BaseClient, BaseServer
from algos.modules import DeepInversionFeatureHook, kl_loss_fn, total_variation_loss
from torch.utils.data import TensorDataset, DataLoader

from utils.data_utils import CustomDataset


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
    FINAL_GLOBAL_REPS = 6 # Used by the server to send the final global representations to the clients for a given epoch

class FedDreamClient(BaseClient):
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config
        self.position = config["position"]

    def set_algo_params(self):
        self.epochs = self.config["epochs"]
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
        self.local_train_freq = self.config["local_train_freq"]
        self.loss_r_feature_layers = []
        self.EPS = 1e-8
        self.synth_dset = CustomDataset(self.config)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

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
    
    def generate_rep(self, reps: torch.Tensor):
        inputs = reps.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        acts = self.model(inputs, position=self.position)
        probs = torch.softmax(acts, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + self.EPS), dim=1).mean()
        loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(self.loss_r_feature_layers) if hasattr(model, "r_feature")])
        loss_fed_prox = inputs - reps
        loss = self.alpha_preds * entropy + self.alpha_tv * total_variation_loss(inputs).to(entropy.device) +\
                self.alpha_l2 * torch.linalg.norm(inputs).to(entropy.device) + self.alpha_f * loss_r_feature +\
                0.5 * torch.linalg.norm(loss_fed_prox).to(entropy.device)
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
        # self.log_utils.log_console("Local warmup rounds done")
        # Signal to the server that the local warmup rounds are done
        self.comm_utils.send_signal(dest=self.server_node,
                                    data=None,
                                    tag=self.tag.DONE_WARMUP)

        start_epochs = self.config.get("start_epochs", 0)
        for round in range(start_epochs, self.epochs):
            for g_step in range(self.global_steps):
                # Wait for the server to send the latest representations
                reps = self.comm_utils.wait_for_signal(src=self.server_node,
                                                       tag=self.tag.START_GEN_REPS)
                reps = reps.to(self.device)
                for l_step in range(self.config["local_steps"]):
                    grads = self.generate_rep(reps)
                # Send the grads to the server
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
            x, y = self.comm_utils.wait_for_signal(src=self.server_node,
                                                   tag=self.tag.FINAL_GLOBAL_REPS)
            # self.utils.logger.log_image(rep, f"client{self.node_id-1}", epoch)
            # self.log_utils.log_console("Round {} done".format(round))
            self.synth_dset.append((x, y))

            if round % self.local_train_freq == 0:
                synth_dloader = DataLoader(self.synth_dset, batch_size=3, shuffle=True)
                student_loss, student_acc, tr_loss, tr_acc = 0., 0., 0., 0.
                for epoch in range(self.config["distill_epochs"]):
                    # Train the model using the selected representations
                    student_loss, student_acc = self.model_utils.train(self.model,
                                                                        self.optim,
                                                                        synth_dloader,
                                                                        kl_loss_fn,
                                                                        self.device,
                                                                        apply_softmax=True,
                                                                        position=self.position,
                                                                        extra_batch=True)
                    tr_loss, tr_acc = self.model_utils.train(self.model,
                                                             self.optim,
                                                             self.dloader,
                                                             self.loss_fn,
                                                             self.device)
                print("student_loss: {}, student_acc: {} at client {}".format(student_loss, student_acc, self.node_id))
                print("tr_loss: {}, tr_acc: {}".format(tr_loss, tr_acc))
                test_loss, test_acc = self.model_utils.test(self.model,
                                                            self._test_loader,
                                                            self.loss_fn,
                                                            self.device)
                print("test_loss: {}, test_acc: {}".format(test_loss, test_acc))
                current_stats = [student_loss, student_acc,
                                tr_loss, tr_acc,
                                test_loss, test_acc]
                self.comm_utils.send_signal(dest=self.server_node,
                                            data=current_stats,
                                            tag=self.tag.CLIENT_STATS)
                # save the model if the test loss is lower than the best loss
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    self.model_utils.save_model(self.model, self.config["saved_models"] + f"user{self.node_id}.pt")
            # self.log_utils.log_console("Round {} done for node_{}".format(round, self.node_id))
            print("Round {} done for node_{}".format(round, self.node_id))


class FedDreamServer(BaseServer):
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
        self.adaptive_distill = self.config.get("adaptive_distill", False)
        if self.adaptive_distill:
            self.EPS = 1e-8
            self.lambda_server = self.config.get("lambda_server", 0.1)
            # set up the server model
            self.set_model_parameters(config)
            self.loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
            self.dset = CustomDataset(config)
            test_dset = self.dset_obj.test_dset
            self._test_loader = DataLoader(test_dset, batch_size=self.config["batch_size"], shuffle=False)
            self.eval_loss_fn = nn.CrossEntropyLoss()
        self.local_train_freq = self.config["local_train_freq"]

    def adam_update(self, grad):
        betas = self.data_optimizer.param_groups[0]['betas']
        # access the optimizer's state
        state = self.data_optimizer.state[self.reps]
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
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.data_optimizer.param_groups[0]['eps'])
        step_size = self.data_optimizer.param_groups[0]['lr'] / bias_correction1
        self.reps.data.addcdiv_(exp_avg, denom, value=-step_size)
        self.data_optimizer.state[self.reps]['step'] += 1

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
        start = time.time()
        g_steps = self.config["global_steps"]
        # TODO: choice of 20 is arbitrary so need to experimentally arrive at a better number
        adaptive_distill_start_round = self.config.get("adaptive_distill_start_round", 20)
        for g_step in range(g_steps):
            for client in self.clients:
                self.comm_utils.send_signal(dest=client, data=self.reps.to("cpu"), tag=self.tag.START_GEN_REPS)
            if self.round > adaptive_distill_start_round and self.adaptive_distill:
                # pass reps on the local model and get the gradients
                inputs = self.reps.clone().detach().requires_grad_(True)
                acts = self.model(inputs)
                probs = torch.softmax(acts, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + self.EPS), dim=1).mean()
                loss = entropy
                loss.backward()
                server_grads = inputs.grad
            grads = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
            grads = torch.stack(grads).to(self.device)
            grads = grads.mean(dim=0)
            if self.round > adaptive_distill_start_round and self.adaptive_distill:
                # negative gradient update to maximize entropy of the server model
                # TODO: choice of lambda is arbitrary so need to experimentally arrive at a better number
                grads = grads - self.lambda_server * server_grads
            self.adam_update(grads)
            if g_step % 500 == 0 or g_step== (g_steps - 1):
                print(f"{g_step}/{g_steps}", time.time() - start)
        for client in self.clients:
            self.comm_utils.send_signal(dest=client, data=self.reps.to("cpu"), tag=self.tag.START_DISTILL)
        acts = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.FINAL_REPS)
        acts = torch.stack(acts)
        acts = acts.mean(dim=0)
        acts = torch.log_softmax(acts, dim=1)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        for client in self.clients:
            self.comm_utils.send_signal(dest=client,
                                        data=(self.reps.to("cpu"), acts.to("cpu")),
                                        tag=self.tag.FINAL_GLOBAL_REPS)
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
            self.data_optimizer = torch.optim.Adam([self.reps], lr=self.config["data_lr"])
            inp, out = self.single_round()
            self.dset.append((inp, out))
            if self.adaptive_distill and round > 10 and round % self.local_train_freq == 0:
                self.dloader = DataLoader(self.dset, batch_size=3, shuffle=True)
                # the choice of 20 here is rather arbitrary
                tr_loss, tr_acc = 0., 0.
                for _ in range(20):
                    tr_loss, tr_acc = self.model_utils.train(self.model, self.optim, self.dloader, self.loss_fn, self.device, apply_softmax=True, extra_batch=True)
                te_loss, te_acc = self.model_utils.test(self.model, self._test_loader, self.eval_loss_fn, self.device)
                self.log_utils.log_tb("tr_loss", tr_loss, round)
                self.log_utils.log_tb("tr_acc", tr_acc, round)
                self.log_utils.log_tb("te_loss", te_loss, round)
                self.log_utils.log_tb("te_acc", te_acc, round)
                self.log_utils.log_console(f"Round {round} tr_loss: {tr_loss}, tr_acc: {tr_acc}, te_loss: {te_loss}, te_acc: {te_acc}")
                if te_acc > self.best_acc:
                    self.best_acc = te_acc
                    # save the model
                    self.model_utils.save_model(self.model, self.config["saved_models"] + "server_model.pt")
            if round % self.local_train_freq == 0:
                stats = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.CLIENT_STATS)
                self.update_stats(stats, round)
            self.log_utils.log_tensor_to_disk((inp, out), f"node", round)
            # Only store first three channel and 64 images for a 8x8 grid
            imgs = self.reps[:64, :3]
            self.log_utils.log_image(imgs, f"reps", round)
            self.log_utils.log_console("Round {} done".format(round))
