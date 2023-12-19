import math
import pdb
import time
from typing import List, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
from algos.algos import synthesize_representations, synthesize_representations_collaborative
from algos.base_class import BaseClient, BaseServer
from utils.generator import Generator
from utils.modules import DeepInversionHook, KLDiv, kldiv, total_variation_loss, reptile_grad, fomaml_grad, reset_l0, reset_bn, put_on_cpu
from torch.utils.data import TensorDataset, DataLoader
from utils.data_utils import CustomDataset
from torchvision import transforms
from kornia import augmentation
# from torchvision.utils import make_grid, save_image
# from algos.fast_meta import FastMetaSynthesizer
from PIL import Image
import torchvision.transforms as T
from algos.fedadam import FedadamOptimizer
import itertools

## change g_step-setup for 10, after aug input same to each model, ismaml, 
# reset, adv=10/1.22-apative distill, 
# check optimizers -> use adam
## done: fedadam for self.generator


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)
    
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
    GENERATOR_UPDATES = 4 # Used by client to send the updated genertor model state
    GENERATOR_DONE = 3 # Used by server to send the updates genertor model state

class FedDreamClient(BaseClient):
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config
        self.set_algo_params()

    def set_algo_params(self):
        self.epochs = self.config["epochs"]
        self.position = self.config["position"]
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
        self.fast_meta = self.config["fast_meta"]
        self.EPS = 1e-8
        self.log_tb_freq = self.config["log_tb_freq"]
        self.log_console =  self.config["log_console"]
        self.transform = self.dset_obj.train_transform
        self.synth_dset = CustomDataset(self.config, transform = None, start_size=10)
        self.kl_loss_fn = KLDiv(T=20)
        self.hooks = []
        self.bn_mmt = None
        if self.fast_meta:
            self.bn_mmt = self.config["bn_mmt"]
            self.lr_z = self.config["lr_z"]
            self.lr_g = self.config["lr_g"]
            self.g_steps = self.config["g_steps"]
            self.adv = self.config["adv"]
            self.bn = self.config["bn"]
            self.oh = self.config["oh"]
            self.gen_warmup = self.config["fast_gen_warmup"]
            self.generator = Generator(nz=256, ngf=64, img_size=32, nc=3).to(self.device)
            normalizer = Normalizer(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010 ))
            self.aug = transforms.Compose([ 
                    augmentation.RandomCrop(size=[32, 32], padding=4),
                    augmentation.RandomHorizontalFlip(),
                    normalizer,
                ])
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(module, self.bn_mmt))           

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
            # self.model_utils.save_model(self.model, self.config["saved_models"] + f"user{self.node_id}.pt")
    
    def generate_rep(self, reps: torch.Tensor):
        for l_step in range(self.config["local_steps"]):
            inputs = reps.clone().detach().requires_grad_(True)
            self.model.zero_grad()
            # acts = self.model(inputs, position=self.position)
            acts = self.model(inputs)
            probs = torch.softmax(acts, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + self.EPS), dim=1).mean()
            loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(self.hooks) if hasattr(model, "r_feature")])
            loss = self.alpha_preds * entropy + self.alpha_tv * total_variation_loss(inputs).to(entropy.device) +\
                    self.alpha_l2 * torch.linalg.norm(inputs).to(entropy.device) + self.alpha_f * loss_r_feature
            loss.backward()
        return inputs.grad
        
    def fast_synthesize(self, reps):
        self.model.eval()
        self.z = reps.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        inputs = self.generator(self.z)
        inputs = self.aug(inputs) # crop and normalize
        t_out = self.model(inputs)
        # loss_oh = F.cross_entropy( t_out, targets )
        probs = torch.softmax(t_out, dim=1)
        entropy = -torch.sum(probs * torch.log(probs  + self.EPS), dim=1).mean()
        loss_bn = sum([model.r_feature for (idx, model) in enumerate(self.hooks) if hasattr(model, "r_feature")])
        loss = self.oh * entropy + self.bn * loss_bn
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return self.z - reps.clone().detach()
        return self.z.grad, loss_bn
        # return self.z

    def update_local_model(self):
        tr_loss, tr_acc, student_loss, student_acc = None, None, None, None
        for _ in range(5):
            tr_loss, tr_acc = self.model_utils.train(self.model,
                                                    self.optim,
                                                    self.dloader,
                                                    self.loss_fn,
                                                    self.device)
        print("train_loss: {}, train_acc: {}".format(tr_loss, tr_acc))
        if self.round % self.local_train_freq == 0:
            synth_dloader = DataLoader(self.synth_dset, batch_size=256, shuffle=True)
            for epoch in range(self.config["distill_epochs"]):
                # Train the model using the selected representations
                student_loss, student_acc = self.model_utils.train(self.model,
                                                                    self.optim,
                                                                    synth_dloader,
                                                                    self.kl_loss_fn,
                                                                    self.device,
                                                                    apply_softmax=True,
                                                                    position=self.position,
                                                                    extra_batch=True)
            # self.synth_dset.reset()
            print("student_loss: {}, student_acc: {} at client {}".format(student_loss, student_acc, self.node_id))
        if self.round % self.log_tb_freq == 0:
            test_loss, test_acc = self.model_utils.test(self.model,
                                                        self._test_loader,
                                                        self.loss_fn,
                                                        self.device)
            print("test_loss: {}, test_acc: {}".format(test_loss, test_acc))
            self.current_stats = [student_loss, student_acc,
                                tr_loss, tr_acc,
                                test_loss, test_acc]
        return

    def add_dreams(self):
        # Wait for the server to send the latest representations
        reps = self.comm_utils.wait_for_signal(src = self.server_node,
                                                tag = self.tag.START_DISTILL)
        reps = reps.to(self.device)
        # send the output of the last layer to the server
        # out = self.model(reps, position=self.position).to("cpu")
        out = self.model(reps).to("cpu")
        self.comm_utils.send_signal(dest=self.server_node,
                                    data=out,
                                    tag=self.tag.FINAL_REPS)
        x, y = self.comm_utils.wait_for_signal(src=self.server_node,
                                                tag=self.tag.FINAL_GLOBAL_REPS)
        # self.utils.logger.log_image(rep, f"client{self.node_id-1}", epoch)
        # self.log_utils.log_console("Round {} done".format(round))
        self.synth_dset.append((x, y))

    def single_round(self):
        for g_step in range(self.global_steps):
            # Wait for the server to send the latest representations
            reps = self.comm_utils.wait_for_signal(src=self.server_node,
                                                    tag=self.tag.START_GEN_REPS)
            reps = reps.to(self.device)
            grads = self.generate_rep(reps)
            # Send the grads to the server
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=grads.to("cpu"),
                                        tag=self.tag.REPS_DONE)
            if g_step == self.global_steps/2 -1:
                self.add_dreams()

    def single_round_fast(self):
        self.model.eval()
        for g_step in range(self.g_steps):
            # Wait for the server to send the latest representations
            gen_state = self.comm_utils.wait_for_signal(src=self.server_node,
                                                        tag=self.tag.GENERATOR_UPDATES)
            self.generator.load_state_dict(gen_state)
            reps = self.comm_utils.wait_for_signal(src=self.server_node,
                                                    tag=self.tag.START_GEN_REPS)
            reps = reps.to(self.device)
            if g_step==0:
                self.z = reps.clone().detach().requires_grad_(True)
                self.optimizer = torch.optim.Adam([
                    {'params': self.generator.parameters()},
                    # {'params': [self.z], 'lr': self.lr_z}
                ], lr=self.lr_g, betas=[0.5, 0.999])
            grads, loss_bn = self.fast_synthesize(reps)
            # Send the grads to the server
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=grads.to("cpu"),
                                        tag=self.tag.REPS_DONE)
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=loss_bn.to("cpu"),
                                        tag=self.tag.REPS_DONE)
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=put_on_cpu(self.generator.state_dict()),
                                        tag=self.tag.GENERATOR_DONE)      
        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()  
            
    def run_protocol(self):
        # Wait for the server to signal to start local warmup rounds
        self.comm_utils.wait_for_signal(src=0, tag=self.tag.START_WARMUP)
        # if self.log_console:
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
        # # Signal to the server that the local warmup rounds are done
        self.comm_utils.send_signal(dest=self.server_node,
                                    data=None,
                                    tag=self.tag.DONE_WARMUP)
        start_epochs = self.config.get("start_epochs", 0)
        for round in range(start_epochs, self.epochs):
            self.round = round
            for i in range(2):
                if self.fast_meta:
                    self.single_round_fast()
                else:
                    self.single_round()
                self.add_dreams()
            self.update_local_model()
            if self.round % self.log_tb_freq == 0:
                self.comm_utils.send_signal(dest=self.server_node,
                                            data=self.current_stats,
                                            tag=self.tag.CLIENT_STATS)
                # save the model if the test loss is lower than the best loss
                if self.current_stats[5] < self.best_loss:
                    self.best_loss = self.current_stats[5]
                    # self.model_utils.save_model(self.model, self.config["saved_models"] + f"user{self.node_id}.pt")
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
        self.distill_batch_size = self.config["distill_batch_size"]
        self.config["inp_shape"][0] = self.distill_batch_size
        self.log_tb_freq = self.config["log_tb_freq"]
        self.log_console =  self.config["log_console"]
        self.adaptive_distill = self.config.get("adaptive_server", False)
        self.fast_meta = self.config["fast_meta"]
        if self.adaptive_distill:
            self.distill_epochs = self.config["distill_epochs"]
            self.adaptive_distill_start_round = self.config["adaptive_distill_start_round"]
            self.EPS = 1e-8
            self.lambda_server = self.config["lambda_server"]
            # set up the server model
            self.set_model_parameters(config)
            self.kl_loss_fn = KLDiv(T=20)
            self.transform = self.dset_obj.train_transform
            self.dset = CustomDataset(self.config, transform = None, start_size=10)
            test_dset = self.dset_obj.test_dset
            self._test_loader = DataLoader(test_dset, batch_size=self.config["batch_size"], shuffle=False)
            self.eval_loss_fn = nn.CrossEntropyLoss()
            self.local_train_freq = self.config["local_train_freq"]
        if self.fast_meta:
            self.ep = 0
            self.gen_warmup = self.config["fast_gen_warmup"]
            self.adv = self.config["adv"]
            self.g_steps = self.config["g_steps"]
            self.lr_g = self.config["lr_g"]
            self.ismaml = self.config["ismaml"]
            self.reset_l0 = reset_l0
            self.nz = 256
            normalizer = Normalizer(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010 ))
            self.aug = transforms.Compose([ 
                    augmentation.RandomCrop(size=[32, 32], padding=4),
                    augmentation.RandomHorizontalFlip(),
                    normalizer,
                ])
            self.meta_generator = Generator(nz=self.nz, ngf=64, img_size=32, nc=3).to(self.device)
            self.generator = Generator(nz=self.nz, ngf=64, img_size=32, nc=3).to(self.device)
            self.meta_optimizer = torch.optim.Adam(self.meta_generator.parameters(), self.lr_g*self.g_steps, betas=[0.5, 0.999])
    
    def aggregate(self, model_wts, probs=None):
        """
        Aggregaste the model weights
        """
        # avg_wts = self.fed_avg(model_wts)
        # All models are sampled currently at every round
        # Each model is assumed to have equal amount of data and hence
        # coeff is same for everyone
        num_clients = len(model_wts)
        # coeff = 1 / num_clients
        avgd_wts = OrderedDict()
        first_model = model_wts[0]

        for client_num in range(num_clients):
            local_wts = model_wts[client_num]
            for key in first_model.keys():
                coeff = probs[client_num] if probs is not None else 1 / num_clients
                if client_num == 0:
                    avgd_wts[key] = coeff * local_wts[key]
                else:
                    avgd_wts[key] += coeff * local_wts[key]
        return avgd_wts
    
    def _aggregate(self, model_wts):
        # All models are sampled currently at every round
        # Each model is assumed to have equal amount of data and hence
        # coeff is same for everyone
        num_clients = len(model_wts)
        coeff = 1 / num_clients
        check_if=lambda name: 'num_batches_tracked' in name
        # accumulate weights
        for client_num in range(num_clients):
            local_wts = model_wts[client_num]
            itr = itertools.chain.from_iterable([self.generator.named_parameters(), self.generator.named_buffers()])
            for name, server_param in itr:
                if check_if(name):
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                local_delta = (server_param - local_wts[name].to(self.device)).mul(coeff).data.type(server_param.dtype)
                if server_param.grad is None: # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.data.add_(local_delta)
        return 
    
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

        # # ---- FedAdam ----
        # lr = self.config["lr_z"]
        # betas = self.data_optimizer.param_groups[0]['betas']
        # beta_1 = 0.9
        # beta_2 = 0.99
        # tau = 1e-9
        # state = self.data_optimizer.state[self.reps]
        # if "m" not in state:
        #     state["m"] = torch.zeros_like(self.reps.data)
        # if "v" not in state:
        #     state["v"] = torch.zeros_like(self.reps.data)
        # state["m"] = beta_1 * state["m"] + (1 - beta_1) * grad
        # state["v"] = beta_2 * state["v"] + (1 - beta_2) * grad**2
        # update = self.reps.data + lr * state["m"] / (state["v"] ** 0.5 + tau)
        # update = update.detach()
        # self.reps.data = update

    def update_stats(self, stats: List[List[float]]):
        """
        Updates the statistics from all the clients
        the reason the server is doing it because
        the server only has the access to the log file as of now
        Args:
            stats (List[List[float]]): List of statistics of the clients
        """
        labels = ["student_loss", "student_acc", "train_loss", "train_acc", "test_loss", "test_acc"]
        for client, stat in enumerate(stats):
            if stat is not None:
                # for i in range(len(stat)):
                for i in [1,3,5]:
                    if stat[i] is not None:
                        self.log_utils.log_tb(f"{labels[i]}/client{client}", stat[i], self.round)
                if self.log_console:
                    self.log_utils.log_console(f"Round {self.round} test accuracy for client {client}: {stat[5]}")

    def update_server_model(self):
        self.dloader = DataLoader(self.dset, batch_size=256, shuffle=True)
        # the choice of 20 here is rather arbitrary
        tr_loss, tr_acc = 0., 0.
        distill_epochs = self.distill_epochs
        for _ in range(distill_epochs):
            tr_loss, tr_acc = self.model_utils.train(self.model, self.optim, self.dloader, self.kl_loss_fn, 
                                                     self.device, apply_softmax=True, extra_batch=True)
        if self.round % self.log_tb_freq == 0:
            te_loss, te_acc = self.model_utils.test(self.model, self._test_loader, self.eval_loss_fn, self.device)
            # self.log_utils.log_tb("train_loss", tr_loss, self.round)
            self.log_utils.log_tb("train_acc", tr_acc, self.round)
            # self.log_utils.log_tb("test_loss", te_loss, self.round)
            self.log_utils.log_tb("test_acc", te_acc, self.round)
            if self.log_console:
                self.log_utils.log_console(f"Round {round} Server training done")
                self.log_utils.log_console(f"Round {round} train_loss: {tr_loss}, train_acc: {tr_acc}, test_loss: {te_loss}, test_acc: {te_acc}")
                if te_acc > self.best_acc:
                    self.best_acc = te_acc
                    # save the model
                    # self.model_utils.save_model(self.model, self.config["saved_models"] + "server_model.pt")
            
    def add_dreams(self):
        reps = self.reps
        imgs = (reps.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
        # performance depends on model output of the transformed input
        for i in range(len(imgs)):
            img = Image.fromarray(imgs[i].transpose(1, 2, 0))
            inp = self.transform(img)
            self.reps[i] = inp
        self.reps = self.reps.detach()   
        for client in self.clients:
            self.comm_utils.send_signal(dest=client, data=self.reps.to("cpu"), tag=self.tag.START_DISTILL)
        acts = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.FINAL_REPS)
        acts = torch.stack(acts)
        acts = acts.mean(dim=0)
        acts = acts.detach()
        # acts = torch.log_softmax(acts, dim=1)
        for client in self.clients:
            self.comm_utils.send_signal(dest=client,
                                        data=(self.reps.to("cpu"), acts.to("cpu")),
                                        tag=self.tag.FINAL_GLOBAL_REPS)                   
        if self.adaptive_distill:
            self.dset.append((self.reps, acts))
        if self.log_console:
                self.log_utils.log_tensor_to_disk((self.reps, acts), f"node", self.round)
                # Only store first three channel and 64 images for a 8x8 grid
                imgs = reps[:64, :3]
                self.log_utils.log_image(imgs, f"reps", self.round)

    def single_round(self):
        if self.log_console:
            self.log_utils.log_console("Starting round {}".format(round))
        global_steps = self.config["global_steps"]
        self.reps = torch.randn(self.config["inp_shape"]).to(self.device)
        self.data_optimizer = torch.optim.Adam([self.reps], lr=self.config["data_lr"])
        # TODO: choice of 20 is arbitrary so need to experimentally arrive at a better number
        for g_step in range(global_steps):
            for client in self.clients:
                self.comm_utils.send_signal(dest=client, data=self.reps.to("cpu"), tag=self.tag.START_GEN_REPS)
            grads = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
            grads = torch.stack(grads).to(self.device)
            grads = grads.mean(dim=0)
            if self.round > self.adaptive_distill_start_round and self.adaptive_distill:
                # pass reps on the local model and get the gradients
                inputs = self.reps.clone().detach().requires_grad_(True)
                acts = self.model(inputs)
                probs = torch.softmax(acts, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + self.EPS), dim=1).mean()
                loss = entropy
                loss.backward()
                server_grads = inputs.grad
                # negative gradient update to maximize entropy of the server model
                grads = grads - self.lambda_server * server_grads
            self.adam_update(grads)
            if self.log_console:
                if g_step % 500 == 0 or g_step== (global_steps - 1):
                    print(f"{g_step}/{global_steps}", time.time() - start)
            if g_step == global_steps/2 -1:    
                self.add_dreams()
        return
    
    def single_round_fast(self):
        start = time.time()
        self.reps = torch.randn(size=(self.distill_batch_size, self.nz), device=self.device).requires_grad_()
        self.data_optimizer = torch.optim.Adam([self.reps], lr=self.config["lr_z"], betas=[0.5, 0.999])
        self.generator.load_state_dict(self.meta_generator.state_dict())
        self.ep += 1
        if (self.ep == 120+self.gen_warmup) and self.reset_l0:
            reset_l0(self.generator)
        for it in range(self.g_steps):
            for client in self.clients:
                self.comm_utils.send_signal(dest=client, data=put_on_cpu(self.generator.state_dict()), 
                                            tag=self.tag.GENERATOR_UPDATES)
                self.comm_utils.send_signal(dest=client, data=self.reps.to("cpu"), tag=self.tag.START_GEN_REPS)
            grads = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
            loss_bn = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
            # coeff = [(1/x)/sum([1/x for x in loss_bn]) for x in loss_bn]
            # grads = sum([g*coeff[i] for i, g in enumerate(grads)]).to(self.device)
            grads = torch.stack(grads).to(self.device)
            grads = grads.mean(dim=0) 
            if self.adaptive_distill and self.round > self.adaptive_distill_start_round:
                # pass reps on the local model and get the gradients
                z_server = self.reps.clone().detach().requires_grad_(True)
                inputs = self.generator(z_server)
                inputs = self.aug(inputs) # crop and normalize
                acts = self.model(inputs)
                probs = torch.softmax(acts, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + self.EPS), dim=1).mean()
                loss = entropy
                loss.backward()
                server_grads = z_server.grad
                # negative gradient update to maximize entropy of the server model
                grads = grads - self.adv * server_grads
            self.adam_update(grads)
            
            # gen_state_dict = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.GENERATOR_DONE)
            # self.generator_optimizer = FedadamOptimizer(params=self.generator.parameters(), lr=self.lr_g, betas=[0.5, 0.999])
            # self.generator_optimizer.add_param_group(dict(params=list(self.generator.buffers())))
            # self.generator_optimizer.zero_grad(set_to_none=True)
            # self._aggregate(gen_state_dict)
            # self.generator_optimizer.step()

            gen_state_dict = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.GENERATOR_DONE)
            avg_gen_state_dict = self.aggregate(gen_state_dict)
            # avg_gen_state_dict = self.aggregate(gen_state_dict, coeff)
            self.generator.load_state_dict(avg_gen_state_dict)
            self.generator = self.generator.to(self.device)
            if self.ismaml:
                if it==0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.meta_generator, self.generator, self.device)
                if it == (self.g_steps-1): self.meta_optimizer.step()
        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.meta_generator, self.generator, self.device)
            self.meta_optimizer.step()
        self.reps = self.generator(self.reps)
        return

    def run_protocol(self):
        if self.log_console:
            self.log_utils.log_console("Starting Server")
        # Get all the students to start local warmup rounds
        for client in self.clients:
            self.comm_utils.send_signal(dest=client, data=None, tag=self.tag.START_WARMUP)
        # Wait for the students to finish warmup rounds
        if self.log_console:
            self.log_utils.log_console("Waiting for students to finish warmup rounds")
        self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.DONE_WARMUP)
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.round = round
            start = time.time()
            for i in range(2):
                if self.fast_meta:
                    self.single_round_fast()
                else:
                    self.single_round()
                self.add_dreams()
            end = time.time()
            if self.log_console:
                self.log_utils.log_console(f"Time taken: {end - start} seconds")
            # if self.adaptive_distill and round >= self.gen_warmup and round % self.local_train_freq == 0:
            if self.adaptive_distill and round>=10 and round % self.local_train_freq == 0:
                self.update_server_model()
            if round % self.log_tb_freq == 0:
                stats = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.CLIENT_STATS)
                self.update_stats(stats)
            if self.log_console:
                self.log_utils.log_console("Round {} done".format(round))