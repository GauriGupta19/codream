import time
from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from algos.base_class import BaseClient, BaseServer
from utils.generator import Generator
from utils.di_hook import DeepInversionHook, get_weighted_bn_loss
from utils.modules import KLDiv, kldiv, reptile_grad, fomaml_grad, reset_l0, reset_bn, put_on_cpu, \
                            get_model_grads, aggregate_model_grads, adam_update,  adam_step, aggregate_models
from torch.utils.data import DataLoader
from utils.data_utils import CustomDataset
from PIL import Image


## change g_step-setup for 10, after aug input same to each model, ismaml, 
# reset, adv=10/1.22-apative distill, 
# check optimizers -> use adam
## done: fedadam for self.generator

#s_model
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
    STUDENT_UPDATES = 2

class FedDreamFastNoniidClient(BaseClient):
    def __init__(self, config):
        super().__init__(config)
        self.tag = CommProtocol()
        self.config = config
        self.set_algo_params()

    def set_algo_params(self):
        self.epochs = self.config["epochs"]
        self.position = self.config["position"]
        self.distill_batch_size = self.config["distill_batch_size"]
        self.distill_epochs = self.config["distill_epochs"]
        self.warmup = self.config["warmup"]
        self.local_train_freq = self.config["local_train_freq"]
        self.EPS = 1e-8
        self.log_tb_freq = self.config["log_tb_freq"]
        self.log_console =  self.config["log_console"]
        self.dset_size = self.config["dset_size"]
        self.synth_dset = CustomDataset(self.config, transform = None, buffer_size=self.dset_size)
        self.kl_loss_fn = KLDiv(T=20)
        self.hooks = []
        self.bn_mmt = 0.9
        self.lr_z = self.config["lr_z"]
        self.lr_g = self.config["lr_g"]
        self.global_steps = self.config["global_steps"]
        self.local_steps = self.config["local_steps"]
        self.optimizer_type = self.config["optimizer_type"]
        self.nx_samples = self.config["nx_samples"] #number of batches of dreams generated
        self.adv = self.config["adv"]
        self.bn = self.config["bn"]
        self.oh = self.config["oh"]
        self.nz = 256
        self.generator = Generator(nz=self.nz, ngf=64, img_size=self.config["inp_shape"][-1], nc=self.config["inp_shape"][1]).to(self.device)
        self.aug = self.dset_obj.gen_transform
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
            self.model_utils.save_model(self.model, self.config["saved_models"] + f"user{self.node_id}.pt")
        
    def train(self, model:nn.Module, optim, batch1, batch2, loss_fn, device: torch.device, **kwargs):
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        total_samples = 0
        # for batch_idx, (data, target) in enumerate(dloader):
        data1, target1 = batch1
        data1, target1 = data1.to(device), target1.to(device)
        
        data2, target2 = batch2
        data2, target2 = data2.to(device), target2.to(device)
        data2 = data2.view(data2.size(0) * data2.size(1), *data2.size()[2:])
        target2 = target2.view(target2.size(0) * target2.size(1), *target2.size()[2:])
        
        total_samples += data1.size(0)
        total_samples += data2.size(0)
        optim.zero_grad()
        
        output1 = model(data1)
        output2 = model(data2)

        if kwargs.get("apply_softmax", False):
            output = nn.functional.log_softmax(output, dim=1) # type: ignore
        if len(target1.size()) > 1 and target1.size(1) == 1:
            target1 = target1.squeeze(dim=1)
        if len(target2.size()) > 1 and target2.size(1) == 1:
            target2 = target2.squeeze(dim=1)

        loss1 = self.loss_fn(output1, target1)
        loss2 = self.kl_loss_fn(output2, target2)
        loss = loss1+loss2

        loss.backward()
        optim.step()
        train_loss += loss.item()
        pred1 = output1.argmax(dim=1, keepdim=True)
        # view_as() is used to make sure the shape of pred and target are the same
        if len(target1.size()) > 1:
            target1 = target1.argmax(dim=1, keepdim=True)
        correct += pred1.eq(target1.view_as(pred1)).sum().item()
        pred2 = output2.argmax(dim=1, keepdim=True)
        # view_as() is used to make sure the shape of pred and target are the same
        if len(target2.size()) > 1:
            target2 = target2.argmax(dim=1, keepdim=True)
        correct += pred2.eq(target2.view_as(pred2)).sum().item()
        acc = correct / total_samples
        return train_loss, acc
    
    def update_local_model(self):
        tr_loss, tr_acc, student_loss, student_acc = None, None, None, None
        synth_dloader = DataLoader(self.synth_dset, batch_size=256, shuffle=True)
        if self.round % self.local_train_freq == 0:
            for i in range(len(synth_dloader)):
                batch1 = next(iter(self.dloader))
                batch2 = next(iter(synth_dloader))
                for epoch in range(self.config["distill_epochs"]):
                    tr_loss, tr_acc = self.train(self.model,
                                                self.optim,
                                                batch1,
                                                batch2,
                                                self.loss_fn,
                                                self.device)
        # if ((self.node_id==1 or self.node_id==4) and self.round<=60):
        #     for _ in range(5):
        #         tr_loss, tr_acc = self.model_utils.train(self.model,
        #                                                 self.optim,
        #                                                 self.dloader,
        #                                                 self.loss_fn,
        #                                                 self.device)
        # if ((self.node_id==2 or self.node_id==3)):
        #     for _ in range(5):
        #         tr_loss, tr_acc = self.model_utils.train(self.model,
        #                                                 self.optim,
        #                                                 self.dloader,
        #                                                 self.loss_fn,
        #                                                 self.device)
        # print("train_loss: {}, train_acc: {}".format(tr_loss, tr_acc))
        # if self.round % self.local_train_freq == 0:
        #     synth_dloader = DataLoader(self.synth_dset, batch_size=256, shuffle=True)
        #     for epoch in range(self.config["distill_epochs"]):
        #     # for epoch in range(20):
        #         # Train the model using the selected representations
        #         student_loss, student_acc = self.model_utils.train(self.model,
        #                                                             self.optim,
        #                                                             synth_dloader,
        #                                                             self.kl_loss_fn,
        #                                                             self.device,
        #                                                             apply_softmax=True,
        #                                                             position=self.position,
        #                                                             extra_batch=True)
        #     # self.synth_dset.reset()
        #     print("student_loss: {}, student_acc: {} at client {}".format(student_loss, student_acc, self.node_id))
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

    def add_dreams(self, targets):
        # Wait for the server to send the latest representations
        reps = self.comm_utils.wait_for_signal(src = self.server_node,
                                                tag = self.tag.START_DISTILL)
        reps = reps.to(self.device)
        # send the output of the last layer to the server
        out = self.model(reps).detach().to("cpu")
        # loss_bn = sum([model.r_feature for (idx, model) in enumerate(self.hooks) if hasattr(model, "r_feature")]).to("cpu")
        self.comm_utils.send_signal(dest=self.server_node,
                                    data=out,
                                    tag=self.tag.FINAL_REPS)
        # self.comm_utils.send_signal(dest=self.server_node,
        #                             data=loss_bn,
        #                             tag=self.tag.FINAL_REPS)
        x, y = self.comm_utils.wait_for_signal(src=self.server_node,
                                                tag=self.tag.FINAL_GLOBAL_REPS)
        # self.utils.logger.log_image(rep, f"client{self.node_id-1}", epoch)
        # self.log_utils.log_console("Round {} done".format(round))
        self.synth_dset.append((x, y))
        # self.synth_dset.append((x, targets))

    def fast_synthesize(self, reps, targets, weights):
        self.model.eval()
        self.z.data = reps.clone().detach().requires_grad_(True)
        generator_grads = None
        for _ in range(self.local_steps):
            self.model.zero_grad()
            inputs = self.generator(self.z)
            inputs = self.aug(inputs) # crop and normalize
            t_out = self.model(inputs)
            label_weights = torch.tensor(self.label_weights, dtype=torch.float32).to(self.device)
            loss_oh = F.cross_entropy(t_out, targets, weight = label_weights)
            # loss_oh = F.cross_entropy(t_out, targets)
            # probs = torch.softmax(t_out, dim=1)
            # entropy = -torch.sum(probs * torch.log(probs  + self.EPS), dim=1).mean()
            loss_bn = sum([model.r_feature for (idx, model) in enumerate(self.hooks) if hasattr(model, "r_feature")])
            loss_adv = loss_oh.new_zeros(1)
            if self.adv>0 and (self.round >= 15):
                s_out = self.s_model(inputs)
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                mask = mask * weights.view(-1,)
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # decision adversarial distillation
            loss = self.oh * loss_oh + self.bn * loss_bn + self.adv * loss_adv
            self.optimizer.zero_grad()
            loss.backward()
            generator_grads = get_model_grads(self.generator)
            # aggregate_model_grads([generator_grads], self.generator, self.device)
            # adam_step(self.optimizer)
            self.optimizer.step()
        # if self.optimizer_type=="avg":
        #     return self.z
        # elif self.optimizer_type=="adam":
        #     return self.z.grad
        # elif self.optimizer_type=="fedadam":
        #     return self.z - reps.clone().detach()
        return self.z, None
        # return self.z.grad, generator_grads
    
    def single_round_fast(self):
        self.model.eval()
        targets, weights = None, None
        for g_step in range(self.global_steps):
            if g_step==0:
                targets = self.comm_utils.wait_for_signal(src=self.server_node,
                                                    tag=self.tag.START_GEN_REPS)
                weights = self.label_weights[targets.numpy()].reshape(-1, 1)
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                targets = targets.to(self.device)
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
                    {'params': [self.z], 'lr': self.lr_z}
                ], lr=self.lr_g, betas=[0.5, 0.999])
            z_grads, generator_grads = self.fast_synthesize(reps, targets, weights)
            # Send the grads to the server
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=z_grads.to("cpu"),
                                        tag=self.tag.REPS_DONE)
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=generator_grads,
                                        tag=self.tag.REPS_DONE)
            self.comm_utils.send_signal(dest=self.server_node,
                                        data=put_on_cpu(self.generator.state_dict()),
                                        tag=self.tag.GENERATOR_DONE)      
        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()  
        self.add_dreams(targets)
    
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
        self.comm_utils.send_signal(dest=self.server_node,
                                    data=self.class_counts,
                                    tag=self.tag.DONE_WARMUP)
        self.label_weights = self.comm_utils.wait_for_signal(src=self.server_node,
                                                  tag=self.tag.DONE_WARMUP)
        # self.label_weights = np.array(self.class_counts) / (np.sum(self.class_counts))   
        start_epochs = self.config.get("start_epochs", 0)
        self.s_model = self.model_utils.get_model("resnet18", self.config["dset"], 
                                   self.device, self.device_ids, num_classes=10)
        for round in range(start_epochs, self.epochs):
            s_state = self.comm_utils.wait_for_signal(src=self.server_node,
                                                        tag=self.tag.STUDENT_UPDATES)
            self.s_model.load_state_dict(s_state)
            self.s_model = self.s_model.to(self.device)
            self.round = round
            for i in range(self.nx_samples):
                self.single_round_fast()
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


class FedDreamFastNoniidServer(BaseServer):
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
        self.MIN_SAMPLES_PER_LABEL = 1
        if self.adaptive_distill:
            self.distill_epochs = self.config["distill_epochs"]
            self.adaptive_distill_start_round = self.config["adaptive_distill_start_round"]
            self.EPS = 1e-8
            # set up the server model
            self.set_model_parameters(config)
            self.kl_loss_fn = KLDiv(T=20)
            self.transform = self.dset_obj.train_transform
            self.dset_size = self.config["dset_size"]
            self.dset = CustomDataset(self.config, transform = None, buffer_size=self.dset_size)
            test_dset = self.dset_obj.test_dset
            self._test_loader = DataLoader(test_dset, batch_size=self.config["batch_size"], shuffle=False)
            self.eval_loss_fn = nn.CrossEntropyLoss()
            self.local_train_freq = self.config["local_train_freq"]
        self.ep = 0
        self.global_steps = self.config["global_steps"]
        self.local_steps = self.config["local_steps"]
        self.nx_samples = self.config["nx_samples"] #number of batches of dreams generated
        self.adv = self.config["adv"]
        self.lr_g = self.config["lr_g"]
        self.ismaml = self.config["ismaml"]
        self.reset_l0 = reset_l0
        self.aug = self.dset_obj.gen_transform
        self.optimizer_type = self.config["optimizer_type"]
        self.nz = 256
        self.meta_generator = Generator(nz=self.nz, ngf=64, img_size=self.config["inp_shape"][-1], nc=self.config["inp_shape"][1]).to(self.device)
        self.generator = Generator(nz=self.nz, ngf=64, img_size=self.config["inp_shape"][-1], nc=self.config["inp_shape"][1]).to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.meta_generator.parameters(), 
                                               self.lr_g*self.local_steps*self.global_steps, betas=[0.5, 0.999])
    
    def get_label_weights(self):
        self.clients_class_counts = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.DONE_WARMUP)
        self.unique_labels = len(self.clients_class_counts[0])
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for client_count in self.clients_class_counts:
                weights.append(client_count[label])
            if np.sum(weights) > self.MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / (np.sum(weights) + self.EPS))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels
    
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
        tr_loss, tr_acc = 0., 0.
        distill_epochs = self.distill_epochs
        for _ in range(distill_epochs):
            tr_loss, tr_acc = self.model_utils.train(self.model, self.optim, self.dloader, self.kl_loss_fn, 
                                                     self.device, apply_softmax=True, extra_batch=True)
        # self.dset.reset()
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
        # weighted logit from the clients ensemble
        t_logit = 0
        for client, act in enumerate(acts):
            weight=self.label_weights[self.targets][:, client].reshape(-1, 1)
            expand_weight=np.tile(weight, (1, self.unique_labels))
            t_logit += act * torch.tensor(expand_weight, dtype=torch.float32)
        acts = t_logit.to(self.device)
        acts = acts.detach()
        # acts = torch.stack(acts)
        # acts = acts.mean(dim=0)
        # acts = acts.detach()
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
    
    def single_round_fast(self):
        start = time.time()
        self.ep += 1
        self.reps = torch.randn(size=(self.distill_batch_size, self.nz), device=self.device).requires_grad_()
        # self.targets = torch.randint(low=0, high=self.dset_obj.NUM_CLS, size=(self.distill_batch_size,))
        self.targets = np.random.choice(self.qualified_labels, self.distill_batch_size)
        self.targets = torch.LongTensor(self.targets)
        self.generator.load_state_dict(self.meta_generator.state_dict())
        if (self.ep == 140) and self.reset_l0:
            reset_l0(self.generator)
        self.data_optimizer = torch.optim.Adam([self.reps], lr=self.config["lr_z"], betas=[0.5, 0.999])
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=[0.5, 0.999])
        for it in range(self.global_steps):
            for client in self.clients:
                if it==0:
                    self.comm_utils.send_signal(dest=client, data=self.targets.to("cpu"), tag=self.tag.START_GEN_REPS)
                self.comm_utils.send_signal(dest=client, data=put_on_cpu(self.generator.state_dict()), 
                                            tag=self.tag.GENERATOR_UPDATES)
                self.comm_utils.send_signal(dest=client, data=self.reps.to("cpu"), tag=self.tag.START_GEN_REPS)
            z_grads = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
            generator_grads = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.REPS_DONE)
            z_grads = torch.stack(z_grads).to(self.device)
            z_grads = z_grads.mean(dim=0)
            
            self.reps = z_grads
            # self.data_optimizer.zero_grad()
            # self.generator_optimizer.zero_grad()
            # self.reps.grad = z_grads
            # aggregate_model_grads(generator_grads, self.generator, self.device)
            # self.data_optimizer.step()
            # self.generator_optimizer.step()
            # adam_step(self.data_optimizer)
            # adam_step(self.generator_optimizer)
            # if self.optimizer_type=="avg":
            #     self.reps = grads
            # if self.optimizer_type=="adam":
            #     self.adam_update(grads)
            gen_state_dict = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.GENERATOR_DONE)
            avg_gen_state_dict = aggregate_models(gen_state_dict)
            self.generator.load_state_dict(avg_gen_state_dict)
            if self.ismaml:
                if it==0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.meta_generator, self.generator, self.device)
                if it == (self.global_steps-1): self.meta_optimizer.step()
        self.reps = self.generator(self.reps)
        self.add_dreams()
        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.meta_generator, self.generator, self.device)
            self.meta_optimizer.step()
        end = time.time()
        if self.log_console:
            self.log_utils.log_console(f"Time taken: {end - start} seconds")
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
        self.label_weights, self.qualified_labels = self.get_label_weights()
        for client in self.clients:
            self.comm_utils.send_signal(dest=client, data=self.label_weights[:, client-1], tag=self.tag.DONE_WARMUP)
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.round = round
            for client in self.clients:
                self.comm_utils.send_signal(dest=client, data=put_on_cpu(self.model.state_dict()), 
                                            tag=self.tag.STUDENT_UPDATES)
            for i in range(self.nx_samples):
                self.single_round_fast()
            if self.adaptive_distill and round>=10 and round % self.local_train_freq == 0:
                self.update_server_model()
            if round % self.log_tb_freq == 0:
                stats = self.comm_utils.wait_for_all_clients(self.clients, tag=self.tag.CLIENT_STATS)
                self.update_stats(stats)
            if self.log_console:
                self.log_utils.log_console("Round {} done".format(round))