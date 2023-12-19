import math
import pdb
import time
import torch.nn as nn
import random, torch
from torch.autograd import Variable
from utils.modules import DeepInversionHook, total_variation_loss
from utils.model_utils import ModelUtils
from torchvision import utils
from utils.generator import Generator
from torchvision import transforms


EPS = 1e-8


class GradFn(nn.Module):
    def __init__(self, model, loss_r_feature_layers, position,
                 alpha_preds=1., alpha_tv=1., alpha_l2=1., alpha_f=1., id_=0):
        super(GradFn, self).__init__()
        self.model = model
        self.loss_r_feature_layers = loss_r_feature_layers
        self.position = position
        self.alpha_preds = alpha_preds
        self.alpha_tv = alpha_tv
        self.alpha_l2 = alpha_l2
        self.alpha_f = alpha_f
        self.id = id_
        self.num = 0

    def forward(self, inputs, position, off1, off2):
        steps = 1
        inputs.retain_grad()
        for i in range(steps):
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))
            self.model.zero_grad()
            acts = self.model(inputs_jit, position=self.position)
            probs = torch.softmax(acts, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1).mean()
            loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(self.loss_r_feature_layers) if hasattr(model, "r_feature")])
            loss = self.alpha_preds * entropy + self.alpha_tv * total_variation_loss(inputs_jit).to(entropy.device) +\
                self.alpha_l2 * torch.linalg.norm(inputs_jit).to(entropy.device) + self.alpha_f * loss_r_feature
            loss.backward()
            self.num += 1
        return inputs.grad.unsqueeze(dim=0)


def adam_update(optimizer, updated_img, grad):
    betas = optimizer.param_groups[0]['betas']
    # access the optimizer's state
    state = optimizer.state[updated_img]
    if 'exp_avg' not in state:
        state['exp_avg'] = torch.zeros_like(updated_img.data)
    if 'exp_avg_sq' not in state:
        state['exp_avg_sq'] = torch.zeros_like(updated_img.data)
    if 'step' not in state:
        state['step'] = 1
    exp_avg = state['exp_avg']
    exp_avg_sq = state['exp_avg_sq']
    bias_correction1 = 1 - betas[0] ** state['step']
    bias_correction2 = 1 - betas[1] ** state['step']
    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(optimizer.param_groups[0]['eps'])
    step_size = optimizer.param_groups[0]['lr'] / bias_correction1
    updated_img.data.addcdiv_(exp_avg, denom, value=-step_size)
    optimizer.state[updated_img]['step'] += 1


def synthesize_representations_collaborative_parallel(config, obj):
    grad_fns, models = [], []
    lr, steps = config["data_lr"], obj["steps"]
    position = config["position"]
    alpha_preds, alpha_tv, alpha_l2, alpha_f = config["alpha_preds"], config["alpha_tv"], config["alpha_l2"], config["alpha_f"]
    orig_img, main_device = obj["orig_img"], obj["device"]
    for i in range(len(obj["model_wts"])):
        loss_r_feature_layers = []
        device = obj["device_ids"][i]
        model = ModelUtils.get_model(config["model"], config["dset"], device, [device,])
        model.load_state_dict(obj["model_wts"][i])
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionHook(module))
        grad_fn = GradFn(model, loss_r_feature_layers, position,
                         alpha_preds=alpha_preds, alpha_tv=alpha_tv, alpha_l2=alpha_l2, alpha_f=alpha_f,
                         id_=i)
        models.append(model)
        grad_fns.append(grad_fn)

    updated_img = orig_img.clone()

    optimizer = torch.optim.Adam([updated_img], lr=lr, betas=(0.5, 0.9), eps = 1e-8)
    lim_0, lim_1 = 2, 2

    def _agg_grads(grads):
        grads_on_same_device = nn.parallel.scatter_gather.gather(grads, main_device)
        grads = grads_on_same_device
        # TODO: check whether mean is better than sum
        return grads.mean(dim=0)

    # time the following loop
    start = time.time()
    for it in range(steps + 1):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        # replicate inputs to len(grad_fns) with each of them on a different device
        inputs = []
        for i in range(len(grad_fns)):
            updated_img_i = updated_img.clone().detach()
            updated_img_i.requires_grad = True
            updated_img_i.retain_grad()
            device = obj["device_ids"][i]
            inputs.append(updated_img_i.to(device))
        # replicate kwargs to len(grad_fns)
        kwargs = {'position': position, 'off1': off1, 'off2': off2}
        kwargs_tup = (kwargs,) * len(grad_fns)

        optimizer.zero_grad()
        grads = nn.parallel.parallel_apply(grad_fns, inputs, kwargs_tup=kwargs_tup, devices=obj["device_ids"])
        grad = _agg_grads(grads)
        adam_update(optimizer, updated_img, grad)
        if it % 500 == 0 or it==steps:
            print(f"{it}/{steps}")


    for grad_fn in grad_fns:
        for item in grad_fn.loss_r_feature_layers:
            item.close()

    # get activations from each model using parallel_apply for the updated image
    acts = nn.parallel.parallel_apply(models, [updated_img] * len(models), devices=obj["device_ids"])
    # unsqueeze every item in the zeroth dimension
    acts = [act.unsqueeze(0) for act in acts]
    acts = nn.parallel.gather(acts, main_device)
    acts = acts.mean(dim=0)
    acts = torch.log_softmax(acts, dim=1)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return updated_img.detach(), acts.detach()


def synthesize_representations_collaborative(config, obj):
    lr, steps = config["data_lr"], obj["steps"]
    position = config["position"]
    alpha_preds, alpha_tv, alpha_l2, alpha_f = config["alpha_preds"], config["alpha_tv"], config["alpha_l2"], config["alpha_f"]
    orig_img = obj["orig_img"]
    models, loss_r_feature_layers = [], []
    for i in range(len(obj["model_wts"])):
        model = ModelUtils.get_model(config["model"], config["dset"], obj["device"], [obj["device"],])
        model.module.load_state_dict(obj["model_wts"][i])
        model.eval()
        loss_r_feature_layers.append([])
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers[i].append(DeepInversionHook(module))
        models.append(model)
    orig_img = torch.load("orig_img.pt").to(obj["device"])
    updated_img = orig_img.clone()
    updated_img.requires_grad = True
    updated_img.retain_grad()

    optimizer = torch.optim.Adam([updated_img], lr=lr, betas=(0.5, 0.9), eps = 1e-8)
    lim_0, lim_1 = 2, 2
    for it in range(steps + 1):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(updated_img, shifts=(off1, off2), dims=(2,3))
        optimizer.zero_grad()
        entropies = 0.
        loss_r_features = 0.
        for i in range(len(models)):
            model = models[i]
            model.zero_grad()
            acts = model.module(inputs_jit, position=position)
            probs = torch.softmax(acts, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1).mean()
            loss_r_feature = sum([m_.r_feature for (idx, m_) in enumerate(loss_r_feature_layers[i]) if hasattr(m_, "r_feature")])
            loss_r_features += loss_r_feature
            entropies += entropy

        # loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        loss = alpha_preds * entropies + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_features
        if it % 5000 == 0 or it==steps:
            print(f"{it}/{steps}", loss_r_features.item(), entropies.item()) # type: ignore

        loss.backward()
        optimizer.step()
        # grad = updated_img.grad
        # print(grad.mean(), grad.std())
        # updated_img.data = updated_img.data - lr * updated_img.grad.data * (1 / len(models))

    # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
    # making the forward pass expensive and hog up memory
    for model in loss_r_feature_layers:
        for item_ in model:
            item_.close()
    acts = torch.zeros_like(acts)
    for model in models:
        model.zero_grad()
        acts += model.module(updated_img, position=position)
    # normalize the activations
    acts = acts / len(models)
    # apply log softmax
    acts = torch.log_softmax(acts, dim=1)
    return updated_img.detach(), acts.detach()


def synthesize_representations(config, obj):
    """
    Synthesize representations for each class
    """
    lr, steps = config["data_lr"], obj["steps"]
    position = config["position"]
    alpha_preds, alpha_tv, alpha_l2, alpha_f = config["alpha_preds"], config["alpha_tv"], config["alpha_l2"], config["alpha_f"]
    orig_img, target_label, model = obj["orig_img"], obj["target_label"], obj["model"]
    loss_r_feature_layers = []

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionHook(module))

    updated_img = orig_img.clone()
    updated_img.requires_grad = True
    updated_img.retain_grad()

    optimizer = torch.optim.Adam([updated_img], lr=lr, betas=(0.5, 0.9), eps = 1e-8)
    lim_0, lim_1 = 2, 2
    for it in range(steps + 1):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(updated_img, shifts=(off1, off2), dims=(2,3))

        model.zero_grad()
        optimizer.zero_grad()
        acts = model(inputs_jit, position=position)
        # ce_loss = nn.CrossEntropyLoss()(acts, target_label)
        probs = torch.softmax(acts, dim=1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1).mean()
        loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers) if hasattr(model, "r_feature")])
        # loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        loss = alpha_preds * entropy + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        if it % 500 == 0 or it==steps:
            if len(target_label.size()) > 1:
                acc = (acts.argmax(dim=1) == target_label.argmax(dim=1)).sum() / acts.shape[0]
            else:
                acc = (acts.argmax(dim=1) == target_label).sum() / acts.shape[0]
            # acc = (acts.argmax(dim=1) == target_label.argmax(dim=1)).sum() / acts.shape[0]
            print(f"{it}/{steps}", loss_r_feature.item(), acc, entropy.item()) # type: ignore

        loss.backward()
        optimizer.step()


class FastMetaSynthesizer():
    def __init__(self, config, dset_obj, device):
        self.lr_g = config["lr_g"]
        self.lr_z = config["lr_z"]
        self.nz = config["inp_shape"][1]
        self.ismaml = config["ismaml"]
        self.iterations = config["steps"]
        # self.iterations = 20
        self.device = device
        self.out_shape = config["out_shape"]
        self.generator = Generator(nz=self.nz, ngf=64, img_size=self.out_shape[2], nc=self.out_shape[1]).to(self.device)
        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        self.aug = dset_obj.gen_transform
            
    def reptile_grad(self, src, tar):
        for p, tar_p in zip(src.parameters(), tar.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).to(self.device)
            p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40


    def fomaml_grad(self, src, tar):
        for p, tar_p in zip(src.parameters(), tar.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).to(self.device)
            p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67


    def synthesize_representations(self, config, obj):

        """
        Synthesize representations for each class
        """
        steps, position = obj["steps"], config["position"]
        alpha_preds, alpha_tv, alpha_l2, alpha_f = config["alpha_preds"], config["alpha_tv"], config["alpha_l2"], config["alpha_f"]

        orig_img, target_label, model = obj["orig_img"], obj["target_label"], obj["model"]
        loss_r_feature_layers = []
        
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionHook(module))
        
        z = orig_img.clone()
        z.requires_grad = True
        z.retain_grad()

        fast_generator = self.generator.clone().to(self.device)
        optimizer = torch.optim.Adam([
                {'params': fast_generator.parameters()},
                {'params': [z], 'lr': self.lr_z}
            ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(1,steps+1):
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs) # crop and normalize
            acts = model(inputs_aug, position=position)
            # ce_loss = nn.CrossEntropyLoss()(acts, target_label)
            probs = torch.softmax(acts, dim=1)

            entropy = -torch.sum(probs * torch.log(probs), dim=1).mean()
            loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers) if hasattr(model, "r_feature")])
            # loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
            loss = alpha_preds * entropy + alpha_tv * total_variation_loss(inputs) + alpha_l2 * torch.linalg.norm(inputs) + alpha_f * loss_r_feature
            
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            if self.ismaml:
                if it==0: self.meta_optimizer.zero_grad()
                self.fomaml_grad(self.generator, fast_generator)
                if it == (steps): self.meta_optimizer.step()

            optimizer.step()

            # REPTILE meta gradient
            if not self.ismaml:
                self.meta_optimizer.zero_grad()
                self.reptile_grad(self.generator, fast_generator)
                self.meta_optimizer.step()

        # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
        # making the forward pass expensive and hog up memory
        for item in loss_r_feature_layers:
            item.close()
        return inputs.detach()

    
    
class Synthesizer():
        
    def synthesize_representations(self, config, obj):
        """
        Synthesize representations for each class
        """
        lr, steps = config["data_lr"], obj["steps"]
        position = config["position"]
        alpha_preds, alpha_tv, alpha_l2, alpha_f = config["alpha_preds"], config["alpha_tv"], config["alpha_l2"], config["alpha_f"]
        orig_img, target_label, model = obj["orig_img"], obj["target_label"], obj["model"]
        loss_r_feature_layers = []

        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionHook(module))

        updated_img = orig_img.clone()
        updated_img.requires_grad = True
        updated_img.retain_grad()

        optimizer = torch.optim.Adam([updated_img], lr=lr, betas=(0.5, 0.9), eps = 1e-8)
        lim_0, lim_1 = 2, 2
        for it in range(1,steps+1):
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(updated_img, shifts=(off1, off2), dims=(2,3))

            model.zero_grad()
            optimizer.zero_grad()
            acts = model(inputs_jit, position=position)[:, :10]
            # ce_loss = nn.CrossEntropyLoss()(acts, target_label)
            probs = torch.softmax(acts, dim=1)
            entropy = -torch.sum(probs * torch.log(probs), dim=1).mean()
            loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers) if hasattr(model, "r_feature")])
            # loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
            loss = alpha_preds * entropy + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
            if it % 500 == 0 or it==steps:
                if len(target_label.size()) > 1:
                    acc = (acts.argmax(dim=1) == target_label.argmax(dim=1)).sum() / acts.shape[0]
                else:
                    acc = (acts.argmax(dim=1) == target_label).sum() / acts.shape[0]
                # acc = (acts.argmax(dim=1) == target_label.argmax(dim=1)).sum() / acts.shape[0]
                print(f"{it}/{steps}", loss_r_feature.item(), acc) # type: ignore

            loss.backward()
            optimizer.step()

        # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
        # making the forward pass expensive and hog up memory
        for item in loss_r_feature_layers:
            item.close()
        return updated_img.detach()
