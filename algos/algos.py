import pdb
import time
import torch.nn as nn
import random, torch
from algos.modules import DeepInversionFeatureHook, total_variation_loss
from utils.model_utils import ModelUtils
from torchvision import utils


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
            # print(loss_r_feature, entropy.device, inputs_jit.device)
            loss = self.alpha_preds * entropy + self.alpha_tv * total_variation_loss(inputs_jit).to(entropy.device) +\
                self.alpha_l2 * torch.linalg.norm(inputs_jit).to(entropy.device) + self.alpha_f * loss_r_feature
            loss.backward()
        #     inputs.data = inputs.data - inputs.grad.data * 0.05
            # if self.id == 1:
                # print("...", self.num, inputs.grad.data.mean(), inputs.grad.data.std(), inputs.grad.data.min(), inputs.grad.data.max())
            self.num += 1
        # exit()
        return inputs.grad.unsqueeze(dim=0), loss_r_feature, entropy, self.id


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
        model.module.load_state_dict(obj["model_wts"][i])
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        grad_fn = GradFn(model, loss_r_feature_layers, position,
                         alpha_preds=alpha_preds, alpha_tv=alpha_tv, alpha_l2=alpha_l2, alpha_f=alpha_f,
                         id_=i)
        models.append(model)
        grad_fns.append(grad_fn)

    orig_img = torch.load("orig_img.pt").to(obj["device"])
    updated_img = orig_img.clone()
    # updated_img.requires_grad = True
    # updated_img.retain_grad()

    optimizer = torch.optim.SGD([updated_img], lr=lr, momentum=0., weight_decay=0.)
    lim_0, lim_1 = 2, 2

    def _agg_grads(grads):
        grads_on_same_device = nn.parallel.scatter_gather.gather(grads, main_device)
        grads = grads_on_same_device
        return grads.mean(dim=0)

    # time the following loop
    start = time.time()
    steps = 500
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

        # optimizer.zero_grad()
        # pdb.set_trace()
        output = nn.parallel.parallel_apply(grad_fns, inputs, kwargs_tup=kwargs_tup, devices=obj["device_ids"])
        # grads = [output[0][0], output[1][0], output[2][0]]
        grads = []
        for i in range(len(output)):
            grads.append(output[i][0])
        # updated_img = output[0][0].unsqueeze(dim=0)
        # loss_r_features = [output[0][1].cuda(), output[1][1].cuda(), output[2][1].cuda()]
        # entropies = [output[0][2].cuda(), output[1][2].cuda(), output[2][2].cuda()]
        ids = [output[0][3], output[1][3], output[2][3]]

        # for i in range(len(ids)):
        #     if ids[i] == 1:
        #         grad = grads[i][0].to(main_device)
        # grad = grads[0][0].to(main_device)
        grad = _agg_grads(grads)
        # print(f"{it}/{steps}", sum(loss_r_features).item(), sum(entropies).item())
        # print(it, grad.data.mean(), grad.std(), grad.min(), grad.max())
        updated_img.grad = grad
        if it % 500 == 0 or it==steps:
            print(f"{it}/{steps}")
        # optimizer.step()
        updated_img.data = updated_img.data - lr * updated_img.grad.data
        updated_img.grad.zero_()

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
                loss_r_feature_layers[i].append(DeepInversionFeatureHook(module))
        models.append(model)
    orig_img = torch.load("orig_img.pt").to(obj["device"])
    updated_img = orig_img.clone()
    updated_img.requires_grad = True
    updated_img.retain_grad()

    optimizer = torch.optim.Adam([updated_img], lr=lr, betas=(0.5, 0.9), eps = 1e-8)
    lim_0, lim_1 = 2, 2
    steps = 500
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
            # print(inputs_jit.shape)
            acts = model.module(inputs_jit, position=position)
            # print(acts.shape)
            probs = torch.softmax(acts, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1).mean()
            loss_r_feature = sum([m_.r_feature for (idx, m_) in enumerate(loss_r_feature_layers[i]) if hasattr(m_, "r_feature")])
            loss_r_features += loss_r_feature
            entropies += entropy

        # loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        loss = alpha_preds * entropies + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_features
        if it % 1 == 0 or it==steps:
            print(f"{it}/{steps}", loss_r_features.item(), entropies.item()) # type: ignore

        loss.backward()
        # optimizer.step()
        grad = updated_img.grad
        print(grad.mean(), grad.std())
        updated_img.data = updated_img.data - lr * updated_img.grad.data * (1 / len(models))

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
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

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
        acts = model.module(inputs_jit, position=position)
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
    
    # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
    # making the forward pass expensive and hog up memory
    for item in loss_r_feature_layers:
        item.close()
    return updated_img.detach()
