import torch.nn as nn
import random, torch
from algos.modules import DeepInversionFeatureHook, total_variation_loss
from utils.model_utils import ModelUtils


EPS = 1e-8

def synthesize_representations_collaborative(config, obj):
    lr, steps = config["data_lr"], obj["steps"]
    position = config["position"]
    alpha_preds, alpha_tv, alpha_l2, alpha_f = config["alpha_preds"], config["alpha_tv"], config["alpha_l2"], config["alpha_f"]
    orig_img = obj["orig_img"]
    models, loss_r_feature_layers = [], []
    for i in range(len(obj["model_wts"])):
        model = ModelUtils.get_model(config["model"], config["dset"], obj["device"], obj["device_ids"])
        model.module.load_state_dict(obj["model_wts"][i])
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        models.append(model)

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
        entropy = 0.
        for model in models:
            model.zero_grad()
            # print(inputs_jit.shape)
            acts = model.module(inputs_jit, position=position)
            # print(acts.shape)
            probs = torch.softmax(acts, dim=1)
            entropy += -torch.sum(probs * torch.log(probs + EPS), dim=1).mean()

        loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers) if hasattr(model, "r_feature")])
        # loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        loss = alpha_preds * entropy + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        if it % 500 == 0 or it==steps:
            print(f"{it}/{steps}", loss_r_feature.item(), entropy.item()) # type: ignore

        loss.backward()
        optimizer.step()

    # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
    # making the forward pass expensive and hog up memory
    for item in loss_r_feature_layers:
        item.close()
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
