import torch.nn as nn
import random, torch
from modules import DeepInversionFeatureHook, total_variation_loss
from generative_model import skip


def run_grad_ascent_on_data(config, obj):
    lr, steps = config["data_lr"], config["steps"]
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

    optimizer = torch.optim.Adam([updated_img], lr=lr, betas=[0.5, 0.9], eps = 1e-8)
    lim_0, lim_1 = 2, 2
    for it in range(steps):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(updated_img, shifts=(off1, off2), dims=(2,3))

        model.zero_grad()
        optimizer.zero_grad()
        acts = model.module(inputs_jit, position=position)[:, :10]
        ce_loss = nn.CrossEntropyLoss()(acts, target_label)
        loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers) if hasattr(model, "r_feature")])
        loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        if it % 500 == 0:
            print(f"{it}/{steps}", loss_r_feature.item(), (acts.argmax(dim=1) == target_label.argmax(dim=1)).sum() / acts.shape[0])

        loss.backward()
        optimizer.step()
    
    # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
    # making the forward pass expensive and hog up memory
    for item in loss_r_feature_layers:
        item.close()
    return updated_img.detach()


def adaptive_run_grad_ascent_on_data(config, obj):
    lr, steps = config["data_lr"], config["steps"]
    position = config["position"]
    alpha_preds, alpha_tv, alpha_l2, alpha_f = config["alpha_preds"], config["alpha_tv"], config["alpha_l2"], config["alpha_f"]
    orig_img, target_label, models_t, model_s = obj["orig_img"], obj["target_label"], obj["models_t"], obj["model_s"]
    loss_r_feature_layers_t, loss_r_feature_layers_s = [], []

    for m in models_t:
        m.eval()
    model_s.eval()
    for m in models_t:
        for module in m.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers_t.append(DeepInversionFeatureHook(module))
    for module in model_s.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers_s.append(DeepInversionFeatureHook(module))

    updated_img = orig_img.clone()
    updated_img.requires_grad = True
    updated_img.retain_grad()

    optimizer = torch.optim.Adam([updated_img], lr=lr, betas=[0.5, 0.9], eps = 1e-8)
    lim_0, lim_1 = 2, 2
    for it in range(steps):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(updated_img, shifts=(off1, off2), dims=(2,3))

        for m in models_t:
            m.zero_grad()
        model_s.zero_grad()
        optimizer.zero_grad()
        ce_loss_t_sum = 0
        for m in models_t:
            acts_t = m.module(inputs_jit, position=position)[:, :10]
            ce_loss_t_sum += nn.CrossEntropyLoss()(acts_t, target_label)
        acts_s = model_s.module(inputs_jit, position=position)[:, :10]
        ce_loss_s = nn.CrossEntropyLoss()(acts_s, target_label)
        loss_r_feature_t = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers_t) if hasattr(model, "r_feature")])
        loss_r_feature_s = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers_s) if hasattr(model, "r_feature")])
        # multiplying with 0 here is a hack to make the loss 0 for the student model, TODO: fix this
        loss = alpha_preds * (ce_loss_t_sum - 0 * ce_loss_s) +\
               alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) +\
               alpha_f * (loss_r_feature_t + loss_r_feature_s)
        if it % 500 == 0:
            acc_t = (acts_t.argmax(dim=1) == target_label.argmax(dim=1)).sum() / acts_t.shape[0]
            acc_s = (acts_s.argmax(dim=1) == target_label.argmax(dim=1)).sum() / acts_s.shape[0]
            print(f"{it}/{steps}", loss_r_feature_t.item(), loss_r_feature_s.item(), acc_s, acc_t)

        loss.backward()
        optimizer.step()
    
    # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
    # making the forward pass expensive and hog up memory
    for item in loss_r_feature_layers_t:
        item.close()
    for item in loss_r_feature_layers_s:
        item.close()
    return updated_img.detach()