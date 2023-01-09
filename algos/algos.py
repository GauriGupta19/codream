import torch.nn as nn
import random, torch
from algos.modules import DeepInversionFeatureHook, total_variation_loss


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
    for it in range(1,steps+1):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(updated_img, shifts=(off1, off2), dims=(2,3))

        model.zero_grad()
        optimizer.zero_grad()
        acts = model.module(inputs_jit, position=position)[:, :10]
        ce_loss = nn.CrossEntropyLoss()(acts, target_label)
        loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers) if hasattr(model, "r_feature")])
        loss = alpha_preds * ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
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
