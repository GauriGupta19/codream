import torch.nn as nn
import random, torch
from modules import DeepInversionFeatureHook, total_variation_loss


def run_grad_ascent_on_data(config, obj):
    lr, steps = config["data_lr"], config["steps"]
    LOWER_IMAGE_BOUND, UPPER_IMAGE_BOUND = obj["data"].IMAGE_BOUND_L.to(obj["device"]), obj["data"].IMAGE_BOUND_U.to(obj["device"])
    alpha_tv, alpha_l2, alpha_f = config["alpha_tv"], config["alpha_l2"], config["alpha_f"]
    orig_img, target_label, model = obj["orig_img"], obj["target_label"], obj["model"]
    loss_r_feature_layers = []

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    updated_img = orig_img.clone()
    updated_img.requires_grad = True
    updated_img.retain_grad()

    lim_0, lim_1 = 2, 2
    for it in range(steps):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(updated_img.detach(), shifts=(off1, off2), dims=(2,3))

        acts = model.module(inputs_jit)[:, :10]
        # loss = nn.functional.mse_loss(target_label, acts).sum()
        ce_loss = nn.CrossEntropyLoss()(acts, target_label)

        # rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
        loss_r_feature = sum([model.r_feature for (idx, model) in enumerate(loss_r_feature_layers)])
        loss = ce_loss + alpha_tv * total_variation_loss(updated_img) + alpha_l2 * torch.linalg.norm(updated_img) + alpha_f * loss_r_feature
        loss.backward()
        # if it % 10 == 0:
        #     display.clear_output(wait=True)
        #     show(deprocess(updated_img[0]).detach().permute(1,2,0))
        #     print((updated_img - orig_img).mean())
        grads = updated_img.grad.data / (torch.std(updated_img.grad.data) + 1e-8)
        updated_img.data = updated_img.data - lr * grads
        updated_img.data = torch.clamp(updated_img.data, min=LOWER_IMAGE_BOUND, max=UPPER_IMAGE_BOUND)
        # optimizer.step()
        model.zero_grad()
        # optimizer.zero_grad()
        updated_img.grad.data.zero_()
    # Removing the hook frees up memory otherwise every call to this function adds up extra Forwardhook
    # making the forward pass expensive and hog up memory
    for item in loss_r_feature_layers:
        item.close()
    return updated_img.detach()