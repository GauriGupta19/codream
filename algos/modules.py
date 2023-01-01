import torch
import torch.nn as nn


kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
ce_loss_fn = nn.CrossEntropyLoss()


def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature_mean = torch.norm(module.running_var.data - var, 2) # type: ignore
        r_feature_std = torch.norm(module.running_mean.data - mean, 2) # type: ignore

        self.r_feature = r_feature_mean + r_feature_std
        # must have no output

    def close(self):
        self.hook.remove()