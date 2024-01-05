import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from collections import OrderedDict

kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
ce_loss_fn = nn.CrossEntropyLoss()

def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)
    
def get_model_grads(model):
    local_gradients = {}
    for name, param in model.named_parameters():
        local_gradients[name] = param.grad.data.to("cpu")
    return local_gradients

def aggregate_model_grads(grads, model, device):
    # Update the server model with the received gradients
    n = len(grads)
    for grad in grads:
        for name, param in model.named_parameters():
            if param.grad is None:
                param.grad = Variable(torch.zeros(param.size())).to(device)
            param.grad.data.add_(grad[name].to(device)/n)

def reptile_grad(src, tar, device):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).to(device)
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40

def fomaml_grad(src, tar, device):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).to(device)
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67

def reset_l0(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

def put_on_cpu(wts):
    for k, v in wts.items():
        wts[k] = v.to("cpu")
    return wts

def adam_update(rep, grad, optimizer):
    betas = optimizer.param_groups[0]['betas']
    # access the optimizer's state
    state = optimizer.state[rep]
    if 'exp_avg' not in state:
        state['exp_avg'] = torch.zeros_like(rep.data)
    if 'exp_avg_sq' not in state:
        state['exp_avg_sq'] = torch.zeros_like(rep.data)
    if 'step' not in state:
        state['step'] = 1
    exp_avg = state['exp_avg']
    exp_avg_sq = state['exp_avg_sq']
    bias_correction1 = 1 - betas[0] ** state['step']
    bias_correction2 = 1 - betas[1] ** state['step']
    exp_avg.mul_(betas[0]).add_(-grad, alpha=1 - betas[0])
    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(optimizer.param_groups[0]['eps'])
    step_size = optimizer.param_groups[0]['lr'] / bias_correction1
    rep.data.addcdiv_(exp_avg, denom, value=-step_size)
    optimizer.state[rep]['step'] += 1

    # # ---- FedAdam ----
    # lr = self.config["lr_z"]
    # betas = optimizer.param_groups[0]['betas']
    # beta_1 = 0.9
    # beta_2 = 0.99
    # tau = 1e-9
    # state = optimizer.state[rep]
    # if "m" not in state:
    #     state["m"] = torch.zeros_like(rep.data)
    # if "v" not in state:
    #     state["v"] = torch.zeros_like(rep.data)
    # state["m"] = beta_1 * state["m"] + (1 - beta_1) * grad
    # state["v"] = beta_2 * state["v"] + (1 - beta_2) * grad**2
    # update = rep.data + lr * state["m"] / (state["v"] ** 0.5 + tau)
    # update = update.detach()
    # rep.data = update

def adam_step(optimizer):
    state = optimizer.state
    if 'step' not in state:
        state['step'] = 1
    for idx, group in enumerate(optimizer.param_groups): 
        betas = group['betas']
        lr = group['lr']
        eps = group['eps']
        # Access the optimizer's state        
        bias_correction1 = 1 - betas[0] ** state['step']
        bias_correction2 = 1 - betas[1] ** state['step']
        for param in group['params']:
            delta = param.grad.data
            if 'exp_avg' not in state[param]:
                state[param]['exp_avg'] = torch.zeros_like(param.data)
            exp_avg = state[param]['exp_avg'] 
            # \beta1 * m_t + (1 - \beta1) * \Delta_t
            exp_avg.mul_(betas[0]).add_(-delta, alpha=1 - betas[0]) 
            if 'exp_avg_sq' not in state[param]:
                state[param]['exp_avg_sq'] = torch.zeros_like(param.data)
            exp_avg_sq = state[param]['exp_avg_sq'] 
            # \beta2 * v_t + (1 - \beta2) * \Delta_t^2
            exp_avg_sq.mul_(betas[1]).addcmul_(delta, delta, value=1 - betas[1])         
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            param.data.addcdiv_(exp_avg, denom, value = -step_size)
    state['step'] += 1

def aggregate_models(model_wts, weights=None, reduction='avg'):
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
                if weights is not None:
                    coeff = weights[client_num]  
                elif reduction=='sum':
                    coeff = 1
                else:
                    coeff = 1 / num_clients
                if client_num == 0:
                    avgd_wts[key] = coeff * local_wts[key]
                else:
                    avgd_wts[key] += coeff * local_wts[key]
        return avgd_wts