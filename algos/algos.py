import torch.nn as nn
import random, torch
from torch.autograd import Variable
from algos.modules import DeepInversionFeatureHook, total_variation_loss
from algos.generator import Generator
from torchvision import transforms


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
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        
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
            acts = model.module(inputs_aug, position=position)
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
