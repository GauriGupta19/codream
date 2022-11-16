import torch, random, numpy
from distrib_utils import WebObj
from log_utils import Utils
from data_utils import get_dataset
from config_utils import load_config
from algos import run_grad_ascent_on_data
from modules import kl_loss_fn, ce_loss_fn
import torch.nn as nn


class Scheduler():
    """ Manages the overall orchestration of experiments
    """
    def __init__(self) -> None:
        pass

    def assign_config_by_path(self, config_path) -> None:
        self.config = load_config(config_path)

    def setup_cuda(self):
        self.device_ids = self.config["device_ids"]
        gpu_id = self.device_ids[0]

        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')
    
    def initialize(self) -> None:
        assert self.config is not None, "Config should be set when initializing"

        # set seeds
        seed = self.config["seed"]
        torch.manual_seed(seed); random.seed(seed); numpy.random.seed(seed)

        self.utils = Utils(self.config)
        self.setup_cuda()

        self.dset_obj = get_dataset(self.config["dset"], self.config["dpath"])
        obj = {
            'device': self.device,
            'device_ids': self.device_ids,
            'dset_obj': self.dset_obj
        }

        self.config["warmup"] = self.config.get("warmup") or 0
        if self.config["load_existing"]:
            self.start_epoch = self.config["start_epoch"]
        else:
            self.start_epoch = 0
        self.web_obj = WebObj(self.config, obj)

    def train(self, model, dloader, optim):
        model.train()
        correct, total_loss, samples = 0., 0, 0
        for x, y_hat in dloader:
            x, y_hat = x.to(self.device), y_hat.to(self.device)
            optim.zero_grad()
            y = model(x)
            correct += (y.argmax(dim=1) == y_hat).sum()
            loss = ce_loss_fn(y, y_hat)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            samples += x.shape[0]
        # print(y_hat, correct / samples)
        return total_loss / float(samples)

    def test(self, model, dloader):
        model.eval()
        correct, total = 0, 0
        for x, y_hat in dloader:
            with torch.no_grad():
                x, y_hat = x.to(self.device), y_hat.to(self.device)
                y = model(x)
                correct += (y.argmax(dim=1) == y_hat).sum()
                total += x.shape[0]
        return correct / float(total)


    def start_iid_clients_isolated(self):
        self.utils.logger.log_console("Starting iid clients isolated training")
        for epoch in range(self.start_epoch, self.config["epochs"]):
            num_clients = self.web_obj.num_clients
            for client_num in range(num_clients):
                c_model = self.web_obj.c_models[client_num]
                self.utils.logger.log_console(f"Evaluating client {client_num}")
                acc = self.test(c_model, self.web_obj.test_loader)
                self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
                self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
                    epoch, acc, client_num
                ))

            for client_num in range(num_clients):
                c_model = self.web_obj.c_models[client_num]
                c_model.train()
                c_optim = self.web_obj.c_optims[client_num]
                c_dloader = self.web_obj.c_dloaders[client_num]
                avg_loss = self.train(c_model, c_dloader, c_optim)
                self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
            # Save models
            for i in range(self.web_obj.num_clients):
                torch.save(self.web_obj.c_models[i].module.state_dict(),
                            "{}/saved_models/c{}.pt".format(self.config["results_path"], i))
                

    def start_iid_clients_collab(self):
        for epoch in range(self.start_epoch, self.config["epochs"]):
            collab_data = []
            # Evaluate on the common test set
            for client_num in range(self.web_obj.num_clients):
                c_model = self.web_obj.c_models[client_num]
                acc = self.test(c_model, self.web_obj.test_loader)
                # log accuracy for every client
                self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
                self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
                    epoch, acc, client_num
                ))

            if epoch >= self.config["warmup"]:
                # Generate collab data
                for client_num in range(self.web_obj.num_clients):
                    """ We generate a zero vector of n (num_classes dimension)
                    then we generate random numbers within range n and substitute
                    zero at every index obtained from random number to be 1
                    This way the zero vector becomes a random one-hot vector
                    """
                    bs = self.config["distill_batch_size"]
                    zeroes = torch.zeros(bs, self.dset_obj.NUM_CLS)
                    ind = torch.randint(low=0, high=10, size=(bs,))
                    zeroes[torch.arange(start=0, end=bs), ind] = 1
                    target = zeroes.to(self.device)
                    # For RGB 3 Channels
                    rand_imgs = torch.randn(bs, 3, self.dset_obj.IMAGE_SIZE, self.dset_obj.IMAGE_SIZE).to(self.device)

                    c_model = self.web_obj.c_models[client_num]
                    c_model.eval()
                    self.utils.logger.log_console(f"generating image for client {client_num}")
                    obj = {"orig_img": rand_imgs, "target_label": target,
                        "data": self.dset_obj, "model": c_model, "device": self.device}
                    updated_imgs = run_grad_ascent_on_data(self.config, obj)
                    acts = c_model(updated_imgs).detach()
                    collab_data.append((updated_imgs, acts))
                    self.utils.logger.log_console(f"generated image")

            # Train each client on their own data and collab data
            for client_num in range(self.web_obj.num_clients):
                c_model = self.web_obj.c_models[client_num]
                c_model.train()
                c_optim = self.web_obj.c_optims[client_num]
                c_dloader = self.web_obj.c_dloaders[client_num]
                if epoch >= self.config["warmup"]:
                    # Train it 10 times on the same distilled dataset
                    for _ in range(self.config["distill_epochs"]):
                        for c_num, (x, y_hat) in enumerate(collab_data):
                            if c_num == client_num:
                                # no need to train on its own distilled data
                                continue
                            x, y_hat = x.to(self.device), y_hat.to(self.device)
                            c_optim.zero_grad()
                            y = c_model(x)
                            y = nn.functional.log_softmax(y, dim=1)
                            loss = kl_loss_fn(y, nn.functional.softmax(y_hat, dim=1))
                            loss.backward()
                            c_optim.step()
                avg_loss = self.train(c_model, c_dloader, c_optim)
                self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
            # Save models
            for i in range(self.web_obj.num_clients):
                torch.save(self.web_obj.c_models[i].module.state_dict(),
                           "{}/saved_models/c{}.pt".format(self.config["results_path"], i))
    

    def start_non_iid_clients_collab(self):
        for epoch in range(self.start_epoch, self.config["epochs"]):
            collab_data = []
            # Evaluate on the common test set
            for client_num in range(self.web_obj.num_clients):
                c_model = self.web_obj.c_models[client_num]
                acc = self.test(c_model, self.web_obj.test_loader)
                # log accuracy for every client
                self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
                self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
                    epoch, acc, client_num
                ))

            if epoch >= self.config["warmup"]:
                # Generate collab data
                for client_num in range(self.web_obj.num_clients):
                    """ We generate a zero vector of n (num_classes dimension)
                    then we generate random numbers within range n and substitute
                    zero at every index obtained from random number to be 1
                    This way the zero vector becomes a random one-hot vector
                    """
                    bs = self.config["distill_batch_size"]
                    zeroes = torch.zeros(bs, self.dset_obj.NUM_CLS)
                    # Hacky way of getting labels from the same distribution
                    ind = next(iter(self.web_obj.c_dloaders[client_num]))[1][:bs]
                    zeroes[torch.arange(start=0, end=bs), ind] = 1
                    target = zeroes.to(self.device)
                    # For RGB 3 Channels
                    rand_imgs = torch.randn(bs, 3, self.dset_obj.IMAGE_SIZE, self.dset_obj.IMAGE_SIZE).to(self.device)

                    c_model = self.web_obj.c_models[client_num]
                    c_model.eval()
                    self.utils.logger.log_console(f"generating image for client {client_num}")
                    obj = {"orig_img": rand_imgs, "target_label": target,
                        "data": self.dset_obj, "model": c_model, "device": self.device}
                    updated_imgs = run_grad_ascent_on_data(self.config, obj)
                    acts = c_model(updated_imgs).detach()
                    collab_data.append((updated_imgs, acts))
                    self.utils.logger.log_console(f"generated image")

            # Train each client on their own data and collab data
            for client_num in range(self.web_obj.num_clients):
                c_model = self.web_obj.c_models[client_num]
                c_model.train()
                c_optim = self.web_obj.c_optims[client_num]
                c_dloader = self.web_obj.c_dloaders[client_num]
                if epoch >= self.config["warmup"]:
                    # Train it 10 times on the same distilled dataset
                    for _ in range(self.config["distill_epochs"]):
                        for c_num, (x, y_hat) in enumerate(collab_data):
                            if c_num == client_num:
                                # no need to train on its own distilled data
                                continue
                            x, y_hat = x.to(self.device), y_hat.to(self.device)
                            c_optim.zero_grad()
                            y = c_model(x)
                            y = nn.functional.log_softmax(y, dim=1)
                            loss = kl_loss_fn(y, nn.functional.softmax(y_hat, dim=1))
                            loss.backward()
                            c_optim.step()
                avg_loss = self.train(c_model, c_dloader, c_optim)
                self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
            # Save models
            for i in range(self.web_obj.num_clients):
                torch.save(self.web_obj.c_models[i].module.state_dict(),
                           "{}/saved_models/c{}.pt".format(self.config["results_path"], i))

    def run_job(self) -> None:
        if self.config["exp_type"] == "iid_clients_collab":
            self.start_iid_clients_collab()
        elif self.config["exp_type"] == "iid_clients_isolated":
            self.start_iid_clients_isolated()
        elif self.config["exp_type"] == "non_iid_clients_collab":
            self.start_non_iid_clients_collab()
