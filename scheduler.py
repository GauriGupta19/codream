from mpi4py import MPI
import torch, random, numpy
from algos.base_class import BaseNode
from algos.fl import FedAvgClient, FedAvgServer
from algos.isolated import IsolatedServer
from utils.log_utils import copy_source_code
from utils.config_utils import load_config

# should be used as: algo_map[algo_name][rank>0](config)
# If rank is 0, then it returns the server class otherwise the client class
algo_map = {
    "fedavg": [FedAvgServer, FedAvgClient],
    "isolated": [IsolatedServer],
}

def get_node(config: dict, rank) -> BaseNode:
    algo_name = config["algo"]
    return algo_map[algo_name][rank>0](config)


class Scheduler():
    """ Manages the overall orchestration of experiments
    """
    def __init__(self) -> None:
        pass

    def assign_config_by_path(self, config_path) -> None:
        self.config = load_config(config_path)

    def initialize(self) -> None:
        assert self.config is not None, "Config should be set when initializing"

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # set seeds
        seed = self.config["seed"]
        torch.manual_seed(seed); random.seed(seed); numpy.random.seed(seed)

        if rank == 0:
            copy_source_code(self.config)
        self.node = get_node(self.config, rank=rank)


    # def new_train(self, model, dloader, optim, dist_data, position=0):
    #     model.train()
    #     correct_local, correct_dist, total_loss, samples_1, samples_2 = 0, 0, 0, 0, 0
    #     for x, y_hat in dloader:
    #         optim.zero_grad()
    #         x_, y_hat_ = dist_data[0].to(self.device), dist_data[1].to(self.device)
    #         y_ = model(x_, position=position)
    #         y_ = nn.functional.log_softmax(y_, dim=1)
    #         loss = kl_loss_fn(y_, nn.functional.softmax(y_hat_, dim=1))
    #         correct_dist += (y_.argmax(dim=1) == y_hat_.argmax(dim=1)).sum()
    #         samples_2 += x_.shape[0]

    #         x, y_hat = x.to(self.device), y_hat.to(self.device)
    #         y = model(x)
    #         correct_local += (y.argmax(dim=1) == y_hat).sum()
    #         samples_1 += x.shape[0]
    #         loss += ce_loss_fn(y, y_hat)

    #         loss.backward()
    #         optim.step()
    #         total_loss += loss.item()

    #     print("train acc on local", correct_local / samples_1)
    #     print("train acc on distilled", correct_dist / samples_2)
    #     print("output y_'s", y_.argmax(dim=1).unique(return_counts=True))
    #     print("distilled y_hat's", y_hat_.argmax(dim=1).unique(return_counts=True))
    #     return total_loss / float(samples_1 + samples_2)


    # def start_iid_clients_isolated(self):
    #     self.utils.logger.log_console("Starting iid clients isolated training")
    #     for epoch in range(self.start_epoch, self.config["epochs"]):
    #         num_clients = self.web_obj.num_clients
    #         for client_num in range(num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             self.utils.logger.log_console(f"Evaluating client {client_num}")
    #             acc = self.test(c_model, self.web_obj.test_loader)
    #             self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
    #             self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
    #                 epoch, acc, client_num
    #             ))

    #         for client_num in range(num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             c_model.train()
    #             c_optim = self.web_obj.c_optims[client_num]
    #             c_dloader = self.web_obj.c_dloaders[client_num]
    #             avg_loss = self.train(c_model, c_dloader, c_optim)
    #             self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    #         # Save models
    #         for i in range(self.web_obj.num_clients):
    #             torch.save(self.web_obj.c_models[i].module.state_dict(),
    #                         "{}/saved_models/c{}.pt".format(self.config["results_path"], i))

    # def start_iid_clients_collab(self):
    #     for epoch in range(self.start_epoch, self.config["epochs"]):
    #         collab_data = []
    #         # Evaluate on the common test set
    #         for client_num in range(self.web_obj.num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             acc = self.test(c_model, self.web_obj.test_loader)
    #             # log accuracy for every client
    #             self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
    #             self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
    #                 epoch, acc, client_num
    #             ))

    #         if epoch >= self.config["warmup"]:
    #             # Generate collab data
    #             for client_num in range(self.web_obj.num_clients):
    #                 """ We generate a zero vector of n (num_classes dimension)
    #                 then we generate random numbers within range n and substitute
    #                 zero at every index obtained from random number to be 1
    #                 This way the zero vector becomes a random one-hot vector
    #                 """
    #                 bs = self.config["distill_batch_size"]
    #                 zeroes = torch.zeros(bs, self.dset_obj.NUM_CLS)
    #                 ind = torch.randint(low=0, high=10, size=(bs,))
    #                 zeroes[torch.arange(start=0, end=bs), ind] = 1
    #                 target = zeroes.to(self.device)
    #                 self.config["inp_shape"][0] = bs
    #                 rand_imgs = torch.randn(self.config["inp_shape"]).to(self.device)

    #                 c_model = self.web_obj.c_models[client_num]
    #                 c_model.eval()
    #                 self.utils.logger.log_console(f"generating image for client {client_num}")
    #                 obj = {"orig_img": rand_imgs, "target_label": target,
    #                     "data": self.dset_obj, "model": c_model, "device": self.device}
    #                 updated_imgs = run_grad_ascent_on_data(self.config, obj)
    #                 # well we can not log all the channels of the intermediate representations so save first 3 channels
    #                 self.utils.logger.log_image(updated_imgs[:, :3], f"adaptive_client{client_num}", epoch)
    #                 acts = c_model(updated_imgs, position=self.config["position"]).detach()
    #                 collab_data.append((updated_imgs, acts))
    #                 self.utils.logger.log_console(f"generated image")

    #         # Train each client on their own data and collab data
    #         for client_num in range(self.web_obj.num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             c_model.train()
    #             c_optim = self.web_obj.c_optims[client_num]
    #             c_dloader = self.web_obj.c_dloaders[client_num]
    #             if epoch >= self.config["warmup"]:
    #                 # Train it 10 times on the same distilled dataset
    #                 for _ in range(self.config["distill_epochs"]):
    #                     for c_num, (x, y_hat) in enumerate(collab_data):
    #                         if c_num == client_num:
    #                             # no need to train on its own distilled data
    #                             continue
    #                         x, y_hat = x.to(self.device), y_hat.to(self.device)
    #                         c_optim.zero_grad()
    #                         y = c_model(x, position=self.config["position"])
    #                         y = nn.functional.log_softmax(y, dim=1)
    #                         loss = kl_loss_fn(y, nn.functional.softmax(y_hat, dim=1))
    #                         loss.backward()
    #                         c_optim.step()
    #             avg_loss = self.train(c_model, c_dloader, c_optim)
    #             self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    #         # Save models
    #         for i in range(self.web_obj.num_clients):
    #             torch.save(self.web_obj.c_models[i].module.state_dict(),
    #                        "{}/saved_models/c{}.pt".format(self.config["results_path"], i))
    
    # def start_iid_clients_adaptive_collab(self):
    #     for epoch in range(self.start_epoch, self.config["epochs"]):
    #         collab_data = []
    #         # Evaluate on the common test set
    #         for client_num in range(self.web_obj.num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             acc = self.test(c_model, self.web_obj.test_loader)
    #             # log accuracy for every client
    #             self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
    #             self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
    #                 epoch, acc, client_num
    #             ))

    #         if epoch >= self.config["warmup"]:
    #             # Generate collab data
    #             for client_num in range(self.web_obj.num_clients):
    #                 """ We generate a zero vector of n (num_classes dimension)
    #                 then we generate random numbers within range n and substitute
    #                 zero at every index obtained from random number to be 1
    #                 This way the zero vector becomes a random one-hot vector
    #                 """
    #                 bs = self.config["distill_batch_size"]
    #                 zeroes = torch.zeros(bs, self.dset_obj.NUM_CLS)
    #                 ind = torch.randint(low=0, high=10, size=(bs,))
    #                 zeroes[torch.arange(start=0, end=bs), ind] = 1
    #                 target = zeroes.to(self.device)

    #                 self.config["inp_shape"][0] = bs
    #                 rand_imgs = torch.randn(self.config["inp_shape"]).to(self.device)

    #                 c_model = self.web_obj.c_models[client_num]
    #                 c_model.eval()
    #                 self.utils.logger.log_console(f"generating image for client {client_num}")
    #                 obj = {"orig_img": rand_imgs, "target_label": target,
    #                        "data": self.dset_obj, "models_t": [m for i, m in enumerate(self.web_obj.c_models) if i != client_num],
    #                        "model_s": self.web_obj.c_models[client_num],
    #                        "device": self.device}
    #                 updated_imgs = adaptive_run_grad_ascent_on_data(self.config, obj)
    #                 # well we can not log all the channels of the intermediate representations so save first 3 channels
    #                 self.utils.logger.log_image(updated_imgs[:, :3], f"adaptive_client{client_num}", epoch)
    #                 acts = c_model(updated_imgs, position=self.config["position"]).detach()
    #                 collab_data.append((updated_imgs, acts))
    #                 self.utils.logger.log_console(f"generated image")

    #         # Train each client on their own data and collab data
    #         for client_num in range(self.web_obj.num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             c_model.train()
    #             c_optim = self.web_obj.c_optims[client_num]
    #             c_dloader = self.web_obj.c_dloaders[client_num]
    #             if epoch >= self.config["warmup"]:
    #                 # Train it 10 times on the same distilled dataset
    #                 for _ in range(self.config["distill_epochs"]):
    #                     for c_num, (x, y_hat) in enumerate(collab_data):
    #                         if c_num == client_num:
    #                             # no need to train on its own distilled data
    #                             continue
    #                         x, y_hat = x.to(self.device), y_hat.to(self.device)
    #                         c_optim.zero_grad()
    #                         y = c_model(x, position=self.config["position"])
    #                         y = nn.functional.log_softmax(y, dim=1)
    #                         loss = kl_loss_fn(y, nn.functional.softmax(y_hat, dim=1))
    #                         loss.backward()
    #                         c_optim.step()
    #             avg_loss = self.train(c_model, c_dloader, c_optim)
    #             self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    #         # Save models
    #         for i in range(self.web_obj.num_clients):
    #             torch.save(self.web_obj.c_models[i].module.state_dict(),
    #                        "{}/saved_models/c{}.pt".format(self.config["results_path"], i))


    # def start_non_iid_federated(self):
    #     self.utils.logger.log_console("Starting iid clients federated averaging")
    #     for epoch in range(self.start_epoch, self.config["epochs"]):
    #         num_clients = self.web_obj.num_clients
    #         for client_num in range(num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             self.utils.logger.log_console(f"Evaluating client {client_num}")
    #             acc = self.test(c_model, self.web_obj.test_loader)
    #             self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
    #             self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
    #                 epoch, acc, client_num
    #             ))

    #         for client_num in range(num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             c_model.train()
    #             c_optim = self.web_obj.c_optims[client_num]
    #             c_dloader = self.web_obj.c_dloaders[client_num]
    #             avg_loss = self.train(c_model, c_dloader, c_optim)
    #             self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    #         self.fed_avg(self.web_obj.c_models)
    #         # Save models
    #         for i in range(self.web_obj.num_clients):
    #             torch.save(self.web_obj.c_models[i].module.state_dict(),
    #                         "{}/saved_models/c{}.pt".format(self.config["results_path"], i))


    # def start_non_iid_clients_collab(self):
    #     for epoch in range(self.start_epoch, self.config["epochs"]):
    #         collab_data = []
    #         # Evaluate on the common test set
    #         for client_num in range(self.web_obj.num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             acc = self.test(c_model, self.web_obj.test_loader)
    #             # log accuracy for every client
    #             self.utils.logger.log_tb(f"test_acc/client{client_num}", acc, epoch)
    #             self.utils.logger.log_console("epoch: {} test_acc:{:.4f} client:{}".format(
    #                 epoch, acc, client_num
    #             ))

    #         if epoch >= self.config["warmup"]:
    #             # Generate collab data
    #             for client_num in range(self.web_obj.num_clients):
    #                 """ We generate a zero vector of n (num_classes dimension)
    #                 then we generate random numbers within range n and substitute
    #                 zero at every index obtained from random number to be 1
    #                 This way the zero vector becomes a random one-hot vector
    #                 """
    #                 bs = self.config["distill_batch_size"]
    #                 zeroes = torch.zeros(bs, self.dset_obj.NUM_CLS)
    #                 # Hacky way of getting labels from the same distribution
    #                 # Get the labels from the first batch of the client's dataloader
    #                 # size of ind should be greater than bs
    #                 ind = next(iter(self.web_obj.c_dloaders[(client_num + 1) % 2]))[1][:bs]
    #                 zeroes[torch.arange(start=0, end=bs), ind] = 1
    #                 target = zeroes.to(self.device)
    #                 print(f"client {client_num} generated", target.argmax(dim=1).unique(return_counts=True))

    #                 self.config["inp_shape"][0] = bs
    #                 rand_imgs = torch.randn(self.config["inp_shape"]).to(self.device)

    #                 c_model = self.web_obj.c_models[client_num]
    #                 c_model.eval()
    #                 self.utils.logger.log_console(f"generating image for client {client_num}")
    #                 # models_t refers to the model of the client that we are generating images from
    #                 # model_s refers to the model of the client that we are generating images for
    #                 obj = {"orig_img": rand_imgs, "target_label": target,
    #                        "data": self.dset_obj, "models_t": [m for i, m in enumerate(self.web_obj.c_models) if i != client_num],
    #                        "model_s": self.web_obj.c_models[client_num],
    #                        "device": self.device}
    #                 # Generate images by running gradient ascent on the random images
    #                 updated_imgs = adaptive_run_grad_ascent_on_data(self.config, obj)
    #                 # Log the generated images
    #                 self.utils.logger.log_image(updated_imgs[:, :3], f"client{client_num}", epoch)
    #                 # Get the activations of the generated images
    #                 acts = self.web_obj.c_models[(client_num + 1) % 2](updated_imgs, position=self.config["position"]).detach()
    #                 # Add the generated images and their activations to the collab data
    #                 collab_data.append((updated_imgs, acts))
    #                 # collab_data.append((updated_imgs, target))
    #                 self.utils.logger.log_console(f"generated image")

    #         # Train each client on their own data and collab data
    #         for client_num in range(self.web_obj.num_clients):
    #             c_model = self.web_obj.c_models[client_num]
    #             c_model.train()
    #             c_optim = self.web_obj.c_optims[client_num]
    #             c_dloader = self.web_obj.c_dloaders[client_num]
    #             if epoch >= self.config["warmup"]:
    #                 # Train it 10 times on the same distilled dataset
    #                 for _ in range(self.config["distill_epochs"]):
    #                     for c_num, (x, y_hat) in enumerate(collab_data):
    #                         if c_num == client_num:
    #                             # no need to train on its own distilled data
    #                             x, y_hat = x.to(self.device), y_hat.to(self.device)
    #                             c_optim.zero_grad()
    #                             y = c_model(x)
    #                             if _ == 0 or _ == self.config["distill_epochs"] - 1:
    #                                 print(f"client num {client_num}, it num {_}")
    #                                 print("y ->", y.argmax(dim=1), "y_hat ->", y_hat.argmax(dim=1))
    #                             y = nn.functional.log_softmax(y, dim=1)
    #                             loss = kl_loss_fn(y, nn.functional.softmax(y_hat, dim=1))
    #                             loss.backward()
    #                             c_optim.step()
    #                 # dist_data = collab_data[client_num]
    #                 # avg_loss = self.new_train(c_model, c_dloader, c_optim, dist_data=dist_data, position=self.config["position"])
    #             avg_loss = self.train(c_model, c_dloader, c_optim)
    #             # avg_loss = self.train(c_model, c_dloader, c_optim)
    #             # for c_num, (x, y_hat) in enumerate(collab_data):
    #             #     if c_num == client_num:
    #             #         # no need to train on its own distilled data
    #             #         continue
    #             #     x, y_hat = x.to(self.device), y_hat.to(self.device)
    #             #     c_optim.zero_grad()
    #             #     y = c_model(x)
    #             #     print(f"after local training of {client_num}", y.argmax(dim=1), y_hat.argmax(dim=1))
    #             self.utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    #         # Save models
    #         for i in range(self.web_obj.num_clients):
    #             torch.save(self.web_obj.c_models[i].module.state_dict(),
    #                        "{}/saved_models/c{}.pt".format(self.config["results_path"], i))


    def run_job(self) -> None:
        # if self.config["exp_type"] == "iid_clients_collab":
        #     self.start_iid_clients_collab()
        # if self.config["exp_type"] == "iid_clients_adaptive_collab":
        #     self.start_iid_clients_adaptive_collab()
        # elif self.config["exp_type"] == "iid_clients_isolated":
        #     self.start_iid_clients_isolated()
        # elif self.config["exp_type"] == "non_iid_clients_collab":
        #     # currently it is adaptive version
        #     self.start_non_iid_clients_collab()
        # if self.config["exp_type"] == "iid_clients_federated":
        self.node.run_protocol()
        # elif self.config["exp_type"] == "non_iid_clients_federated":
        #     self.start_non_iid_federated()
