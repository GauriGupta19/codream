import pdb
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.data import Subset


class CIFAR10_DSET():
    def __init__(self, dpath) -> None:
        self.IMAGE_SIZE = 32
        self.NUM_CLS = 10
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.num_channels = 3
        self.gen_transform = T.Compose(
            [
                T.RandomCrop(size=[32, 32], padding=4),
                T.RandomHorizontalFlip(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        self.train_dset = CIFAR10(
            root=dpath, train=True, download=True, transform=train_transform
        )
        self.test_dset = CIFAR10(
            root=dpath, train=False, download=True, transform=test_transform
        )
        self.IMAGE_BOUND_L = torch.tensor((-self.mean / self.std).reshape(1, -1, 1, 1)).float()
        self.IMAGE_BOUND_U = torch.tensor(((1 - self.mean) / self.std).reshape(1, -1, 1, 1)).float()

class MNIST_DSET():
    def __init__(self, dpath) -> None:
        self.IMAGE_SIZE = 28
        self.NUM_CLS = 10
        self.mean = 0.1307
        self.std = 0.3081
        self.num_channels = 1
        self.gen_transform = T.Compose(
            [
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        train_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        self.train_dset = MNIST(
            root=dpath, train=True, download=True, transform=train_transform
        )
        self.test_dset = MNIST(
            root=dpath, train=False, download=True, transform=test_transform
        )


def get_dataset(dname, dpath):
    dset_mapping = {"cifar10": CIFAR10_DSET,"mnist":MNIST_DSET}
    return dset_mapping[dname](dpath)


"""def get_noniid_dataset(dname, dpath, num_users, n_class, nsamples, rate_unbalance):
    obj = get_dataset(dname, dpath)
    # Chose euqal splits for every user
    if dname == "cifar10":
        obj.user_groups_train, obj.user_groups_test = cifar_extr_noniid(obj.train_dset, obj.test_dset,
                                                                        num_users, n_class, nsamples,
                                                                        rate_unbalance)
    return obj"""


def non_iid_unbalanced_dataidx_map(dset_obj, n_parties, beta=0.4):
    train_dset = dset_obj.train_dset
    n_classes = dset_obj.NUM_CLS
    
    N = len(train_dset)
    labels = np.array(train_dset.targets)
    min_size = 0
    min_require_size = 10
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(n_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    net_dataidx_map = {}        
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map

        
def non_iid_balanced(dset_obj, n_client, n_data_per_clnt, alpha=0.4):
    
    trn_y = np.array(dset_obj.train_dset.targets)
    trn_x = np.array(dset_obj.train_dset.data)
    n_cls = dset_obj.NUM_CLS
    height = width = dset_obj.IMAGE_SIZE
    channels = dset_obj.num_channels

    clnt_data_list = (np.ones(n_client) * n_data_per_clnt).astype(int)
    cls_priors   = np.random.dirichlet(alpha=[alpha]*n_cls,size=n_client)
    prior_cumsum = np.cumsum(cls_priors, axis=1)
    idx_list = [np.where(trn_y==i)[0] for i in range(n_cls)]
    cls_amount = [len(idx_list[i]) for i in range(n_cls)]

    clnt_x = [ np.zeros((clnt_data_list[clnt__], height, width, channels)).astype(np.float32) for clnt__ in range(n_client) ]
    clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(n_client) ]

    while(np.sum(clnt_data_list)!=0):
        curr_clnt = np.random.randint(n_client)
        # If current node is full resample a client
        # print('Remaining Data: %d' %np.sum(clnt_data_list))
        if clnt_data_list[curr_clnt] <= 0:
            continue
        clnt_data_list[curr_clnt] -= 1
        curr_prior = prior_cumsum[curr_clnt]
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if trn_y is out of that class
            if cls_amount[cls_label] <= 0:
                continue
            cls_amount[cls_label] -= 1
            clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
            clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

            break

    clnt_x = np.asarray(clnt_x)
    clnt_y = np.asarray(clnt_y)
    
    return clnt_x, clnt_y

    
    
def non_iid_labels(train_dataset, samples_per_client, classes):
    print(classes)
    all_data=Subset(train_dataset,[i for i,(x, y) in enumerate(train_dataset) if y in classes])
    perm=torch.randperm(len(all_data))
    return Subset(all_data,perm[:samples_per_client])



