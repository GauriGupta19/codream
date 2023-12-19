import pdb
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets import MNIST
import medmnist
from torch.utils.data import Subset, Dataset
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, config, transform = None, buffer_size=1*256):
        self.samples = []
        self.max_size = buffer_size
        ld_path = config.get("load_data_path", None)
        self.transform = transform
        if ld_path is not None:
            filepaths = ld_path + "/*.pt"
            filepaths = glob(filepaths)
            print(f"found {len(filepaths)} batches in data checkpoints")
            for fp in filepaths:
                samples = torch.load(fp)
                self.samples.append(samples)
        
    def __getitem__(self, index):
        sample = self.samples[index]
        img = sample[0]
        if self.transform is not None:
            img = self.transform(img)
        return img.unsqueeze(0), sample[1].unsqueeze(0)
 
    def __len__(self):
        return len(self.samples)

    def append(self, sample):
        # Add a new sample to the datasret
        # self.samples.append(sample)
        list_of_tuples = list(zip(sample[0], sample[1]))
        self.samples += list_of_tuples
        if len(self.samples) > self.max_size:
            self.samples = self.samples[-self.max_size:]

    def reset(self):
       self.samples = [] 
        
class PathMNIST_DSET():
    def __init__(self, dpath) -> None:
        dpath = "./imgs/"
        self.IMAGE_SIZE = 28
        self.NUM_CLS = 9
        self.mean = np.array([0.5])
        self.std = np.array([0.5])
        data_flag = 'pathmnist'
        info = medmnist.INFO[data_flag]
        self.num_channels = info['n_channels']
        self.data_class = getattr(medmnist, info['python_class'])
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])
        self.train_dset = self.data_class(root=dpath, split='train', transform=transform, download=True)
        self.test_dset = self.data_class(root=dpath, split='test', transform=transform, download=True)

class CIFAR100_DSET():
    def __init__(self, dset) -> None:
        dpath = "./imgs/"
        self.IMAGE_SIZE = 32
        self.NUM_CLS = 100
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.mean = np.array(CIFAR100_TRAIN_MEAN)
        self.std = np.array(CIFAR100_TRAIN_STD)
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
        self.train_transform = T.Compose(
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
        self.train_dset = CIFAR100(
            root=dpath, train=True, download=True, transform=self.train_transform
        )
        self.test_dset = CIFAR100(
            root=dpath, train=False, download=True, transform=test_transform
        )
        self.IMAGE_BOUND_L = torch.tensor((-self.mean / self.std).reshape(1, -1, 1, 1)).float()
        self.IMAGE_BOUND_U = torch.tensor(((1 - self.mean) / self.std).reshape(1, -1, 1, 1)).float()

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
        self.train_transform = T.Compose(
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
            root=dpath, train=True, download=True, transform=self.train_transform
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
        self.train_transform = T.Compose(
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
            root=dpath, train=True, download=True, transform=self.train_transform
        )
        self.test_dset = MNIST(
            root=dpath, train=False, download=True, transform=test_transform
        )

        
def get_dataset(dname, dpath):
    dset_mapping = {"cifar10": CIFAR10_DSET,
                    "mnist": MNIST_DSET,
                    "cifar100": CIFAR100_DSET,
                    "pathmnist": PathMNIST_DSET}
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
    indices = [ [-1]*clnt_data_list[clnt__] for clnt__ in range(n_client) ]
    # clnt_x = [ np.zeros((clnt_data_list[clnt__], height, width, channels)).astype(np.float32) for clnt__ in range(n_client) ]
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
            
            indices[curr_clnt][clnt_data_list[curr_clnt]] = idx_list[cls_label][cls_amount[cls_label]]
            # clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
            clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]
            break
    # clnt_x = np.asarray(clnt_x)
    clnt_y = np.asarray(clnt_y)
    return indices, clnt_y

def plot_training_distribution(indices, clnt_y, n_client, n_cls, path):
    # indices, clnt_y = non_iid_balanced(dset_obj, n_client, n_data_per_clnt, alpha)         
    x = [[i]*n_cls for i in range(n_client)]
    y = [np.arange(n_cls) for i in range(n_client)]
    s = []
    for clt in range(n_client):
        labels = [np.where(clnt_y[clt]==i)[0].shape[0] for i in range(n_cls)]
        s.append(np.array(labels))
    
    plt.scatter(x, y, s = s)
    plt.title('Training label distribution')
    plt.xlabel('Client id')
    plt.ylabel('Training labels')
    plt.savefig(path + f"noniid_data.png", bbox_inches='tight')
    torch.save(s, path + f"size_labels.pt")    
    
def non_iid_labels(train_dataset, samples_per_client, classes):
    all_data=Subset(train_dataset,[i for i,(x, y) in enumerate(train_dataset) if y in classes])
    perm=torch.randperm(len(all_data))
    return Subset(all_data,perm[:samples_per_client])
