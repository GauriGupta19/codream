import pdb
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10


class CIFAR10_DSET():
    def __init__(self, dpath) -> None:
        self.IMAGE_SIZE = 32
        self.NUM_CLS = 10
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
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


def get_dataset(dname, dpath):
    dset_mapping = {"cifar10": CIFAR10_DSET}
    return dset_mapping[dname](dpath)


def get_noniid_dataset(dname, dpath, num_users, n_class, nsamples, rate_unbalance):
    obj = get_dataset(dname, dpath)
    # Chose euqal splits for every user
    if dname == "cifar10":
        obj.user_groups_train, obj.user_groups_test = cifar_extr_noniid(obj.train_dset, obj.test_dset,
                                                                        num_users, n_class, nsamples,
                                                                        rate_unbalance)
    return obj

def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # stores the image idxs with their corresponding labels
    # array([[    0,     1,     2, ..., 49997, 49998, 49999],
    #        [    6,     9,     9, ...,     9,     1,     1]])
    idxs_labels = np.vstack((idxs, labels))
    # sorts the whole thing based on labels
    # array([[29513, 16836, 32316, ..., 36910, 21518, 25648],
    #       [    0,     0,     0, ...,     9,     9,     9]])
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    # Same things as above except that it is test set now
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])


    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))

    return dict_users_train, dict_users_test