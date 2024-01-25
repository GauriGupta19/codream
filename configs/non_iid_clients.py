fl_mnist_1 = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fl_mnist_01 = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_0.1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fl_svhn_1 = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fl_svhn_01 = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_0.1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fl_cifar10_1 = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fl_cifar10_01 = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fedprox_svhn_01 = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fedprox",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_0.1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [0], "node_3": [0], "node_4": [0]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fedprox_svhn_1 = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fedprox",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [0], "node_3": [0], "node_4": [0]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fedprox_cifar10_01 = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fedprox",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [0], "node_3": [0], "node_4": [0]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fedprox_cifar10_1 = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fedprox",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [0], "node_3": [0], "node_4": [0]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fedprox_mnist_01 = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fedprox",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_0.1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [0], "node_3": [0], "node_4": [0]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

fedprox_mnist_1 = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fedprox",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [0], "node_3": [0], "node_4": [0]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

moon_svhn_1 = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_moon",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [3], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

moon_svhn_01 = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_moon",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_0.1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [3], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

moon_cifar10_01 = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_moon",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [3], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

moon_cifar10_1 = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_moon",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [3], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

moon_mnist_1 = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_moon",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [3], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

moon_mnist_01 = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_moon",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_0.1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [3], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

isolated_mnist_1 = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_isolated",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

isolated_mnist_01 = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_isolated",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_0.1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

isolated_cifar10_01 = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_isolated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

isolated_cifar10_1 = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_isolated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

isolated_svhn_1 = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_isolated",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

isolated_svhn_01 = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_isolated",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_0.1/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "alpha": 0.1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha", "seed"]
}

centralized_svhn_01 = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "non_iid_balanced_clients_centralized",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_0.1/",
    "dpath": "./imgs/svhn",
    "seed": 0,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
   "num_clients": 1, "samples_per_client": 1000,
    "client_data_units": 4, "alpha": 0.1,

    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": ["alpha", "client_data_units", "epochs", "seed"]
}

centralized_svhn_1 = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "non_iid_balanced_clients_centralized",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_1/",
    "dpath": "./imgs/svhn",
    "seed": 0,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
   "num_clients": 1, "samples_per_client": 1000,
    "client_data_units": 4, "alpha": 1,

    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": ["alpha", "client_data_units", "epochs", "seed"]
}

centralized_cifar10_01 = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "non_iid_balanced_clients_centralized",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    "seed": 0,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
   "num_clients": 1, "samples_per_client": 1000,
    "client_data_units": 4, "alpha": 0.1,

    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": ["alpha", "client_data_units", "epochs", "seed"]
}

centralized_cifar10_1 = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "non_iid_balanced_clients_centralized",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 0,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
   "num_clients": 1, "samples_per_client": 1000,
    "client_data_units": 4, "alpha": 1,

    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": ["alpha", "client_data_units", "epochs", "seed"]
}

centralized_mnist_01 = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "non_iid_balanced_clients_centralized",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_0.1/",
    "dpath": "./imgs/mnist",
    "seed": 0,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
   "num_clients": 1, "samples_per_client": 50,
    "client_data_units": 4, "alpha": 0.1,

    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": ["alpha", "client_data_units", "epochs", "seed"]
}

centralized_mnist_1 = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "non_iid_balanced_clients_centralized",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 0,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
   "num_clients": 1, "samples_per_client": 50,
    "client_data_units": 4, "alpha": 1,

    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": ["alpha", "client_data_units", "epochs", "seed"]
}

# current_config = fl_mnist_1
# current_config =  feddream_fast
# current_config = feddream_fast_noniid
# current_config =  fl
# current_config = fedprox
# current_config = moon
# current_config = isolated
# current_config = centralized