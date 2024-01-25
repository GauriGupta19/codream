fl_svhn = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "iid_clients_fl",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/iid/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [1], "node_4": [1]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

fl_cifar10 = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "iid_clients_fl",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [1], "node_4": [1]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

fl_mnist = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "iid_clients_fl",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 50,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [1], "node_4": [1]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

fedprox_svhn = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "iid_clients_fedprox",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/iid/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [3], "node_1": [3], "node_2": [3], "node_3": [2], "node_4": [2]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

fedprox_cifar10 = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "iid_clients_fedprox",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [3], "node_1": [3], "node_2": [3], "node_3": [2], "node_4": [2]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

fedprox_mnist = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "iid_clients_fedprox",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 50,
    "device_ids": {"node_0": [3], "node_1": [3], "node_2": [3], "node_3": [2], "node_4": [2]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

moon_svhn = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "iid_clients_moon",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/iid/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [1], "node_4": [0]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

moon_cifar10 = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "iid_clients_moon",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [1], "node_4": [0]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

moon_mnist = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "iid_clients_moon",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 50,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [1], "node_4": [0]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

isolated_svhn = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "iid_clients_isolated",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/iid/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [0], "node_2": [1], "node_3": [2], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

isolated_cifar10 = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "iid_clients_isolated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [0], "node_2": [1], "node_3": [2], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

isolated_mnist = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "iid_clients_isolated",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [0], "node_2": [1], "node_3": [2], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

centralized_svhn = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "iid_clients_centralized",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/iid/",
    "dpath": "./imgs/svhn",
    "seed": 4,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
    "num_clients": 1, "samples_per_client": 4000,
    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["seed"]
}

centralized_cifar10 = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "iid_clients_centralized",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
    "num_clients": 1, "samples_per_client": 4000,
    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["seed"]
}

centralized_mnist = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "iid_clients_centralized",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1]},

    # Learning setup
    "num_clients": 1, "samples_per_client": 200,
    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["seed"]
}

feddream_fast_cifar10 = {
    # adaptive_distill_start_round: 30 also works fine
    # need not wait for gen warmup, can start client training immediately
    "seed": 4,
    "algo": "feddream_fast",
    "exp_id": "fast",
    "exp_type": "iid_clients_feddream_fast_entropy",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [4], "node_1": [2], "node_2": [3], "node_3": [6], "node_4": [7]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},
    
    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],
    # Params for gradient descent on data
    "global_steps": 1, "local_steps": 5, "nx_samples": 5, 
    # for local training
    "distill_batch_size": 256, "distill_epochs": 100, "dset_size": 25*256, 
    "warmup": 20, "local_train_freq": 5,

    # adaptive distillation parameters
    "adaptive_server": True,  "adaptive_distill_start_round": 10, 

    #fast-meta deepinversion parameters
    "lr_z": 0.0015, "lr_g": 5e-3,
    "adv": 1.33, "bn": 10, "oh": 0.5, "bn_mmt": 0.9,
    "reset_bn": 0, "reset_l0": 1,"ismaml": 0, "optimizer_type": "avg",

    "log_console": True, "log_tb_freq": 1, 
    "exp_keys": ["warmup", "local_steps", "nx_samples", "seed"]
}

feddream_fast_svhn = {
    # adaptive_distill_start_round: 30 also works fine
    # need not wait for gen warmup, can start client training immediately
    "seed": 4,
    "algo": "feddream_fast",
    "exp_id": "fast",
    "exp_type": "iid_clients_feddream_fast_entropy",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/iid/",
    "dpath": "./imgs/svhn",
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [4], "node_1": [2], "node_2": [3], "node_3": [6], "node_4": [7]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},
    
    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],
    # Params for gradient descent on data
    "global_steps": 1, "local_steps": 5, "nx_samples": 5, 
    # for local training
    "distill_batch_size": 256, "distill_epochs": 100, "dset_size": 25*256, 
    "warmup": 20, "local_train_freq": 5,

    # adaptive distillation parameters
    "adaptive_server": True,  "adaptive_distill_start_round": 10, 

    #fast-meta deepinversion parameters
    "lr_z": 0.0015, "lr_g": 5e-3,
    "adv": 1.33, "bn": 10, "oh": 0.5, "bn_mmt": 0.9,
    "reset_bn": 0, "reset_l0": 1,"ismaml": 0, "optimizer_type": "avg",

    "log_console": True, "log_tb_freq": 1, 
    "exp_keys": ["warmup", "local_steps", "nx_samples", "seed"]
}

feddream_fast_mnist = {
    # adaptive_distill_start_round: 30 also works fine
    # need not wait for gen warmup, can start client training immediately
    "seed": 4,
    "algo": "feddream_fast",
    "exp_id": "fast",
    "exp_type": "iid_clients_feddream_fast_entropy",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    # Learning setup
    "num_clients": 4, "samples_per_client": 50,
    "device_ids": {"node_0": [4], "node_1": [2], "node_2": [3], "node_3": [6], "node_4": [7]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},
    
    # params for model
    "position": 0, "inp_shape": [0, 1, 28, 28],
    # Params for gradient descent on data
    "global_steps": 1, "local_steps": 5, "nx_samples": 5, 
    # for local training
    "distill_batch_size": 256, "distill_epochs": 100, "dset_size": 25*256, 
    "warmup": 50, "local_train_freq": 5,

    # adaptive distillation parameters
    "adaptive_server": True,  "adaptive_distill_start_round": 10, 

    #fast-meta deepinversion parameters
    "lr_z": 0.0015, "lr_g": 5e-3,
    "adv": 1.33, "bn": 10, "oh": 0.5, "bn_mmt": 0.9,
    "reset_bn": 0, "reset_l0": 1,"ismaml": 0, "optimizer_type": "avg",

    "log_console": True, "log_tb_freq": 1, 
    "exp_keys": ["warmup", "local_steps", "nx_samples", "seed"]
}


# current_config = fl_svhn
# current_config = feddream_fast
# current_config = fl
# current_config = fedprox
# current_config = moon
# current_config = isolated
# current_config = centralized