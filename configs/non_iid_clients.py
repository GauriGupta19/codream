feddream_fast = {
    "seed": 1,
    "algo": "feddream_fast",
    "exp_id": "distadam",
    "exp_type": "non_iid_balanced_clients_feddream_fast",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_0.1/",
    "dpath": "./imgs/mnist",
    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "alpha": 0.1,
    # "samples_per_label":1000,
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [4], "node_3": [5], "node_4": [3]},
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
    "exp_keys": ["alpha", "local_steps", "warmup", "nx_samples", "dset_size"]
}

feddream = {
    "seed": 4,
    "algo": "feddream",
    "exp_id": "distadam",
    "exp_type": "non_iid_balanced_clients_feddream",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "alpha": 0.1,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [3], "node_3": [3], "node_4": [4]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],
    # Params for gradient descent on data
    "data_lr": 0.05, "global_steps": 2000, "local_steps": 1,
    "alpha_preds": 1, "alpha_tv": 2.5e-5, "alpha_l2": 3e-8, "alpha_f": 10,
    # for local training
    "distill_batch_size": 256, "distill_epochs": 100, "dset_size":10*256, 
    "warmup": 20, "local_train_freq": 5,

    # adaptive distillation parameters
    "adaptive_server": True, "lambda_server": 10, "adaptive_distill_start_round": 10, 
    "log_console": True, "log_tb_freq": 1, 
    "exp_keys": ["alpha"]
}

fl = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 8,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [1], "node_4": [1]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["alpha"]
}

avgkd = {
    "algo": "avgkd",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_avgkd",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_0.1/",
    "dpath": "./imgs/svhn",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [1], "node_4": [1]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "epochs": 400, "local_runs": 20,
    # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},
    "model": "resnet18", "model_lr": 0.01, "batch_size": 256,
    "exp_keys": ["alpha"]
}

fedprox = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fedprox",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 9,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [4], "node_1": [4], "node_2": [5], "node_3": [5], "node_4": [4]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["alpha"]
}

moon = {
    "algo": "moon",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_moon",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [2], "node_2": [3], "node_3": [3], "node_4": [2]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["alpha"]
}

scaffold = {
    "algo": "scaffold",
    "exp_id": 1,
    "exp_type": "non_iid_balanced_clients_scaffold",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    "device_ids": {"node_0": [2], "node_1": [4], "node_2": [3], "node_3": [2], "node_4": [1]},
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "local_runs": 5,
    "epochs": 400, "model": "resnet18",
    "lr_client": 0.1, "batch_size": 256,
    "lr_server": 1.,
    "model_lr": 0.1, # decoy parameter not used in scaffold
    "exp_keys": ["alpha", "local_runs"]
}

fedgen = {
    "algo": "fedgen",
    "exp_id": "test",
    "exp_type": "fedgen",
    "exp_type": "non_iid_balanced_clients_fedgen",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    "heterogeneous_models": False,
    "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1",},
    "seed": 4,
    "device_ids": {"node_0": [1], "node_1": [0], "node_2": [2], "node_3": [3], "node_4": [3],},
    # Learning setup
    "num_clients": 4,
    "samples_per_client": 1000,
    "alpha": 0.1,
    # "class_per_client": 2,
    "epochs": 400,
    "local_runs": 5,
    "model": "resnet18",
    "model_lr": 0.005,
    "batch_size": 256,
    "exp_keys": ["alpha"],
}

isolated = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_isolated",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [2], "node_1": [3], "node_2": [4], "node_3": [5], "node_4": [2]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["alpha"]
}

centralized = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "non_iid_balanced_clients_centralized",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/alpha_1/",
    "dpath": "./imgs/svhn",
    "seed": 3,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [3]},

    # Learning setup

    "num_clients": 1, "samples_per_client": 1000,
    "client_data_units": 4, "alpha": 0.1,

    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": ["client_data_units", "epochs"]
}


# current_config = feddream
current_config = feddream_fast
# current_config =  fl
# current_config =  avgkd
# current_config = fedprox
# current_config = moon
# current_config = isolated
# current_config = centralized
# current_config = scaffold
# current_config = fedgen
