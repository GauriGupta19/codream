feddream_fast = {
    "seed": 4,
    "algo": "feddream_fast",
    "exp_id": "fast",
    "exp_type": "iid_clients_feddream_fast",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "device_ids": {"node_0": [1], "node_1": [2], "node_2": [3], "node_3": [6], "node_4": [7]},
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
    "exp_keys": ["local_steps", "nx_samples", "optimizer_type"]
}

feddream = {
    # adaptive_distill_start_round: 30 also works fine
    # need not wait for gen warmup, can start client training immediately
    "seed": 4,
    "algo": "feddream",
    "exp_id": "",
    "exp_type": "iid_clients_feddream",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 2, "samples_per_client": 1000,
    "device_ids": {"node_0": [1], "node_1": [2], "node_2": [1]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg8"},
    
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
    "exp_keys": []
}

fl = {
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
    "device_ids": {"node_0": [3], "node_1": [6], "node_2": [2], "node_3": [4], "node_4": [5]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo"]
}

avgkd = {
    "algo": "avgkd",
    "exp_id": 10,
    "exp_type": "iid_clients_avgkd",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 50,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [1], "node_4": [1]},
    # communication epochs = 20 and local runs = 20 as per paper on AvgKD
    "epochs": 400, "local_runs": 50,
    # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},
    "model": "resnet18", "model_lr": 0.01, "batch_size": 256,
    "exp_keys": ["algo"]
}

fedprox = {
    "algo": "fedprox",
    "exp_id": 10,
    "exp_type": "iid_clients_fedprox",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/iid/",
    "dpath": "./imgs/mnist",
    "seed": 3,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    # Learning setup
    "num_clients": 4, "samples_per_client": 50,
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [1], "node_4": [0]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

moon = {
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
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [1], "node_4": [2]},
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo"]
}

scaffold = {
    "algo": "scaffold",
    "exp_id": 1,
    "exp_type": "iid_clients_scaffold",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": {"node_0": [7], "node_1": [6], "node_2": [7], "node_3": [5], "node_4": [4]},
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "local_runs": 1,
    "epochs": 400, "model": "resnet18",
    "lr_client": 0.1, "batch_size": 256,
    "lr_server": 1.,
    "model_lr": 0.1, # decoy parameter not used in scaffold
    "exp_keys": []
}

fedgen = {
    "algo": "fedgen",
    "exp_id": "test",
    "exp_type": "fedgen",
    "exp_type": "iid_clients_fedgen",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    "heterogeneous_models": False,
    "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1",},
    "device_ids": {"node_0": [1], "node_1": [2], "node_2": [3], "node_3": [4], "node_4": [2],},
    # Learning setup
    "num_clients": 4,
    "samples_per_client": 1000,
    # "alpha": 0.1,
    # "class_per_client": 2,
    "epochs": 400,
    "local_runs": 5,
    "model": "resnet18",
    "model_lr": 0.005,
    "batch_size": 256,
    "exp_keys": ["algo"],
}

isolated = {
    "algo": "isolated",
    "exp_id": 10,
    "exp_type": "iid_clients_isolated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/cifar10",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [6], "node_1": [6], "node_2": [6], "node_3": [7], "node_4": [7]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

centralized = {
    "algo": "centralized",
    "exp_id": 6,
    "exp_type": "iid_clients_centralized",
    "dset": "svhn",
    "dump_dir": "./expt_dump/svhn/iid/",
    "dpath": "./imgs/svhn",
    "seed": 2,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [3]},

    # Learning setup
    "num_clients": 1, "samples_per_client": 4000,
    "epochs": 400,
    "model": "resnet18",
    "model_lr": 0.1, "batch_size": 256,
    "exp_keys": []
}


# current_config = feddream
current_config = feddream_fast
# current_config = fl
# current_config = avgkd
# current_config = fedprox
# current_config = moon
# current_config = isolated
# current_config = centralized
# current_confg = fedgen
# current_config = scaffold
