
feddream_fast = {
    # adaptive_distill_start_round: 30 also works fine
    # need not wait for gen warmup, can start client training immediately
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
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [4], "node_4": [5]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    "heterogeneous_models": True, "models": {"0": "resnet18", "1": "resnet34", "2": "vgg11", "3": "wrn16_1", "4": "wrn40_1"},
    
    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],
    # Params for gradient descent on data
    "global_steps": 1, "local_steps": 2, "nx_samples": 2, 
    # for local training
    "distill_batch_size": 256, "distill_epochs": 100, "dset_size": 10*256, 
    "warmup": 20, "local_train_freq": 5,

    # adaptive distillation parameters
    "adaptive_server": True,  "adaptive_distill_start_round": 10, 

    #fast-meta deepinversion parameters
    "lr_z": 0.0015, "lr_g": 5e-3,
    "adv": 1.33, "bn": 10, "oh": 0.5, "bn_mmt": 0.9,
    "reset_bn": 0, "reset_l0": 1,"ismaml": 0, "optimizer_type": "avg",

    "log_console": True, "log_tb_freq": 1, 
    "exp_keys": ["local_steps", "nx_samples", "optimizer_type", "heterogeneous_models"]
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
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [2], "node_3": [2], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

independent_dreams = {
    "algo": "dare",
    "exp_id": "dare",
    "exp_type": "iid_clients_independent_dreams",
    "load_existing": False,
    "checkpoint_paths": {"1": "expt_dump/iid_clients_distill_reps_cifar10_1clients_25000samples_distadam_num_clients_1_samples_per_client_25000_distill_epochs_10_global_steps_2000_local_steps_0_warmup_250_seed2/saved_models/user1.pt"},
    "dset": "pathmnist",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs",
    "seed": 3,
    # Learning setup
    "num_clients": 1, "top_k": 1, "samples_per_client": 1000,
    # "device_ids": {"node_0": [4], "node_1": [7], "node_2": [6], "node_3": [1], "node_4": [2], "node_5": [3]},
    "device_ids": {"node_0": [0], "node_1": [1]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself

    "epochs": 500, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256,

    # params for model

    "position": 0, "inp_shape": [0, 3, 28, 28],
    "method": "orig", "ismaml": 0,
    # "position": 4, "inp_shape": [0, 256, 8, 8], "out_shape": [0, 256, 8, 8],
    
    # "method": "orig", "ismaml": 1,
    # "lr_g": 5e-3, "lr_z": 0.015,
    # "position": 0, "inp_shape": [0, 256], "out_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    "inversion_algo": "send_reps",
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 200,
    "first_time_steps": 2000,

    "exp_keys": ["distill_epochs", "steps", "position", "warmup"]
}

collab_dreams = {
    "algo": "distill_reps",
    "exp_id": "distadam",
    "exp_type": "iid_clients_collab",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/iid/",
    "dpath": "./imgs/",
    "seed": 2,
    # Learning setup
    "num_clients": 2, "top_k": 1, "samples_per_client": 1000,
    # "device_ids": {"node_0": [0], "node_1": [1,2], "node_2": [2,3], "node_3": [3], "node_4": [0], "node_5": [1]},
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself

    "epochs": 500, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256,

    # params for model
    # "position": 0, "inp_shape": [0, 3, 28, 28], "method": "orig",
    "position": 4, "inp_shape": [0, 256, 8, 8], "method": "orig",

    # Params for gradient descent on data
    # "inversion_algo": "send_model_centralized",
    "inversion_algo": "send_grads",
    "fedadam": False, "distadam": True,
    "server_lr": 0.01, "server_beta_1": 0.9, "server_beta_2": 0.99, "server_tau": 1e-9,
    "global_steps": 2000, "local_steps": 0, "local_lr": 1e-3,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 256, "distill_epochs": 20, "warmup": 20, "lambda": 0.1,
    "first_time_steps": 100, "adaptive_distill": True, "append_dataset": True,

    "exp_keys": ["adaptive_distill", "append_dataset", "distill_epochs", "global_steps", "local_steps","warmup", "position"]
}

iid_clients_distill_distadam_pathmnist = {
    "algo": "distill_reps",
    "exp_id": "distadam",
    "exp_type": "iid_clients_collab",
    "load_existing": True,
    "checkpoint_paths": {},
    "dset": "pathmnist",
    "dump_dir": "./expt_dump/pathmnist/iid/",
    "dpath": "./imgs/",
    "seed": 2,
    # Learning setup
    "num_clients": 5, "top_k": 1, "samples_per_client": 12000,
    # "device_ids": {"node_0": [6,1,4,2,5], "node_1": [6], "node_2": [1], "node_3": [4], "node_4": [2], "node_5": [5]},
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [3], "node_4": [0], "node_5": [1,2]},
    # "device_ids": {"node_0": [0], "node_1": [1]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself

    "epochs": 2000, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 28, 28], "method": "orig",
    "log_tb_freq": 10, 

    # Params for gradient descent on data
    # "inversion_algo": "send_model_centralized",
    "inversion_algo": "send_grads",
    "fedadam": False, "distadam": True,
    "server_lr": 0.01, "server_beta_1": 0.9, "server_beta_2": 0.99, "server_tau": 1e-9,
    "global_steps": 2000, "local_steps": 0, "local_lr": 1e-3,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 200,
    "first_time_steps": 100, "adaptive_distill": False,

    "exp_keys": ["distill_epochs", "global_steps", "local_steps","position"]
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
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [2], "node_3": [2], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 1000,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "seed"]
}

centralized = {
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
    "exp_keys": []
}

# current_config = independent_dreams
# current_config = collab_dreams
# current_config = feddream
current_config = feddream_fast
# current_config = fl
# current_config = isolated
# current_config = centralized
