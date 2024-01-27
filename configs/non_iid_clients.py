feddream_fast = {
    "seed": 4,
    "algo": "feddream_fast",
    "exp_id": "distadam",
    "exp_type": "non_iid_balanced_clients_feddream_fast_wo_adaptive",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 1,
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [5], "node_3": [6], "node_4": [3]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],
    # Params for gradient descent on data
    "global_steps": 1, "local_steps": 5, "nx_samples": 2, 
    # for local training
    "distill_batch_size": 256, "distill_epochs": 100, "dset_size": 10*256, 
    "warmup": 20, "local_train_freq": 5,

    # adaptive distillation parameters
    "adaptive_server": True,  "adaptive_distill_start_round": 10, 

    #fast-meta deepinversion parameters
    "lr_z": 0.0015, "lr_g": 5e-3,
    "adv": 0, "bn": 10, "oh": 0.5, "bn_mmt": 0.9,
    "reset_bn": 0, "reset_l0": 1,"ismaml": 0, "optimizer_type": "avg",

    "log_console": True, "log_tb_freq": 1, 
    "exp_keys": ["alpha", "local_steps", "warmup", "nx_samples"]
}

feddream_fast_indp = {
    # adaptive_distill_start_round: 30 also works fine
    # need not wait for gen warmup, can start client training immediately
    "seed": 4,
    "algo": "feddream_fast_indp",
    "exp_id": "fast",
    "exp_type": "non_iid_balanced_clients_feddream_fast_independent",
    "load_existing": False,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "alpha": 0.1,
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [4], "node_4": [5]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "resnet34", "2": "vgg11", "3": "wrn16_1", "4": "wrn40_1"},
    
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
    "reset_bn": 0, "reset_l0": 1,"ismaml": 0,

    "log_console": True, "log_tb_freq": 1, 
    "exp_keys": ["local_steps", "alpha", "nx_samples"]
}

feddream_fast_noniid = {
    "seed": 4,
    "algo": "feddream_fast_noniid",
    "exp_id": "distadam",
    "exp_type": "non_iid_balanced_labels_fast_noniid",
    "load_existing": False,
    "checkpoint_paths": {
        "1": "expt_dump/cifar10/alpha_0.1/non_iid_balanced_clients_feddream_fast_noniid_pretrained_cifar10_4clients_1000samples_distadam_alpha_0.1_local_steps_1_nx_samples_2_optimizer_type_avg_dset_size_2560_seed4/saved_models/user1.pt",
        "2": "expt_dump/cifar10/alpha_0.1/non_iid_balanced_clients_feddream_fast_noniid_pretrained_cifar10_4clients_1000samples_distadam_alpha_0.1_local_steps_1_nx_samples_2_optimizer_type_avg_dset_size_2560_seed4/saved_models/user2.pt",
        "3": "expt_dump/cifar10/alpha_0.1/non_iid_balanced_clients_feddream_fast_noniid_pretrained_cifar10_4clients_1000samples_distadam_alpha_0.1_local_steps_1_nx_samples_2_optimizer_type_avg_dset_size_2560_seed4/saved_models/user3.pt",
        "4": "expt_dump/cifar10/alpha_0.1/non_iid_balanced_clients_feddream_fast_noniid_pretrained_cifar10_4clients_1000samples_distadam_alpha_0.1_local_steps_1_nx_samples_2_optimizer_type_avg_dset_size_2560_seed4/saved_models/user4.pt",

    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
    "device_ids": {"node_0": [3], "node_1": [7], "node_2": [6], "node_3": [4], "node_4": [5]},
    "epochs": 400, "model": "resnet18",
    "model_lr": 0.2, "batch_size": 256,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "resnet34", "2": "vgg11", "3": "wrn16_1", "4": "wrn40_1"},

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
    "exp_keys": ["alpha", "local_steps", "nx_samples", "optimizer_type", "dset_size", "samples_per_label"]
}

non_iid_labels_clients_independent_dreams = {
    "algo": "dare",
    "exp_id": 1,
    "exp_type": "non_iid_labels_clients_independent_dreams",
    "load_existing": False, "start_epoch": 0,
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs",
    "seed": 2,
    # Learning setup
    "num_clients": 5, "top_k": 1, "samples_per_client": 12000, "class_per_client": 2,
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [4], "node_3": [6], "node_4": [7], "node_5": [5]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    
    "epochs": 1000, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256, 
    
    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32], "method": "orig",

    # Params for gradient descent on data
    "inversion_algo": "send_reps",
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 150,
    "first_time_steps": 2000,
    
        "exp_keys": ["class_per_client", "steps", "position", "warmup", "inversion_algo"]
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

non_iid_labels_clients_collab = {
    "algo": "distill_reps",
    "exp_id": "distadam",
    "exp_type": "non_iid_labels_clients_collab",
    "load_existing": True, "start_epoch": 0,
    "checkpoint_paths": {},
    "dset": "cifar100",
    "dump_dir": "./expt_dump/cifar100/alpha_1/",
    "dpath": "./imgs",
    "seed": 2,
    # Learning setup
    "num_clients": 5, "top_k": 1, "samples_per_client": 12000, "class_per_client": 20,
    "device_ids": {"node_0": [6], "node_1": [1], "node_2": [2], "node_3": [3], "node_4": [4], "node_5": [5]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    
    "epochs": 1000, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256, 
    
    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32], "method": "orig",

    # Params for gradient descent on data
    "inversion_algo": "send_grads",
    "fedadam": False, "distadam": True,
    "server_lr": 0.01, "server_beta_1": 0.9, "server_beta_2": 0.99, "server_tau": 1e-9,
    "global_steps": 2000, "local_steps": 0, "local_lr": 1e-3,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 150,
    "first_time_steps": 2000,
    
    "exp_keys": ["class_per_client", "steps", "position", "warmup", "inversion_algo"]
}

non_iid_balanced_clients_independent = {
    "algo": "dare",
    "exp_id": 1,
    "exp_type": "non_iid_balanced_clients_independent",
    "load_existing": False, "start_epoch": 0,
    "dset": "cifar100",
    "dump_dir": "./expt_dump/cifar100/alpha_1/",
    "dpath": "./imgs",
    "seed": 2,
    # Learning setup
    "num_clients": 4, "top_k": 1, "samples_per_client": 10000, "alpha": 0.1,
    "device_ids": {"node_0": [0], "node_1": [4], "node_2": [4], "node_3": [0], "node_4": [5]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    
    "epochs": 1000, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256, 
    
    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32], "method": "orig",

    # Params for gradient descent on data
    "inversion_algo": "send_reps",
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 150,
    "first_time_steps": 2000,
    
    "exp_keys": ["alpha", "steps", "position", "warmup", "inversion_algo"]
}

non_iid_balanced_clients_collab = {
    "algo": "distill_reps",
    "exp_id": "distadam",
    "exp_type": "non_iid_balanced_clients_collab",
    "load_existing": False, "start_epoch": 0,
    "checkpoint_paths": {},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/cifar10/alpha_1/",
    "dpath": "./imgs",
    "seed": 4,
    # Learning setup
    "num_clients": 2, "top_k": 1, "samples_per_client":4000, "alpha":1,
    "device_ids": {"node_0": [1], "node_1": [2], "node_2": [3]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    
    "epochs": 500, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256, 
    
    # params for model
    # "position": 0, "inp_shape": [0, 3, 32, 32], "method": "orig",
    "position": 4, "inp_shape": [0, 256, 8, 8], "method": "orig",

    # Params for gradient descent on data
    "inversion_algo": "send_grads",
    "fedadam": False, "distadam": True,
    "server_lr": 0.01, "server_beta_1": 0.9, "server_beta_2": 0.99, "server_tau": 1e-9,
    "global_steps": 2000, "local_steps": 0, "local_lr": 1e-3,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 200,
    "first_time_steps": 2000, "adaptive_distill": True,
    
    "exp_keys": ["alpha", "adaptive_distill", "distill_epochs", "global_steps", "position", "warmup"]
}

fl = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_fl",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [1], "node_1": [1], "node_2": [1], "node_3": [0], "node_4": [0]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha"]
}

# avgkd = {
#     "algo": "avgkd",
#     "exp_id": 10,
#     "exp_type": "non_iid_balanced_clients_avgkd_seed4",
#     "dset": "cifar10",
#     "dump_dir": "./expt_dump/cifar10/alpha_0.1/",
#     "dpath": "./imgs/cifar10",
#     "seed": 4,
#     # server can have overlapping device ids with clients because
#     # both are not used at the same time
#     "device_ids": {"node_0": [6], "node_1": [6], "node_2": [6], "node_3": [7], "node_4": [7]},

#     # Learning setup
#     "num_clients": 4, "samples_per_client": 1000, "samples_per_label":400, "alpha": 0.1,
#     "epochs": 400, "local_runs": 20,
#     # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},
#     "model": "resnet18", "model_lr": 0.01, "batch_size": 256,
#     "exp_keys": ["algo", "alpha"]
# }

avgkd = {
    "algo": "avgkd",
    "exp_id": 10,
    "exp_type": "non_iid_balanced_clients_avgkd_seed4",
    "dset": "mnist",
    "dump_dir": "./expt_dump/mnist/alpha_1/",
    "dpath": "./imgs/mnist",
    "seed": 4,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [1], "node_4": [1]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "samples_per_label":400, "alpha": 1,
    "epochs": 400, "local_runs": 50,
    # "heterogeneous_models": True, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},
    "model": "resnet18", "model_lr": 0.01, "batch_size": 256,
    "exp_keys": ["algo", "alpha"]
}

fedprox = {
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
    "exp_keys": ["algo", "alpha"]
}

moon = {
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
    "exp_keys": ["algo", "alpha"]
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
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [0], "node_4": [3]},

    # Learning setup
    "num_clients": 4, "samples_per_client": 50, "alpha": 1,
    "epochs": 400, "local_runs": 5,
    # optional for het models
    "heterogeneous_models": False, "models": {"0": "resnet18", "1": "wrn16_1", "2": "vgg11", "3": "resnet34", "4": "wrn40_1"},

    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    "exp_keys": ["algo", "alpha"]
}

centralized = {
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

    "exp_keys": ["alpha", "client_data_units", "epochs"]
}

# current_config = non_iid_labels_clients_independent
# current_config = non_iid_labels_clients_collab
# current_config = non_iid_balanced_clients_independent
# current_config = non_iid_balanced_clients_collab
# current_config = feddream
# current_config =  feddream_fast
# current_config = feddream_fast_indp
# current_config = feddream_fast_noniid
# current_config =  fl
current_config =  avgkd
# current_config = fedprox
# current_config = moon
# current_config = isolated
# current_config = centralized