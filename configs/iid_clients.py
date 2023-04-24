iid_clients_collab_new = {
    "algo": "dare",
    "exp_id": "mvt2_data_log2_more2",
    "exp_type": "iid_clients_collab_entropy",
    "load_existing": True,
    "checkpoint_paths": {"1": "expt_dump/iid_clients_collab_entropy_cifar10_5clients_5000samples_mvt2_distill_epochs_25_steps_2000_position_0_warmup_150_seed2/saved_models/user1.pt",
                         "2": "expt_dump/iid_clients_collab_entropy_cifar10_5clients_5000samples_mvt2_distill_epochs_25_steps_2000_position_0_warmup_150_seed2/saved_models/user2.pt",
                         "3": "expt_dump/iid_clients_collab_entropy_cifar10_5clients_5000samples_mvt2_distill_epochs_25_steps_2000_position_0_warmup_150_seed2/saved_models/user3.pt",
                         "4": "expt_dump/iid_clients_collab_entropy_cifar10_5clients_5000samples_mvt2_distill_epochs_25_steps_2000_position_0_warmup_150_seed2/saved_models/user4.pt",
                         "5": "expt_dump/iid_clients_collab_entropy_cifar10_5clients_5000samples_mvt2_distill_epochs_25_steps_2000_position_0_warmup_150_seed2/saved_models/user5.pt"},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # Learning setup
    "num_clients": 5, "top_k": 1, "samples_per_client": 5000,
    "device_ids": {"node_0": [1], "node_1": [2], "node_2": [3], "node_3": [5], "node_4": [6], "node_5": [7]},
    # "device_ids": {"node_0": [], "node_1": [0], "node_2": [1]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself

    "epochs": 1000, "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    "inversion_algo": "send_model",
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 150,
    "first_time_steps": 2000,

    "exp_keys": ["distill_epochs", "steps", "position", "warmup"]
}

feddream = {
    "seed": 2,
    "algo": "feddream",
    "exp_id": "1",
    "exp_type": "iid_clients_feddream",
    "load_existing": True,
    "checkpoint_paths": {
        '1': './expt_dump/2k_clients/user1.pt',
        '2': './expt_dump/2k_clients/user2.pt',
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 2, "samples_per_client": 2000,
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [0]},
    "epochs": 1000, "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    "inversion_algo": "send_grads",
    "data_lr": 0.05, "global_steps": 2000, "local_steps": 1,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    # for local training
    "warmup": 100,
    "distill_batch_size": 128, "distill_epochs": 20,
    "local_train_freq": 5,
    
    # adaptive distillation parameters
    "adaptive_server": True,
    "adaptive_client": False,
    "lambda_server": 0.1,

    "exp_keys": ["adaptive_server"]
}

iid_clients_distill = {
    "algo": "distill_reps",
    "exp_id": "mvt2_data_log3_more",
    "exp_type": "iid_clients_distill_reps",
    "load_existing": True,
    "checkpoint_paths": {
'1': './expt_dump/iid_clients_distill_reps_cifar10_24clients_1000samples_mvt2_data_log3_distill_epochs_10_global_steps_2000_position_0_warmup_250_inversion_algo_send_grads_adaptive_distill_False_heterogeneous_models_False_seed2/saved_models/user13.pt',
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # Learning setup
    "num_clients": 24, "top_k": 1, "samples_per_client": 1000,
    # "device_ids": {"node_0": [6,1,4,2,5], "node_1": [6], "node_2": [1], "node_3": [4], "node_4": [2], "node_5": [5]},
    # "device_ids": {"node_0": [6,1,2,3,4,5,7,0], "node_1": [1], "node_2": [2], "node_3": [3], "node_4": [4], "node_5": [5], "node_6": [6], "node_7": [7], "node_8": [3]},
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [1], "node_3": [0], "node_4": [1], "node_5": [0], "node_6": [1], "node_7": [0], "node_8": [1], "node_9": [0], "node_10": [1], "node_11": [0], "node_12": [1],
                   "node_13": [0], "node_14": [0], "node_15": [1], "node_16": [0], "node_17": [1], "node_18": [0], "node_19": [1], "node_20": [0], "node_21": [1], "node_22": [0], "node_23": [1], "node_24": [0]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    "epochs": 1000, "model": "resnet34",
    "heterogeneous_models": False,
    "models": {
        "1": "resnet18",
        "2": "resnet34",
        "3": "resnet50",
        "4": "resnet101",
    },
    "model_lr": 3e-4, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    # "inversion_algo": "send_model_centralized",
    "inversion_algo": "send_grads",
    "data_lr": 0.05, "global_steps": 2000, "local_steps": 1,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 250,
    "first_time_steps": 2000, "adaptive_distill": False,
    
    "exp_keys": ["distill_epochs", "global_steps", "position", "warmup", "inversion_algo", "adaptive_distill", "heterogeneous_models"]
}


iid_clients_scaffold = {
    "algo": "scaffold",
    "exp_id": 1,
    "exp_type": "iid_clients_scaffold",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [0],},
    # Learning setup
    "num_clients": 2, "samples_per_client": 500,
    "epochs": 1000, "model": "resnet34",
    "lr_client": 0.1, "batch_size": 256,
    "lr_server": 1.,
    "exp_keys": []
}


iid_clients_isolated_new = {
    "algo": "isolated",
    "exp_id": 6,
    "exp_type": "iid_clients_isolated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [0,2]},

    # Learning setup
    "num_clients": 1, "samples_per_client": 50000,

    "epochs": 150,
    "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256,

    "exp_keys": []
}

iid_clients_federated_new = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "iid_clients_federated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0,1], "node_2": [2,3]},

    # Learning setup
    "num_clients": 2, "samples_per_client": 500,

    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,

    "exp_keys": []
}

iid_clients_distill_fedavg = {
    "algo": "distill_reps",
    "exp_id": "fedavg",
    "exp_type": "iid_clients_distill_reps",
    "load_existing": True,
    "checkpoint_paths": {
      "1": "expt_dump/4client_models/user1.pt",
      "2": "expt_dump/4client_models/user2.pt",
      "3": "expt_dump/4client_models/user3.pt",
      "4": "expt_dump/4client_models/user4.pt",
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # Learning setup
    "num_clients": 4, "top_k": 1, "samples_per_client": 6000,
    # "device_ids": {"node_0": [6,1,4,2,5], "node_1": [6], "node_2": [1], "node_3": [4], "node_4": [2], "node_5": [5]},
    # "device_ids": {"node_0": [6,1,2,3,4,5,7,0], "node_1": [1], "node_2": [2], "node_3": [3], "node_4": [4], "node_5": [5], "node_6": [6], "node_7": [7], "node_8": [3]},
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [2], "node_4": [3]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself

    "epochs": 1000, "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    # "inversion_algo": "send_model_centralized",
    "inversion_algo": "send_grads",
    "fedadam": False, "distadam": False,
    "server_lr": 0.01, "server_beta_1": 0.9, "server_beta_2": 0.99, "server_tau": 1e-9,
    "global_steps": 400, "local_steps": 0, "local_lr": 1e-3,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 250,
    "first_time_steps": 2000, "adaptive_distill": False,

    "exp_keys": ["distill_epochs", "global_steps", "local_steps", "position", "warmup", "inversion_algo", "adaptive_distill", "fedadam", "distadam"]
}

iid_clients_distill_fedavg = {
    "algo": "distill_reps",
    "exp_id": "distadam",
    "exp_type": "iid_clients_distill_reps",
    "load_existing": True,
    "checkpoint_paths": {
      "1": "expt_dump/4client_models/user1.pt",
      "2": "expt_dump/4client_models/user2.pt",
      "3": "expt_dump/4client_models/user3.pt",
      "4": "expt_dump/4client_models/user4.pt",
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # Learning setup
    "num_clients": 4, "top_k": 1, "samples_per_client": 6000,
    # "device_ids": {"node_0": [6,1,4,2,5], "node_1": [6], "node_2": [1], "node_3": [4], "node_4": [2], "node_5": [5]},
    # "device_ids": {"node_0": [6,1,2,3,4,5,7,0], "node_1": [1], "node_2": [2], "node_3": [3], "node_4": [4], "node_5": [5], "node_6": [6], "node_7": [7], "node_8": [3]},
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [2], "node_4": [3]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself

    "epochs": 1000, "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    # "inversion_algo": "send_model_centralized",
    "inversion_algo": "send_grads",
    "fedadam": False, "distadam": True,
    "server_lr": 0.01, "server_beta_1": 0.9, "server_beta_2": 0.99, "server_tau": 1e-9,
    "global_steps": 400, "local_steps": 0, "local_lr": 1e-3,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 250,
    "first_time_steps": 2000, "adaptive_distill": False,

    "exp_keys": ["distill_epochs", "global_steps", "local_steps", "position", "warmup", "inversion_algo", "adaptive_distill", "fedadam", "distadam"]
}

iid_clients_distill_fedadam = {
    "algo": "distill_reps",
    "exp_id": "fedadam",
    "exp_type": "iid_clients_distill_reps",
    "load_existing": True,
    "checkpoint_paths": {
      "1": "expt_dump/4client_models/user1.pt",
      "2": "expt_dump/4client_models/user2.pt",
      "3": "expt_dump/4client_models/user3.pt",
      "4": "expt_dump/4client_models/user4.pt",
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # Learning setup
    "num_clients": 4, "top_k": 1, "samples_per_client": 6000,
    # "device_ids": {"node_0": [6,1,4,2,5], "node_1": [6], "node_2": [1], "node_3": [4], "node_4": [2], "node_5": [5]},
    # "device_ids": {"node_0": [6,1,2,3,4,5,7,0], "node_1": [1], "node_2": [2], "node_3": [3], "node_4": [4], "node_5": [5], "node_6": [6], "node_7": [7], "node_8": [3]},
    "device_ids": {"node_0": [3], "node_1": [3], "node_2": [1], "node_3": [2], "node_4": [0]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself

    "epochs": 1000, "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    # "inversion_algo": "send_model_centralized",
    "fedadam": True, "distadam": True,
    "inversion_algo": "send_grads",
    "server_lr": 0.01, "server_beta_1": 0.9, "server_beta_2": 0.99, "server_tau": 1e-9,
    "global_steps": 400, "local_steps": 0, "local_lr": 1e-3,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 250,
    "first_time_steps": 2000, "adaptive_distill": False,

    "exp_keys": ["distill_epochs", "global_steps", "local_steps", "position", "warmup", "inversion_algo", "adaptive_distill", "fedadam", "distadam"]
}

# current_config = iid_clients_collab_new
# current_config = iid_clients_isolated_new
# current_config = iid_clients_federated_new
#current_config = iid_clients_distill_fedavg
# current_config = iid_clients_distill_fedadam
current_config = feddream
# current_config = iid_clients_distill_distadam
#current_config = iid_clients_distill_fedadam
