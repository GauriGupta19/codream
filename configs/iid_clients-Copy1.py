iid_clients_collab_new = {
    # "algo": "fedavg",
    "algo": "moon",
    "exp_id": "full",
    "exp_type": "fedprox",
    # "exp_type": "iid_clients_collab_entropy",
    "load_existing": False,
    "checkpoint_paths": {"1": "cifar10_resnet34.pth"},
    # "checkpoint_paths": {"1": "/mas/camera/Experiments/corral/4clients_diff_acc/12k.pt"},
    # "checkpoint_paths": {"1": "expt_dump/iid_clients_collab_entropy_cifar10_2clients_5000samples_mvt2_data_log3_distill_epochs_10_steps_10_position_0_warmup_150_seed2/saved_models/user1.pt",
    #                      "2": "expt_dump/iid_clients_collab_entropy_cifar10_2clients_5000samples_mvt2_data_log3_distill_epochs_10_steps_10_position_0_warmup_150_seed2/user2.pt"},
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 0,
    # Learning setup
    "num_clients": 1, "top_k": 1, "samples_per_client": 50000,
    # "device_ids": {"node_0": [1], "node_1": [2], "node_2": [3], "node_3": [5], "node_4": [6], "node_5": [7]},
    "device_ids": {"node_0": [0], "node_1": [1]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    "model": "resnet34", "model_lr": 0.1, "batch_size": 256,
    
    # params for model
    # "method": "orig", "epochs": 1000,
    # "position": 0, "inp_shape": [0, 3, 32, 32],
    # "steps": 1000, "first_time_steps": 1000,
    
    #fast_meta
    "method": "fast_meta", "epochs": 220+200,
    "position": 0, "inp_shape": [0, 256], "out_shape": [0, 3, 32, 32],
    "ismaml": 1,  "reset_l0": 1, "warmup_g": 20,
    "steps": 10, "lr_z": 0.01, "lr_g": 2e-3,

    # Params for gradient descent on data
    "inversion_algo": "send_reps",
    "data_lr": 0.05, "bn_mmt": 0.9,
    "alpha_preds": 0.5, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 200,
    "first_time_steps": 2000, 
    
    "exp_keys": ["distill_epochs", "steps", "position", "warmup"]
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

current_config = iid_clients_collab_new
# current_config = iid_clients_isolated_new
