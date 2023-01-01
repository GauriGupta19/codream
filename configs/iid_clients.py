iid_clients_collab_new = {
    "algo": "dare",
    "exp_id": 6,
    "exp_type": "iid_clients_collab",
    "load_existing": False, "start_epoch": 500,
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": {"node_0": [], "node_1": [4,5], "node_2": [6,7]},
    # Learning setup
    "num_clients": 2, "samples_per_client": 1000,
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself 
    "top_k": 1, 
    
    "epochs": 1000, "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256, 
    
    # params for model
    "position": 4, "inp_shape": [0, 256, 8, 8],

    # Params for gradient descent on data
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 10, "alpha_tv": 2.5e-5, "alpha_l2": 3e-8, "alpha_f": 10.0,
    "distill_batch_size": 256, "distill_epochs": 10, "warmup": 30,
    
    "exp_keys": ["distill_epochs", "steps", "position"]
}

iid_clients_adaptive_collab = {
    "exp_id": 10,
    "exp_type": "iid_clients_adaptive_collab",
    "load_existing": True, "start_epoch": 16,
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": [7],
    # Learning setup
    "num_clients": 2, "samples_per_client": 2000,
    
    "epochs": 1000,
    "model_lr": 3e-4, "batch_size": 256,

    # Params for model
    "position": 4, "inp_shape": [0, 256, 8, 8],
    
    # Params for gradient descent on data
    "data_lr": 0.05, "steps": 2000, "alpha_preds": 10, "alpha_tv": 2.5e-5, "alpha_l2": 3e-8, "alpha_f": 1.0,
    "distill_batch_size": 256, "distill_epochs": 10, "warmup": 20,

    "exp_keys": ["distill_epochs", "steps", "position"]
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
    "device_ids": {"node_0": [1,2]},

    # Learning setup
    "num_clients": 1, "samples_per_client": 2000,

    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,
    
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
    "device_ids": {"node_0": [1], "node_1": [1,2], "node_2": [3,4]},

    # Learning setup
    "num_clients": 2, "samples_per_client": 1000,
    
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,
    
    "exp_keys": []
}

current_config = iid_clients_collab_new
