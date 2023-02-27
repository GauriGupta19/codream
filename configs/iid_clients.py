iid_clients_collab_new = {
    "algo": "dare",
    "exp_id": 6,
    "exp_type": "iid_clients_collab_entropy",
    "load_existing": False, "start_epoch": 500,
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # Learning setup
    "num_clients": 2, "top_k": 1, "samples_per_client": 1000,
    "device_ids": {"node_0": [], "node_1": [0], "node_2": [1]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    
    "epochs": 1000, "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 64, 
    
    # params for model
    # "method": "orig", "ismaml": 0,
    # "position": 4, "inp_shape": [0, 256, 8, 8], "out_shape": [0, 256, 8, 8],
    
    "method": "fast_meta", "ismaml": 1,
    "lr_g": 5e-3, "lr_z": 0.015,
    
    "position": 0, "inp_shape": [0, 256], "out_shape": [0, 3, 32, 32],
    
    


    # Params for gradient descent on data
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 10, "alpha_tv": 2.5e-7, "alpha_l2": 0., "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 20,
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
    "device_ids": {"node_0": [0], "node_1": [0,1], "node_2": [2,3]},

    # Learning setup
    "num_clients": 2, "samples_per_client": 500,
    
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4, "batch_size": 256,
    
    "exp_keys": []
}

current_config = iid_clients_collab_new
