iid_clients_collab = {
    "exp_id": 1,
    "exp_type": "iid_clients_collab",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": [0,1,2,3],
    # Learning setup
    "num_clients": 3, "samples_per_client": 2000,
    
    "epochs": 500,
    "model_lr": 3e-4, "batch_size": 256, 
    
    # Params for gradient descent on data
    "data_lr": 0.05, "steps": 2000, "alpha_preds": 10, "alpha_tv": 2.5e-5, "alpha_l2": 3e-8, "alpha_f": 10.0,
    "distill_batch_size": 256, "distill_epochs": 10, "warmup": 20,
    
    "exp_keys": ["distill_epochs", "steps"]
}

iid_clients_isolated = {
    "exp_id": 0,
    "exp_type": "iid_clients_isolated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": [0,1,2,3],

    # Learning setup
    "num_clients": 3, "samples_per_client": 2000,
    
    "epochs": 500,
    "model_lr": 3e-4, "batch_size": 256,
    
    "exp_keys": []
}

current_config = iid_clients_collab