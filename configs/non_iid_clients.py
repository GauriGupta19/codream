non_iid_clients_collab = {
    "exp_id": 5,
    "load_existing": True, "start_epoch": 10,
    "exp_type": "non_iid_clients_collab",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": [1,2,3,0],
    # Learning setup
    "num_clients": 2, "samples_per_client": 5000, "class_per_client": 2,
    
    "epochs": 500,
    "model_lr": 3e-4, "batch_size": 256,
    
    # Params for gradient descent on data
    "warmup": 10,
    "data_lr": 0.05, "steps": 2000, "alpha_preds": 1., "alpha_tv": 0.001, "alpha_l2": 0., "alpha_f": 10.,
    "distill_batch_size": 128, "distill_epochs": 10,
    
    "exp_keys": ["distill_epochs", "steps", "class_per_client"]
}

non_iid_clients_federated = {
    "exp_id": 0,
    "exp_type": "non_iid_clients_federated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": [0,1,2,3],

    # Learning setup
    "num_clients": 2, "samples_per_client": 2000, "class_per_client": 2,
    
    "epochs": 500,
    "model_lr": 3e-4, "batch_size": 256,
    
    "exp_keys": ["num_clients", "class_per_client"]
}

current_config = non_iid_clients_collab