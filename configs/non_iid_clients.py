non_iid_clients_collab = {
    "exp_id": 0,
    "load_existing": True,
    "start_epoch": 9,
    "exp_type": "non_iid_clients_collab",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    "device_ids": [1,2,3,4],
    # Learning setup
    "num_clients": 2, "samples_per_client": 5000, "class_per_client": 2,
    
    "epochs": 500,
    "model_lr": 3e-4, "batch_size": 256,
    
    # Params for gradient descent on data
    "warmup": 10,
    "data_lr": 0.05, "steps": 4000, "alpha_preds": 100., "alpha_tv": 2.5e-5, "alpha_l2": 3e-8, "alpha_f": 10.0,
    "distill_batch_size": 256, "distill_epochs": 10,
    
    "exp_keys": ["distill_epochs", "steps", "class_per_client"]
}

current_config = non_iid_clients_collab