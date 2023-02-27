iid_clients_collab_new = {
    "algo": "dare",
    "exp_id": "mvt2_data_log3",
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
    "inversion_algo": "send_reps",
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 150,
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
