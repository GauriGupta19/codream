non_iid_clients_collab_new = {
    "algo": "dare",
    "exp_id": 1,
    "exp_type": "non_iid_clients_collab",
    "load_existing": False, "start_epoch": 500,
    "dset": "mnist",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/mnist",
    "seed": 2,
    # Learning setup
    "num_clients": 2, "top_k": 1, "samples_per_client": 1000, "class_per_client": 2, "sp": [[0,1],[2,3]],
    "device_ids": {"node_0": [], "node_1": [4], "node_2": [5]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    
    "epochs": 1000, "model": "ResNet18",
    "model_lr": 0.1, "batch_size": 128, 
    
    # params for model
    "position": 0, "inp_shape": [0, 1, 28, 28],

    # Params for gradient descent on data
    "data_lr": 0.05, "steps": 2000,
    "alpha_preds": 100, "alpha_tv": 2.5e-5, "alpha_l2": 3e-8, "alpha_f": 10.0,
    "distill_batch_size": 128, "distill_epochs": 10, "warmup": 20,
    "first_time_steps": 2000,
    "inversion_algo": "random_deepinversion",
    
    "exp_keys": ["distill_epochs", "steps", "position", "warmup", "inversion_algo"]
}

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
    "model_lr": 0.1, "batch_size": 256,
    
    # Params for gradient descent on data
    "warmup": 10,
    "data_lr": 0.05, "steps": 2000, "alpha_preds": 1., "alpha_tv": 0.001, "alpha_l2": 0., "alpha_f": 10.,
    "distill_batch_size": 128, "distill_epochs": 10,
    
    "exp_keys": ["distill_epochs", "steps", "class_per_client"]
}

non_iid_clients_federated = {
    "algo": "fedprox",
    "exp_id": "test",
    "exp_type": "non_iid_clients_federated",
    "dset": "cifar100",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar100",
    "seed": 1,
    "device_ids": {"node_0": [0], "node_1": [1], "node_2": [1], "node_3": [2], "node_4": [2]},
    
    # Learning setup
    "num_clients":4, "samples_per_client": 1000, "alpha": 0.1,
    # "class_per_client": 2,
    
    "epochs": 500, "local_runs": 5,
    "model": "resnet18", "model_lr": 0.1, "batch_size": 256,
    
    "exp_keys": ["algo", "alpha", "epochs", "local_runs", "dset", "num_clients",  "samples_per_client", "model"]
}

feddream = {
    "seed": 2,
    "algo": "feddream",
    "exp_id": "1",
    "exp_type": "non_iid_clients_feddream",
    "load_existing": False,
    "checkpoint_paths": {
        '1': './expt_dump/2k_clients/user1.pt',
        '2': './expt_dump/2k_clients/user2.pt',
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    # Learning setup
    "num_clients": 4, "samples_per_client": 3000, "alpha": 1,
    "device_ids": {"node_0": [2], "node_1": [3], "node_2": [4], "node_3": [5], "node_4":[6]},
    "epochs": 1000, "model": "resnet34",
    "model_lr": 0.1, "batch_size": 256,

    # params for model
    "position": 0, "inp_shape": [0, 3, 32, 32],

    # Params for gradient descent on data
    "inversion_algo": "send_grads",
    "data_lr": 0.05, "global_steps": 2000, "local_steps": 1,
    "alpha_preds": 0.1, "alpha_tv": 2.5e-3, "alpha_l2": 1e-7, "alpha_f": 10.0,
    # for local training
    "warmup": 150,
    "distill_batch_size": 128, "distill_epochs": 20,
    "local_train_freq": 5,

    # adaptive distillation parameters
    "adaptive_server": True,
    "adaptive_client": False,
    "lambda_server": 0.1,

    "exp_keys": ["adaptive_server"]
}

current_config = non_iid_clients_federated
# current_config = feddream
