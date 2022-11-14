import jmespath, importlib, os


def load_config(config_path):
    path = '.'.join(config_path.split('.')[1].split('/')[1:])
    config = importlib.import_module(path).current_config
    return process_config(config)

def process_config(config):
    config['num_gpus'] = len(config.get('device_ids'))
    config['batch_size'] = config.get('batch_size', 64) * config['num_gpus']
    config['seed'] = config.get('seed') or 1
    config['load_existing'] = config.get('load_existing') or False

    experiment_name = "{}_{}_{}clients_{}samples_{}".format(
        config['exp_type'],
        config['dset'],
        config['num_clients'],
        config['samples_per_client'],
        config['exp_id'])
    for exp_key in config["exp_keys"]:
        item = jmespath.search(exp_key, config)
        assert item is not None
        key = exp_key.split(".")[-1]
        assert key is not None
        experiment_name += "_{}_{}".format(key, item)
    
    experiments_folder = config["dump_dir"]
    results_path = experiments_folder + experiment_name + f"_seed{config['seed']}"

    log_path = results_path + "/logs/"
    images_path = results_path + "/images/"

    config["experiment_name"] = experiment_name
    config["log_path"] = log_path
    config["images_path"] = images_path
    config["results_path"] = results_path
    config["saved_models"] = results_path + "/saved_models/"

    return config