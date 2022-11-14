import os, pickle, shutil, logging
from tensorboardX import SummaryWriter
from shutil import copytree, copy2
from glob import glob


class Utils():
    def __init__(self, config) -> None:
        self.config = config
        self.copy_source_code(config["results_path"])
        self.logger = Logger(self.config["log_path"], config["load_existing"])
    
    def check_and_create_path(self, path):
        if os.path.isdir(path):
            print("Experiment in {} already present".format(path))
            if self.config["load_existing"]:
                print("Continue with loading checkpoint")
                return
            inp = input("Press e to exit, r to replace it: ")
            if inp == "e":
                exit()
            elif inp == "r":
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                print("Input not understood")
                exit()
        else:
            os.makedirs(path)

    def copy_source_code(self, path):
        print("exp path:", path)
        if os.path.isdir(path):
            # throw a prompt
            self.check_and_create_path(path)
        else:
            os.makedirs(path)
            denylist = ["./__pycache__/", "./.ipynb_checkpoints/",
                        "./imgs/", '/'.join(path.split('/')[:-1])+'/']
            folders = glob(r'./*/')
            print(denylist, folders)

            # For copying python files
            for file_ in glob(r'./*.py'):
                copy2(file_, path)

            # For copying json files
            for file_ in glob(r'./*.json'):
                copy2(file_, path)

            for folder in folders:
                if folder not in denylist:
                    # Remove first char which is . due to the glob
                    copytree(folder, path + folder[1:])

            # For saving models in the future
            os.mkdir(self.config.get('saved_models'))
            os.mkdir(self.config.get('log_path'))
            print("source code copied to exp_dump")


class Logger():
    def __init__(self, log_dir, load_existing) -> None:
        log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
                     "%(filename)s::%(lineno)d::%(message)s"
        logging.basicConfig(filename="{log_path}/log_console.log".format(
                                                     log_path=log_dir),
                            level='DEBUG', format=log_format)
        logging.getLogger().addHandler(logging.StreamHandler())
        self.log_dir = log_dir
        self.init_tb(load_existing)

    def init_tb(self, load_existing):
        tb_path = self.log_dir + "/tensorboard"
        # if not os.path.exists(tb_path) or not os.path.isdir(tb_path):
        if not load_existing:
            os.makedirs(tb_path)
        self.writer = SummaryWriter(tb_path)

    def log_console(self, msg):
        logging.info(msg)

    def log_tb(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)