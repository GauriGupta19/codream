import os, pickle, shutil, logging, torch
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from shutil import copytree, copy2
from glob import glob
from PIL import Image


# Normalize an image
def deprocess(img):
    inv_normalize = T.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(img)
    img = 255*img
    return img.type(torch.uint8)


class Utils():
    def __init__(self, config) -> None:
        self.config = config
        self.copy_source_code(config["results_path"])
        self.logger = Logger(self.config["log_path"], config["load_existing"])
    
    def check_and_create_path(self, path):
        if os.path.isdir(path):
            print("Experiment in {} already present".format(path))
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
        if self.config["load_existing"]:
            print("Continue with loading checkpoint")
            return
        else:
            # if os.path.isdir(path):
                # throw a prompt
            self.check_and_create_path(path)
            # else:
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

    def log_image(self, imgs, key, iteration):
        # imgs = deprocess(imgs.detach().cpu())[:64]
        grid = make_grid(imgs.detach().cpu(), normalize=True, scale_each=True)
        # grid = imgs
        # save_image(grid, f"./expt_dump/temp/expt{key}{iteration}.png")
        im_ob = Image.fromarray(grid.permute(1, 2, 0).numpy(), mode='RGB')
        im_ob.save(f"./expt_dump/temp/expt{key}{iteration}.png")
        self.writer.add_image(key, grid.numpy(), iteration)

    def log_console(self, msg):
        logging.info(msg)

    def log_tb(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)