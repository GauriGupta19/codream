import argparse
import socket
from scheduler import Scheduler
import torch
import subprocess

# b_default = "./configs/iid_clients.py"
b_default = "./configs/non_iid_clients.py"
parser = argparse.ArgumentParser(description='Run collaborative learning experiments')
parser.add_argument('-b', nargs='?', default=b_default, type=str,
                    help='filepath for benchmark config, default: {}'.format(b_default))
parser.add_argument('-seed', nargs='?', default=4, type=int)
args = parser.parse_args()

scheduler = Scheduler()
scheduler.assign_config_by_path(args.b, args.seed)
scheduler.initialize()
scheduler.run_job()