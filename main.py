import argparse
import socket
from scheduler import Scheduler
import torch
import subprocess

a = torch.tensor([3, 5]).to('cuda:1')#cuda()
print(a.device)
b_default = "./configs/iid_clients.py"
parser = argparse.ArgumentParser(description='Run collaborative learning experiments')
parser.add_argument('-b', nargs='?', default=b_default, type=str,
                    help='filepath for benchmark config, default: {}'.format(b_default))
args = parser.parse_args()

scheduler = Scheduler()
scheduler.assign_config_by_path(args.b)
scheduler.initialize()
scheduler.run_job()
