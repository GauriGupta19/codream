from mpi4py import MPI
import torch, random, numpy
from algos.base_class import BaseNode
from algos.dare import DAREClient, DAREServer
from algos.distill_reps import DistillRepsClient, DistillRepsServer
from algos.feddream import FedDreamClient, FedDreamServer
from algos.feddream_fast import FedDreamFastClient, FedDreamFastServer
from algos.feddream_fast_independent import (
    FedDreamFastClientIndp,
    FedDreamFastServerIndp,
)
from algos.feddream_fast_noniid import (
    FedDreamFastNoniidClient,
    FedDreamFastNoniidServer,
)
from algos.fl import FedAvgClient, FedAvgServer
from algos.isolated import IsolatedClient, IsolatedServer
from algos.scaffold import SCAFFOLDClient, SCAFFOLDServer
from algos.fedprox import FedProxClient, FedProxServer
from algos.moon import MoonClient, MoonServer
from algos.centralized import CentralizedServer
from algos.fedgen import FedGenClient, FedGenServer
from utils.log_utils import copy_source_code
from utils.config_utils import load_config

# should be used as: algo_map[algo_name][rank>0](config)
# If rank is 0, then it returns the server class otherwise the client class
algo_map = {
    "fedavg": [FedAvgServer, FedAvgClient],
    "isolated": [IsolatedServer, IsolatedClient],
    "fedprox": [FedProxServer, FedProxClient],
    "moon": [MoonServer, MoonClient],
    "centralized": [CentralizedServer],
    "dare": [DAREServer, DAREClient],
    "distill_reps": [DistillRepsServer, DistillRepsClient],
    "scaffold": [SCAFFOLDServer, SCAFFOLDClient],
    "feddream": [FedDreamServer, FedDreamClient],
    "feddream_fast": [FedDreamFastServer, FedDreamFastClient],
    "feddream_fast_indp": [FedDreamFastServerIndp, FedDreamFastClientIndp],
    "fedgen": [FedGenServer, FedGenClient],
}


def get_node(config: dict, rank) -> BaseNode:
    algo_name = config["algo"]
    print(algo_name)
    return algo_map[algo_name][rank > 0](config)


class Scheduler:
    """Manages the overall orchestration of experiments"""

    def __init__(self) -> None:
        pass

    def assign_config_by_path(self, config_path, seed) -> None:
        self.config = load_config(config_path, seed)

    def initialize(self) -> None:
        assert self.config is not None, "Config should be set when initializing"

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # set seeds
        seed = self.config["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)

        if rank == 0:
            copy_source_code(self.config)
        self.node = get_node(self.config, rank=rank)

    def run_job(self) -> None:
        self.node.run_protocol()
