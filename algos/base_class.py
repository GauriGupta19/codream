from abc import ABC, abstractmethod
import torch, numpy
from utils.comm_utils import CommUtils
from utils.data_utils import cifar_extr_noniid, get_dataset
from torch.utils.data import DataLoader, Subset

from utils.log_utils import LogUtils
from utils.model_utils import ModelUtils

class BaseNode(ABC):
    def __init__(self, config) -> None:
        self.comm_utils = CommUtils()
        self.node_id = self.comm_utils.rank

        if self.node_id == 0:
            self.log_utils = LogUtils(config)
            self.log_utils.log_console("Config: {}".format(config))

        self.setup_cuda(config)
        self.model_utils = ModelUtils()
        self.dset_obj = get_dataset(config["dset"], config["dpath"])
        self.set_data_parameters()

    def set_data_parameters(self):
        self.TAG_DONE = 0
        self.TAG_START = 1
        self.TAG_SET_REP = 2

        self.best_acc = 0.

    def setup_cuda(self, config):
        # Need a mapping from rank to device id
        device_ids_map = config["device_ids"]
        node_name = "node_{}".format(self.node_id)
        self.device_ids = device_ids_map[node_name]
        gpu_id = self.device_ids[0]

        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

    def set_model_parameters(self, config):
        # Model related parameters
        optim = torch.optim.Adam
        self.model = self.model_utils.get_model(config["model"], self.device, self.device_ids)
        self.optim = optim(self.model.parameters(), lr=config["model_lr"])
        self.loss_fn = torch.nn.CrossEntropyLoss()

    @abstractmethod
    def run_protocol(self):
        raise NotImplementedError


class BaseClient(BaseNode):
    """
    Abstract class for all algorithms
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.server_node = 0
        self.set_parameters(config)

    def set_parameters(self, config):
        """
        Set the parameters for the client
        """        
        self.set_model_parameters(config)
        self.set_data_parameters(config)

    def set_data_parameters(self, config):
        train_dset = self.dset_obj.train_dset
        test_dset = self.dset_obj.test_dset
        samples_per_client = config["samples_per_client"]
        batch_size = config["batch_size"]
        # Subtracting 1 because rank 0 is the server
        client_idx = self.node_id - 1
        """This is a dangerous line because if the permutation is not the same for all clients,
        then the samples will be same for clients which is not what we want
        Should not happen because we are using the same seed for all clients"""
        # TODO: Verify properly that indices remain same for all clients
        if config["exp_type"].startswith("non_iid"):
            user_groups_train, user_groups_test = cifar_extr_noniid(train_dset, test_dset,
                                                                    config["num_clients"], config["class_per_client"],
                                                                    config["samples_per_client"], rate_unbalance=1)
            dset = Subset(train_dset, list(user_groups_train[client_idx].astype(int)))
        else:
            indices = numpy.random.permutation(len(train_dset))
            dset = Subset(train_dset, indices[client_idx*samples_per_client:(client_idx+1)*samples_per_client])
        self.dloader = DataLoader(dset, batch_size=batch_size*len(self.device_ids), shuffle=True)
        self._test_loader = DataLoader(test_dset, batch_size=batch_size)

    def local_train(self, dataset, **kwargs):
        """
        Train the model locally
        """
        raise NotImplementedError

    def local_test(self, dataset, **kwargs):
        """
        Test the model locally
        """
        raise NotImplementedError

    def get_representation(self, **kwargs):
        """
        Share the model representation
        """
        raise NotImplementedError

    def run_protocol(self):
        raise NotImplementedError


class BaseServer(BaseNode):
    """
    Abstract class for orchestrator
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_clients = config["num_clients"]
        self.clients = list(range(1, self.num_clients+1))

    def set_data_parameters(self, config):
        test_dset = self.dset_obj.test_dset
        batch_size = config["batch_size"]
        self._test_loader = DataLoader(test_dset, batch_size=batch_size)

    def aggregate(self, representation_list, **kwargs):
        """
        Aggregate the knowledge from the clients
        """
        raise NotImplementedError

    def test(self, dataset, **kwargs):
        """
        Test the model on the server
        """
        raise NotImplementedError

    def get_model(self, **kwargs):
        """
        Get the model
        """
        raise NotImplementedError

    def run_protocol(self):
        raise NotImplementedError