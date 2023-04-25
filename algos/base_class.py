from abc import ABC, abstractmethod
import torch, numpy
from utils.comm_utils import CommUtils
from utils.data_utils import get_dataset, non_iid_labels, non_iid_balanced, non_iid_unbalanced_dataidx_map
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
        
        if self.node_id == 1:
            if config["exp_type"].startswith("non_iid"):
                self.split_data = non_iid_balanced(self.dset_obj, config["num_clients"], config["samples_per_client"], config["alpha"])
        self.set_constants()

    def set_constants(self):
        self.best_loss = torch.inf
        self.best_acc = 0.

    def setup_cuda(self, config):
        # Need a mapping from rank to device id
        device_ids_map = config["device_ids"]
        node_name = "node_{}".format(self.node_id)
        self.device_ids = device_ids_map[node_name]
        if len(self.device_ids) > 0:
            gpu_id = self.device_ids[0]
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

    def set_model_parameters(self, config):
        # Model related parameters
        optim = torch.optim.SGD
        self.model = self.model_utils.get_model(config["model"], config["dset"], self.device, self.device_ids)
        # load a checkpoint if load_existing is set to True
        if config["load_existing"]:
            node_checkpoint_path = config["checkpoint_paths"].get(str(self.node_id), None)
            if node_checkpoint_path is not None:
                try:
                    self.model_utils.load_model(self.model, node_checkpoint_path)
                except FileNotFoundError:
                    print("No saved model found for node {}".format(self.node_id))
            else:
                print("No checkpoint path specified for node {}".format(self.node_id))
            # self.model_utils.load_model(self.model, config["saved_models"] + f"user{self.node_id}.pt")
            # self.model_utils.load_model(self.model, config["checkpoint"])
        self.optim = optim(self.model.parameters(), lr=config["model_lr"], momentum=0.9, weight_decay=1e-4)
        milestones = [ ms for ms in [120,150,180] ]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.2)
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
        if config["exp_type"].startswith("non_iid"):
            train_x, train_y = self.split_data
            dset = (train_x[client_idx], train_y[client_idx])
            
        if config["exp_type"].startswith("non_iid_labels"):
            sp = config["sp"]
            dset = non_iid_labels(train_dset,config["samples_per_client"],sp[client_idx])
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
        self.set_data_parameters(config)

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