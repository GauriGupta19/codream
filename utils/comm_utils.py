from mpi4py import MPI


class CommUtils():
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def send_signal(self, dest, data, tag=None):
        if tag is not None:
            self.comm.send(data, dest=dest, tag=tag)
        else:
            self.comm.send(data, dest=dest)

    def wait_for_signal(self, src, tag=None):
        """
        Wait for a signal from a source node
        """
        if tag is not None:
            recv_data = self.comm.recv(source=src, tag=tag)
        else:
            recv_data = self.comm.recv(source=src)
        return recv_data

    def wait_for_all_clients(self, client_ids, signal):
        """
        """
        data_list = []
        for client_id in client_ids:
            data = self.wait_for_signal(client_id, signal)
            data_list.append(data)
        return data_list
