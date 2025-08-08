from fl_client import SCCClient
import numpy as np

class EdgeServer:
    def __init__(self, client_ids, client_datasets):
        self.clients = [SCCClient(cid, client_datasets[cid]) for cid in client_ids]

    def aggregate(self, parameters):
        total_size = 0
        aggregated = [np.zeros_like(p) for p in parameters[0]]
        for client in self.clients:
            client_params, client_size = client.fit(parameters)
            for i, p in enumerate(client_params):
                aggregated[i] += p * client_size
            total_size += client_size
        return [p / total_size for p in aggregated] if total_size > 0 else parameters