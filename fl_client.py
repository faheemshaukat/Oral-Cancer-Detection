import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import Parameters
from models import SCCModel
from collections import OrderedDict

class SCCClient:
    def __init__(self, cid, trainset):
        self.cid = cid
        self.model = SCCModel('efficientnet_b4').to('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters):
        self.set_parameters(parameters)
        for epoch in range(10):  # 10 local epochs as per methodology
            for data, labels in self.trainloader:
                data, labels = data.to(self.model.device), labels.to(self.model.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset)