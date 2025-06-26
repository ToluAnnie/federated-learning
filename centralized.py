import torch
from torch import nn, optim
from flwr.server import Server
from flwr.server.strategy import FedAvg
from models.federated_logistic_regression import FederatedLogisticRegression
from privacy.differential_privacy import add_differential_privacy

class CentralizedServer:
    def __init__(self):
        self.model = FederatedLogisticRegression()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.strategy = FedAvg()

    def train(self, clients, rounds=5):
        server = Server(self.model, self.optimizer, strategy=self.strategy)
        for round_num in range(rounds):
            print(f"Round {round_num + 1}")
            server.fit(clients)
            server.evaluate(clients)
            # Aggregate updates with differential privacy
            updated_weights = add_differential_privacy(server.get_weights())
            server.set_weights(updated_weights)