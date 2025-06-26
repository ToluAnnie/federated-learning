import torch
from torch import nn, optim
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from models.federated_logistic_regression import FederatedLogisticRegression
from privacy.differential_privacy import DifferentialPrivacy

class CentralizedServer(fl.server.Server):
    def __init__(self):
        super().__init__(client_manager=fl.server.SimpleClientManager())
        self.model = FederatedLogisticRegression(input_dim=5)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.strategy = FedAvg()

    def train(self, clients, rounds=5):
        server = Server(self.model, self.optimizer, strategy=self.strategy)
        for round_num in range(rounds):
            print(f"Round {round_num + 1}")
            server.fit(clients)
            server.evaluate(clients)
            # Aggregate updates with differential privacy
            dp = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0)
            updated_weights = dp.add_noise_to_weights(server.get_weights())
            server.set_weights(updated_weights)
