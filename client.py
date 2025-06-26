import torch
from torch.utils.data import DataLoader
from flwr.client import Client
from data.synthetic_health_data import load_client_data
from models.federated_logistic_regression import FederatedLogisticRegression
from privacy.differential_privacy import DifferentialPrivacy

class Client(Client):
    def __init__(self, client_id):
        self.client_id = client_id
        self.data = load_client_data(client_id)
        self.model = FederatedLogisticRegression()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, num_epochs=10):
        dataloader = DataLoader(self.data, batch_size=32)
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = torch.nn.BCELoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()
            # Apply differential privacy to model parameters before sending updates
            params = list(self.model.parameters())
            noisy_params = add_differential_privacy(params)
            # Replace parameters with noisy versions
            for param, noisy_param in zip(params, noisy_params):
                param.data.copy_(noisy_param)
        return self.model.parameters()

    def evaluate(self):
        # Optional: Add evaluation logic if needed
        pass
