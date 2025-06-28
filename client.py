import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.client import NumPyClient, start_client
from flwr.common import ndarrays_to_parameters
from data.synthetic_health_data import load_client_data
from models.federated_logistic_regression import FederatedLogisticRegression
from privacy.differential_privacy import DifferentialPrivacy
import torch.optim as optim
import argparse
import numpy as np

class FederatedClient(NumPyClient):
    """
    Federated learning client for diabetes risk prediction.
    """
    def __init__(self, client_id: int):
        self.client_id = client_id
        df_data = load_client_data(num_samples=1000)

        X = torch.tensor(df_data.drop('diabetes_risk', axis=1).values, dtype=torch.float32)
        y = torch.tensor(df_data['diabetes_risk'].values, dtype=torch.float32)
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=32)

        self.model = FederatedLogisticRegression(input_dim=5)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.dp_module = DifferentialPrivacy(epsilon=1.0, sensitivity=0.1)

        print(f"Client {self.client_id} initialized with {len(self.dataset)} samples.")

    def get_parameters(self, config):
        """Gets the model parameters as a list of NumPy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        """Sets the model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {key: torch.tensor(value) for key, value in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Trains the model on the local dataset."""
        self.set_parameters(parameters)

        num_epochs = int(config.get("num_epochs", 1))

        self.model.train()
        for epoch in range(num_epochs):
            for inputs, labels in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = torch.nn.BCELoss()(outputs.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()

        updated_params = self.get_parameters(config={})
        noisy_params = [self.dp_module.add_noise_to_weights(p) for p in updated_params]

        print(f"Client {self.client_id} finished training and applied DP.")

        return noisy_params, len(self.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluates the model on the local dataset."""
        self.set_parameters(parameters)

        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        criterion = torch.nn.BCELoss()

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                total_loss += loss.item() * len(inputs)

                predictions = (outputs > 0.5).squeeze().long()
                correct += (predictions == labels.long()).sum().item()
                total += len(labels)

        avg_loss = total_loss / total
        accuracy = correct / total

        print(f"Client {self.client_id} evaluation: Loss {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return avg_loss, total, {"accuracy": accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client for federated learning.")
    parser.add_argument("--client-id", type=int, required=True, help="The ID of the client.")
    args = parser.parse_args()

    start_client(server_address="127.0.0.1:8080", client=FederatedClient(client_id=args.client_id))
