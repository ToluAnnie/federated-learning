import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from flwr.client import Client
from data.synthetic_health_data import load_client_data
from models.federated_logistic_regression import FederatedLogisticRegression
from privacy.differential_privacy import DifferentialPrivacy

class Client(Client):
    def __init__(self, client_id):
        self.client_id = client_id
        self.data = load_client_data(client_id)
        # Debug: Print columns to verify 'diabetes_risk' exists
        print("Client data columns:", self.data.columns.tolist())
        # Convert to PyTorch tensors
        self.X = torch.tensor(self.data.drop('diabetes_risk', axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(self.data['diabetes_risk'].values, dtype=torch.float32)
        self.model = FederatedLogisticRegression(input_dim=5)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, num_epochs=10):
        print(f"Client {self.client_id} connected to server and starting training...")
        # Create TensorDataset from the tensors
        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=32)
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

if __name__ == "__main__":
    # Example usage: Run client with ID 1
    client = Client(client_id=1)
    client.fit(num_epochs=1)
