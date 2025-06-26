import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from flwr.client import Client, start_client
from flwr.common import FitRes, EvaluateRes
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
        self.y = torch.tensor(self.data['diabetes_risk'].values, dtype=torch.float32).unsqueeze(1)
        self.model = FederatedLogisticRegression(input_dim=5)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, fit_ins):
        # Use default number of epochs if not provided in config
        num_epochs = fit_ins.config.get("num_epochs", 10)
        print(f"Client {self.client_id} connected to server and starting training...")
        # Use BCEWithLogitsLoss which combines sigmoid and BCELoss
        criterion = torch.nn.BCEWithLogitsLoss()
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
            dp = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0)
            noisy_params = dp.add_noise_to_weights(params)
            # Replace parameters with noisy versions
            for param, noisy_param in zip(params, noisy_params):
                param.data.copy_(noisy_param)
        # Return FitRes object with parameters, number of examples, and empty metrics
        # Redundant import already present at top of file
        return FitRes(
            parameters=self.model.parameters(),
            num_examples=len(self.data),
            metrics={}
        )

    def evaluate(self, evaluate_ins):
        # Optional: Add evaluation logic if needed
        # Return a dummy result to avoid NoneType error
        return EvaluateRes(
            loss=0.0,
            parameters=self.model.parameters()
        )

if __name__ == "__main__":
    # Start Flower client with server address and configuration
    start_client(
        server_address="localhost:8080",
        client=Client(client_id=1)
    )
