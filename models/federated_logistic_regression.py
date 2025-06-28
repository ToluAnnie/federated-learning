import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FederatedLogisticRegression(nn.Module):
    """
    Federated learning model for diabetes risk prediction.
    """
    def __init__(self, input_dim):
        super(FederatedLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def train_model(self, data_loader, epochs=10, learning_rate=0.01):
        """
        Train the model on local data.
        """
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                loss_value = loss.item()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value:.4f}")

        return self.state_dict()

# Example usage (can be run locally for testing)
if __name__ == "__main__":
    from data.synthetic_health_data import generate_synthetic_health_data
    
    # Create synthetic data
    data = generate_synthetic_health_data(1000)

    # Convert to PyTorch tensors
    X = torch.tensor(data.drop('diabetes_risk', axis=1).values, dtype=torch.float32)
    y = torch.tensor(data['diabetes_risk'].values, dtype=torch.float32)

    # Create data loader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train model
    model = FederatedLogisticRegression(input_dim=5)
    model.train_model(data_loader, epochs=5)
