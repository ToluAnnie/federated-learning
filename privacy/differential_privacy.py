from diffprivlib import mechanisms
import numpy as np
import torch

class DifferentialPrivacy:
    """
    Privacy-preserving module for federated learning.
    """
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        """
        Initialize with differential privacy parameters.
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.laplace_mech = mechanisms.Laplace(epsilon=self.epsilon, sensitivity=self.sensitivity)

    def add_noise_to_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Add differential privacy noise to model weights.
        """
        noisy_weights = np.zeros_like(weights)
        for i in np.ndindex(weights.shape):
            noisy_weights[i] = self.laplace_mech.randomise(weights[i])
        return noisy_weights

# Example usage (can be run locally for testing)
if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    from data.synthetic_health_data import generate_synthetic_health_data
    from models.federated_logistic_regression import FederatedLogisticRegression
    
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
    
    # Apply differential privacy to model weights
    dp = DifferentialPrivacy(epsilon=1.0, sensitivity=0.1)
    
    # Get weights as NumPy arrays
    weights_np = model.linear.weight.detach().numpy()
    noisy_weights_np = dp.add_noise_to_weights(weights_np)
    
    print("Original weights:")
    print(weights_np)
    print("\nNoisy weights (with differential privacy):")
    print(noisy_weights_np)
