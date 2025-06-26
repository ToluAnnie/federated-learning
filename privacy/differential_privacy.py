from diffprivlib import mechanisms
import torch

class DifferentialPrivacy:
    """
    Privacy-preserving module for federated learning.
    This implementation can be published to GitHub.
    """
    def __init__(self, epsilon=1.0, sensitivity=1.0):
        """
        Initialize with differential privacy parameters.
        Local configuration file should contain these parameters.
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.laplace_mech = mechanisms.Laplace(epsilon=self.epsilon, sensitivity=self.sensitivity)
    
    def add_noise_to_weights(self, weights):
        """
        Add differential privacy noise to model weights before sharing.
        This ensures patient data never leaves the device.
        """
        noisy_weights = []
        for weight in weights:
            # Add noise to each weight parameter
            noisy_weight = self.laplace_mech.randomise(weight.item())
            noisy_weights.append(torch.tensor([noisy_weight]))
        return torch.cat(noisy_weights)

# Example usage (can be run locally for testing)
if __name__ == "__main__":
    # Create synthetic data
    from data.synthetic_health_data import generate_synthetic_health_data
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
    noisy_weights = dp.add_noise_to_weights(model.state_dict()['linear.weight'])
    print("Noisy weights (with differential privacy):")
    print(noisy_weights)