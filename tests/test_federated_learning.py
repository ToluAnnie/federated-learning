import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.federated_logistic_regression import FederatedLogisticRegression
from data.synthetic_health_data import generate_synthetic_health_data
import unittest

class TestFederatedLearning(unittest.TestCase):
    """
    Unit tests for the federated learning components.
    """
    def test_model_training(self):
        """
        Test if the model can be initialized and trained without errors.
        """
        data = generate_synthetic_health_data(100)
        X = torch.tensor(data.drop('diabetes_risk', axis=1).values, dtype=torch.float32)
        y = torch.tensor(data['diabetes_risk'].values, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = FederatedLogisticRegression(input_dim=5)
        
        # Check if training runs without raising an exception
        try:
            model.train_model(data_loader, epochs=2)
            train_successful = True
        except Exception as e:
            print(f"Training failed with error: {e}")
            train_successful = False
            
        self.assertTrue(train_successful, "Model training failed.")
        
    def test_model_forward_pass(self):
        """
        Test if the forward pass produces an output of the correct shape.
        """
        input_dim = 5
        model = FederatedLogisticRegression(input_dim=input_dim)
        dummy_input = torch.randn(10, input_dim)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, torch.Size([10, 1]), "Forward pass output shape is incorrect.")

if __name__ == "__main__":
    unittest.main()
