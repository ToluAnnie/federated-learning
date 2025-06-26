import unittest
import torch
from models.federated_logistic_regression import FederatedLogisticRegression
from privacy.differential_privacy import DifferentialPrivacy
from data.synthetic_health_data import generate_synthetic_health_data

class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        """Create test data and model instances"""
        self.data = generate_synthetic_health_data(100)
        self.X = torch.tensor(self.data.drop('diabetes_risk', axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(self.data['diabetes_risk'].values, dtype=torch.float32)
        self.dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True)
        self.model = FederatedLogisticRegression(input_dim=5)
        self.dp = DifferentialPrivacy(epsilon=1.0, sensitivity=0.1)
    
    def test_model_training(self):
        """Test if model can be trained without errors"""
        initial_weights = self.model.state_dict()['linear.weight'].clone()
        self.model.train_model(self.data_loader, epochs=1)
        final_weights = self.model.state_dict()['linear.weight']
        self.assertFalse(torch.equal(initial_weights, final_weights), 
                        "Model weights should change after training")
    
    def test_differential_privacy(self):
        """Test if differential privacy adds noise to weights"""
        weights = self.model.state_dict()['linear.weight'].clone()
        noisy_weights = self.dp.add_noise_to_weights(weights)
        self.assertTrue(torch.any(noisy_weights != weights), 
                        "Differential privacy should add noise to weights")
    
    def test_data_shape(self):
        """Test if data dimensions match model expectations"""
        self.assertEqual(self.X.shape[1], 5, 
                        f"Expected 5 features, got {self.X.shape[1]}")

if __name__ == '__main__':
    unittest.main()