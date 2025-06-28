import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from typing import Dict, Any, Callable
import torch
from models.federated_logistic_regression import FederatedLogisticRegression

def get_on_fit_config_fn() -> Callable[[int], Dict[str, Any]]:
    """Returns a function that provides a custom config for the fit round."""
    def fit_config(server_round: int) -> Dict[str, Any]:
        """Returns the configuration for a fit round."""
        config = {
            "num_epochs": 10,
        }
        return config
    return fit_config

def main():
    """Starts the Flower server."""

    # Create an instance of the model to get its initial parameters.
    initial_model = FederatedLogisticRegression(input_dim=5)
    
    # Fix: Use .detach().cpu().numpy() to safely convert the parameters.
    initial_parameters = ndarrays_to_parameters(
        [p.detach().cpu().numpy() for p in initial_model.parameters()]
    )

    # Initialize the FedAvg strategy with custom configurations.
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        fraction_evaluate=1.0,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=initial_parameters,
        on_fit_config_fn=get_on_fit_config_fn(),
    )

    # Create a Flower server configuration.
    config = ServerConfig(num_rounds=5)

    # Start the Flower server with the configured strategy.
    # Note: The CentralizedServer is not a standard part of flwr.server.start_server,
    # so we can use a standard Server instance.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
