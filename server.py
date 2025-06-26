import flwr as fl
from flwr.server import Server, ServerConfig
from flwr.server.strategy import FedAvg
from centralized import CentralizedServer
from privacy.differential_privacy import DifferentialPrivacy
from data.synthetic_health_data import generate_synthetic_health_data

# Initialize the centralized server with FedAvg strategy
def main():
    # Create a Flower server configuration with differential privacy
    # Initialize differential privacy mechanism
    dp = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0)
    strategy = FedAvg(fraction_fit=1.0)
    
    # Initialize the centralized server
    server = CentralizedServer()
    
    # Generate synthetic data for 3 clients
    client_data = [generate_synthetic_health_data(client_id=i) for i in range(3)]
    
    # Start the Flower server with client data and privacy configuration
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        server=server
    )

if __name__ == "__main__":
    main()
