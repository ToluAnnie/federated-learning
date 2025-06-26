import flwr as fl
from flwr.server import Server, ServerConfig
from flwr.server.strategy import FedAvg
from centralized import CentralizedServer
from privacy.differential_privacy import DifferentialPrivacy

# Initialize the centralized server with FedAvg strategy
def main():
    # Create a Flower server configuration
    strategy = FedAvg(fraction_fit=1.0)
    
    # Initialize the centralized server
    server = CentralizedServer()
    
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        server=server
    )

if __name__ == "__main__":
    main()
