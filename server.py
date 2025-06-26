import flwr as fl
from flwr.server import Server, ServerConfig
from flwr.server.strategy import FedAvg
from centralized import CentralizedServer
from privacy.differential_privacy import add_differential_privacy

# Initialize the centralized server with FedAvg strategy
def main():
    # Create a Flower server configuration
    config = ServerConfig(num_rounds=5, fraction_fit=1.0, fraction_eval=1.0)
    
    # Initialize the centralized server
    server = CentralizedServer()
    
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=FedAvg(),
        server=server
    )

if __name__ == "__main__":
    main()