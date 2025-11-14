import flwr as fl

# Startet den Server und aggregiert Gewichte mit FedAvg
if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(),
    )
