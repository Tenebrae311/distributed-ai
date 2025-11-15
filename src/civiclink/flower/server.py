import flwr as fl
from flwr.server.strategy import FedAvg

# Aggregationsfunktion für deine custom Metriken:
def weighted_average(metrics_list):
    """Aggregiert Metriken mit Gewichtung nach Anzahl Samples."""
    # metrics_list: List von Tuples (num_examples, metrics_dict) pro Client
    total_examples = sum(num for num, _ in metrics_list)
    agg = {}
    for _, metrics in metrics_list:
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + v * (_ / total_examples)
    return agg

# Strategy konfigurieren
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Server starten
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)
