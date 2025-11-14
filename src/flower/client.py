import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import read_csv

from pipeline.model import FTTransformer
from pipeline.engine import train, val
from pipeline.dataset import create_train_val_datasets

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        
        # Optimizer und Criterion für Training
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self, config):
        """Modell-Parameter als NumPy Arrays zurückgeben"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Modell-Parameter von NumPy Arrays setzen"""
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        """Training für eine Runde"""
        # Parameter vom Server setzen
        self.set_parameters(parameters)
        
        # Training für eine Epoche (Ihre train Funktion erwartet optimizer und criterion)
        train_loss = train(self.model, self.trainloader, self.optimizer, self.criterion)
        
        # Aktualisierte Parameter zurückgeben
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"train_loss": float(train_loss)}

    def evaluate(self, parameters, config):
        """Evaluation auf Testdaten"""
        # Parameter vom Server setzen
        self.set_parameters(parameters)
        
        # Validation (Ihre val Funktion gibt (loss, roc_auc, precision) zurück)
        val_loss, roc_auc, precision = val(self.model, self.testloader)
        
        # Metrics als Dictionary zurückgeben
        metrics = {
            "roc_auc": float(roc_auc),
            "precision": float(precision)
        }
        
        return float(val_loss), len(self.testloader.dataset), metrics

def create_client(client_id: int = 0):
    """Client mit tabellarischen Daten erstellen"""
    
    # Tabellendaten laden
    table_path = "data/fraud-detection.csv"
    numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    categorical_cols = ["step", "type", "nameOrig", "nameDest"]
    label_col = "isFraud"
    
    df = read_csv(table_path)
    
    # Dataset erstellen (jeder Client bekommt einen Teil der Daten)
    # Hier können Sie die Daten je nach client_id aufteilen
    total_size = len(df)
    start_idx = (client_id * total_size) // 3  # Annahme: 3 Clients
    end_idx = ((client_id + 1) * total_size) // 3
    
    client_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Train/Val Split für diesen Client
    train_set, val_set = create_train_val_datasets(
        client_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=label_col,
        val_ratio=0.2
    )
    
    # DataLoader erstellen
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
    
    # Modell erstellen
    model = FTTransformer(
        num_numeric=len(numeric_cols),
        cat_cardinalities=train_set.get_cat_cardinalities()
    )
    
    return FlowerClient(model, trainloader, testloader)

def main():
    """Client starten"""
    import sys
    
    # Client ID aus Kommandozeile oder Standard
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    print(f"Starting Flower Client {client_id}")
    
    # Client erstellen
    client = create_client(client_id)
    
    # Client mit Server verbinden
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )

if __name__ == "__main__":
    main()