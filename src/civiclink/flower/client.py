import flwr as fl
import pandas as pd
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.run_train import TrainingPipeline

class TaxFLClient(fl.client.NumPyClient):
    def __init__(self, table_path, numeric_cols, categorical_cols, label_col, cid, num_clients, num_local_epochs=1):
        self.cid = int(cid)
        self.num_clients = num_clients
        self.num_local_epochs = num_local_epochs

        # load total dataset
        df = pd.read_csv(table_path, sep=";")

        # split in num_clients parts to simulate differents datasets
        df_client = self.split_dataframe(df, self.cid, self.num_clients)

        # create pipeline with client dataframe
        self.pipeline = TrainingPipeline(
            df=df_client,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            label_col=label_col,
            batch_size=32,
            lr=5e-4,
            weight_decay=1e-4,
        )

    @staticmethod
    def split_dataframe(df, cid, num_clients):
        """Teilt das DataFrame gleichmäßig in num_clients Teile auf"""
        total = len(df)
        chunk_size = total // num_clients

        start = cid * chunk_size
        end = (cid + 1) * chunk_size if cid < num_clients - 1 else total

        return df.iloc[start:end].reset_index(drop=True)

    def get_parameters(self, config=None):
        return self.pipeline.get_parameters()

    def fit(self, parameters, config=None):
        self.pipeline.set_parameters(parameters)
        
        # local training over multiple epochs
        train_loss = 0.0
        for _ in range(self.num_local_epochs):
            train_loss += self.pipeline.train_one_epoch()
            self.pipeline.scheduler.step(train_loss)

        new_params = self.pipeline.get_parameters()
        num_samples = len(self.pipeline.train_set)
        return new_params, num_samples, {"train_loss": train_loss / self.num_local_epochs}
    
    def evaluate(self, parameters, config=None):
        self.pipeline.set_parameters(parameters)
        val_loss, roc_auc, precision, mean_prob = self.pipeline.evaluate_once()
        num_samples = len(self.pipeline.val_set)

        return val_loss, num_samples, {
            "roc_auc": roc_auc,
            "precision": precision,
            "mean_prob": mean_prob
        }
    
def start_client():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()

    table_path = "data/fake_steuerdaten_labels_not_random.csv"

    numeric_cols = [
        "Summe_Einkuenfte_Brutto", 
        "Summe_Werbungskosten", 
        "Summe_Sonderausgaben", 
        "Summe_Ausserg_Belastungen", 
        "Erstattungsbetrag_Erwartet",
        "Anzahl_Tage_Homeoffice",
        "Entfernung_Wohnung_Arbeit",
        "Kosten_Arbeitsmittel",
        "Kosten_Bewirtung",
        "Kosten_Geschaeftsreisen",
        "Alter",
        "Anzahl_Kinder",
        "Veraenderung_Einkommen_Vj_Prozent",
        "Veraenderung_Werbungskosten_Vj_Prozent",
        "Veraenderung_Spenden_Vj_Prozent",
        "Differenz_Einkommen_Lohnbescheid",
        "Differenz_Kapitalertraege_Bank",
        "Differenz_Rente_Meldung",
        "Ratio_Werbungskosten_zu_Einkommen",
        "Ratio_Spenden_zu_Einkommen",
        "Ratio_Krankheitskosten_zu_Einkommen",
        "Ratio_Gewinn_zu_Umsatz",
        "Abweichung_Werbungskosten_von_Berufsgruppe",
        "Abweichung_Gewinnmarge_von_Branche"
    ]

    categorical_cols = [
        "Familienstand",
        "Steuerklasse",
        "Bundesland",
        "Religionszugehörigkeit",
        "Einkunftsart",
        "Branche_Selbststaendig",
        "Hat_Anlage_N",
        "Hat_Anlage_V",
        "Hat_Anlage_KAP",
        "Hat_Anlage_Kind",
        "Hat_Anlage_G"
    ]

    label_col = "Label"

    client = TaxFLClient(
        table_path=table_path,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        label_col=label_col,
        cid=args.cid,
        num_clients=args.num_clients,
        num_local_epochs=5
    )

    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=client
    )


if __name__ == "__main__":
    start_client()
