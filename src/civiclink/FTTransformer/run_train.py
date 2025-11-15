import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler, DataLoader

from pandas import read_csv

from src.FTTransformer.pipeline.engine import train, val
from src.FTTransformer.pipeline.dataset import create_train_val_datasets, TabularDataset
from src.FTTransformer.pipeline.model import FTTransformer
from src.FTTransformer.pipeline.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from pandas import DataFrame


class TrainingPipeline:
    def __init__(
        self,
        df: DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        label_col: str,
        batch_size: int = 16,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        val_ratio: float = 0.2,
        use_sampler: bool = True,
        max_pos_weight: float = 10.0,
        grad_clip: float = 0.5,
    ):
        self.df = df
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.label_col = label_col

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_ratio = val_ratio
        self.use_sampler = use_sampler
        self.max_pos_weight = max_pos_weight
        self.grad_clip = grad_clip

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # will be filled during pipeline steps
        self.train_set, self.val_set = self.load_data()
        self.train_loader, self.val_loader = self.prepare_dataloaders()
        self.model, self.optimizer = self.build_model()
        self.criterion = self.compute_class_weights()
        self.scheduler, self.early_stopper = self.setup_training_controls()

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "roc_auc": [],
            "precision": [],
        }


    # --------------------------------------------------------------
    # 1. Load and split the data
    # --------------------------------------------------------------
    def load_data(self) -> tuple[TabularDataset, TabularDataset]:
        return create_train_val_datasets(
            self.df,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            target_col=self.label_col,
            val_ratio=self.val_ratio,
        )


    # --------------------------------------------------------------
    # 2. Prepare dataloaders (with optional sampler)
    # --------------------------------------------------------------
    def prepare_dataloaders(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if self.use_sampler:
            labels = self.train_set.y.cpu().numpy().astype(int)
            class_sample_count = [(labels == t).sum() for t in sorted(set(labels))]
            class_weights = {i: 1.0 / max(1, c) for i, c in enumerate(class_sample_count)}
            sample_weights = [class_weights[int(l)] for l in labels]

            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

            print(f"Using WeightedRandomSampler: class_counts={class_sample_count}")

            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader


    # --------------------------------------------------------------
    # 3. Model creation
    # --------------------------------------------------------------
    def build_model(self) -> tuple[nn.Module, optim.Optimizer]:
        model = FTTransformer(
            num_numeric=len(self.numeric_cols),
            cat_cardinalities=self.train_set.get_cat_cardinalities(),
            dropout=0.3,
        ).to(self.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return model, optimizer


    # --------------------------------------------------------------
    # 4. Compute pos_weight for BCEWithLogitsLoss
    # --------------------------------------------------------------
    def compute_class_weights(self) -> nn.Module:
        labels = self.train_set.y
        num_pos = int((labels == 1).sum().item())
        num_neg = len(self.train_set) - num_pos

        if num_pos > 0:
            pos_weight_val = float(num_neg / num_pos)
        else:
            pos_weight_val = 1.0

        # cap pos_weight
        # If using a sampler we already balance batches — avoid double-weighting
        if self.use_sampler:
            pos_weight_capped_val = 1.0
        else:
            pos_weight_capped_val = float(min(pos_weight_val, self.max_pos_weight))

        print(
            f"Computed pos_weight: {pos_weight_val:.3f} (pos={num_pos}, neg={num_neg}), effective {pos_weight_capped_val:.3f}"
        )

        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_capped_val, dtype=torch.float32, device=self.device)
        )


    # --------------------------------------------------------------
    # 5. LR scheduler + early stopping
    # --------------------------------------------------------------
    def setup_training_controls(self) -> tuple[optim.lr_scheduler.ReduceLROnPlateau, EarlyStopping]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        early_stopper = EarlyStopping(
            patience=30,
            min_delta=0.0,
            save_path="best_ft_transformer.pt",
        )
        return scheduler, early_stopper


    # --------------------------------------------------------------
    # 6. Execute training loop
    # --------------------------------------------------------------
    def run(self, num_epochs: int = 300):
        for epoch in range(num_epochs):
            train_loss = train(
                self.model, self.train_loader, self.optimizer, self.criterion, self.device, grad_clip=self.grad_clip
            )
            val_loss, roc_auc, precision, mean_prob = val(
                self.model, self.val_loader, self.device, self.criterion
            )

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"ROC AUC: {roc_auc:.4f}, Precision: {precision:.4f}, "
                f"MeanProb: {mean_prob:.4f}, LR: {current_lr:.6f}"
            )
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["roc_auc"].append(roc_auc)
            self.history["precision"].append(precision)

            self.early_stopper.step(val_loss, self.model)
            if self.early_stopper.early_stop:
                print("Early stopping triggered.")
                break
        # self.plot_training()

    def plot_training(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss plot
        plt.figure(figsize=(10,5))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
        plt.legend(); plt.title("Training & Validation Loss")
        plt.savefig("training_validation_loss.png")

        # Metrics plot
        plt.figure(figsize=(10,5))
        plt.plot(epochs, self.history["roc_auc"], label="ROC AUC")
        plt.plot(epochs, self.history["precision"], label="Precision")
        plt.xlabel("Epoch"); plt.ylabel("Metric Value"); plt.grid(True)
        plt.legend(); plt.title("Validation Metrics")
        plt.savefig("validation_metrics.png")

    def get_parameters(self):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            tensor = torch.tensor(new_p, dtype=p.data.dtype)
            p.data = tensor.to(self.device)

    def train_one_epoch(self):
        return train(
            self.model,
            self.train_loader,
            self.optimizer,
            self.criterion,
            self.device,
            grad_clip=self.grad_clip
        )

    def evaluate_once(self):
        return val(
            self.model,
            self.val_loader,
            self.device,
            self.criterion
        )


def main():
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

    df = read_csv(table_path, sep=";")

    pipeline = TrainingPipeline(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        label_col=label_col,
        batch_size=32,
        lr=5e-4,
        weight_decay=5e-4,
    )
    pipeline.run(num_epochs=300)
   
if __name__ == "__main__":
    main()