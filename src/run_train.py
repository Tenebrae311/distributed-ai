import torch
import torch.nn as nn
import torch.optim as optim
from pipeline.engine import train, val
from pipeline.dataset import create_train_val_datasets
from pipeline.model import FTTransformer
from pandas import read_csv
from pipeline.early_stopping import EarlyStopping

# raw data information
#table_path = "data/fraud-detection.csv"
#numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
#categorical_cols = ["step", "type", "nameOrig", "nameDest"]
#label_col = "isFraud"

#table_path = "data/dummy.csv"
#numeric_cols = ["age", "income"]  
#categorical_cols = ["country", "gender"]
#label_col = "label"

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
# create datasets
train_set, val_set = create_train_val_datasets(
    df,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    target_col=label_col,
    val_ratio=0.2
)
# dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
# model
model = FTTransformer(
    num_numeric=len(numeric_cols),
    cat_cardinalities=train_set.get_cat_cardinalities()
)
# training tools
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min', 
    factor=0.5, 
    patience=10,
    min_lr=1e-6
)

num_epochs = 200
early_stopper = EarlyStopping(
    patience=20,         # Anzahl Epochen ohne Verbesserung
    min_delta=0.0,       # Mindestverbesserung im Loss
    save_path="best_ft_transformer.pt"
)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, roc_auc, precision = val(model, val_loader)

    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"ROC AUC: {roc_auc:.4f}, Precision: {precision:.4f}, LR: {current_lr:.6f}")
    
    early_stopper.step(val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping triggered. Training stopped.")
        break
