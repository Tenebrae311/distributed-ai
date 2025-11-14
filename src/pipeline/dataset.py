import torch
from torch.utils.data import Dataset
from pandas import DataFrame

class TabularDataset(Dataset):
    def __init__(self, df: DataFrame, numeric_cols: list[str], categorical_cols: list[str], target_col: str):
        """
        df: Pandas DataFrame
        numeric_cols: Liste der numerischen Spaltennamen
        categorical_cols: Liste der kategorischen Spaltennamen (können strings sein)
        target_col: Zielspalte (0 = korrekt, 1 = fehlerhaft für Regression, oder 0/1/2 für Klassifikation)
        """

        # Numerische Features
        self.x_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

        # Kategorische Features → immer als Strings behandeln, dann zu Integer
        cat_data = df[categorical_cols].astype(str).copy()
        self.cat_mappings = {}
        self.categorical_cols = categorical_cols

        for col in categorical_cols:
            unique_vals = sorted(cat_data[col].unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.cat_mappings[col] = mapping
            cat_data[col] = cat_data[col].map(mapping)

        self.x_cat = torch.tensor(cat_data.values, dtype=torch.long)

        # Zielwerte
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32)  # Regression: float, Klassifikation: long ggf. ändern

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]
    
    def get_cat_cardinalities(self):
        """
        Gibt die Kardinalitäten (Anzahl einzigartiger Werte) für kategorische Features zurück.
        
        Returns:
            list[int]: Liste der Kardinalitäten in der Reihenfolge der categorical_cols
        """
        return [len(self.cat_mappings[col]) for col in self.categorical_cols]

def create_train_val_datasets(
        df: DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        target_col: str,
        val_ratio: float = 0.5,
        shuffle: bool = True,
        seed: int = 42
    ):
    """
    Erstellt Train- und Validierungs-Datasets aus einem DataFrame.
    """

    # ggf. mischen
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split-Index
    val_size = int(len(df) * val_ratio)

    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]

    # Datasets erzeugen
    train_set = TabularDataset(df_train, numeric_cols, categorical_cols, target_col)
    val_set   = TabularDataset(df_val, numeric_cols, categorical_cols, target_col)

    return train_set, val_set
