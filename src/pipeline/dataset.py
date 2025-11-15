import torch
from torch.utils.data import Dataset
from pandas import DataFrame


class TabularDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        target_col: str,
        cat_mappings: dict[str, dict],
    ):
        """
        Erwartet bereits erstellte cat_mappings.
        Neue Kategorien → UNK (Index 0)
        """

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.cat_mappings = cat_mappings

        # Numerische Features
        self.x_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

        # Kategorische Features immer als strings
        df_cat = df[categorical_cols].astype(str).copy()

        # Kategorien in Indices wandeln
        df_cat = df_cat.apply(lambda col: col.map(self.cat_mappings[col.name]))
        self.x_cat = torch.tensor(df_cat.values, dtype=torch.long)

        # Target
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]

    def get_cat_cardinalities(self):
        return [len(self.cat_mappings[col]) for col in self.categorical_cols]


def create_train_val_datasets(
    df: DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
):
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_size = int(len(df) * val_ratio)
    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]

    # ------------------------------------------------------------
    # Standardize numeric features using statistics from the training split
    # ------------------------------------------------------------
    if len(numeric_cols) > 0:
        # compute mean/std on train
        means = df_train[numeric_cols].mean()
        stds = df_train[numeric_cols].std().replace(0, 1.0)

        # apply standardization
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_train[numeric_cols] = (df_train[numeric_cols] - means) / stds
        df_val[numeric_cols] = (df_val[numeric_cols] - means) / stds

    # ------------------------------------------------------------
    # ZENTRAL: Kategorische Mappings erzeugen (nur einmal!)
    # ------------------------------------------------------------
    cat_mappings = {}
    for col in categorical_cols:
        unique_vals = sorted(df_train[col].astype(str).unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        cat_mappings[col] = mapping

    # ------------------------------------------------------------
    # Datasets erzeugen — beide nutzen die Mappings!
    # ------------------------------------------------------------
    train_set = TabularDataset(
        df_train, numeric_cols, categorical_cols, target_col, cat_mappings
    )

    val_set = TabularDataset(
        df_val, numeric_cols, categorical_cols, target_col, cat_mappings
    )

    return train_set, val_set
