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
        cat_mappings: dict[str, dict] | None = None,
    ):
        """
        Wenn cat_mappings=None: neue Mappings erzeugen (nur beim Training!)
        Wenn cat_mappings!=None: dieselben Mappings verwenden (Validation!)
        """

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # Numerische Features
        self.x_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

        # Kategorische Features immer als string
        df_cat = df[categorical_cols].astype(str).copy()

        # Wenn keine Mappings gegeben → neue erzeugen
        self.cat_mappings = {} if cat_mappings is None else cat_mappings

        if cat_mappings is None:
            # Trainings-Mappings erstellen
            for col in categorical_cols:
                unique_vals = sorted(df_cat[col].unique())
                mapping = {val: idx+1 for idx, val in enumerate(unique_vals)}
                mapping["<UNK>"] = 0
                self.cat_mappings[col] = mapping
        else:
            # Validation: neue Kategorien → UNK
            for col in categorical_cols:
                df_cat[col] = df_cat[col].apply(
                    lambda x: x if x in self.cat_mappings[col] else "<UNK>"
                )

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

    # Train-Set erzeugt Mappings
    train_set = TabularDataset(
        df_train, numeric_cols, categorical_cols, target_col, cat_mappings=None
    )

    # Validation-Set nutzt dieselben Mappings → Konsistenz!
    val_set = TabularDataset(
        df_val,
        numeric_cols,
        categorical_cols,
        target_col,
        cat_mappings=train_set.cat_mappings,
    )

    return train_set, val_set
