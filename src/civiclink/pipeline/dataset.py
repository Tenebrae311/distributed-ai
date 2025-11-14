import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, df, numeric_cols, categorical_cols, target_col):
        """
        df: Pandas DataFrame
        numeric_cols: Liste der numerischen Spaltennamen
        categorical_cols: Liste der kategorischen Spaltennamen
        target_col: Zielspalte (Werte: 0,1,2)
        """

        # Werte als numpy → torch
        self.x_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

        # kategorische Features zu LongTensor (Pflicht für nn.Embedding)
        self.x_cat = torch.tensor(df[categorical_cols].values, dtype=torch.long)

        # Zielwerte → Klassenlabels 0,1,2
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]
