import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities,
                 d_model=32, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()

        # Numerische Embeddings (ein Linear-Layer für alle numerischen Features)
        self.num_embedding = nn.Linear(num_numeric, d_model)

        # Kategorische Embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities
        ])

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head für 3 Klassen
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 3)   # <--- 3 Klassen
        )

    def forward(self, x_num, x_cat):
        # numerische Features → d_model Dimension
        num_emb = self.num_embedding(x_num).unsqueeze(1)

        # kategorische Features einzeln einbetten
        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_embs.append(emb(x_cat[:, i]).unsqueeze(1))

        # Token zusammenfügen
        tokens = torch.cat([num_emb] + cat_embs, dim=1)

        # Transformer
        encoded = self.transformer(tokens)

        # CLS-Token (erster Token)
        cls_token = encoded[:, 0]

        # (B, 3)
        return self.head(cls_token)
