import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(
        self,
        num_numeric,
        cat_cardinalities,
        d_model=24,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        dim_feedforward=128
    ):
        super().__init__()

        # Numerische Embeddings
        self.num_embedding = nn.Linear(num_numeric, d_model)

        # Kategorische Embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities
        ])

        # Transformer Encoder (tiefer, größere FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            norm_first=True  # wichtig für Stabilität (Pre-Norm Transformer)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Kleinere Head, weniger Kapazität um Overfitting zu reduzieren
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 1 Output für binäre Regression (logit)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # Numerische Features → d_model Dimension
        num_emb = self.num_embedding(x_num).unsqueeze(1)

        # Kategorische Features einbetten + Validierung
        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_indices = x_cat[:, i]

            if torch.any(cat_indices < 0) or torch.any(cat_indices >= emb.num_embeddings):
                raise ValueError(
                    f"Kategorisches Feature {i} hat ungültige Indizes: "
                    f"[{cat_indices.min().item()}, {cat_indices.max().item()}], "
                    f"erwartet 0–{emb.num_embeddings-1}"
                )

            cat_embs.append(emb(cat_indices).unsqueeze(1))

        # Token zusammenfügen
        tokens = torch.cat([num_emb] + cat_embs, dim=1)

        # Transformer
        encoded = self.transformer(tokens)

        # CLS (numerisches Token)
        cls_token = encoded[:, 0]

        # Output → Logit
        return self.head(cls_token).squeeze(1)
