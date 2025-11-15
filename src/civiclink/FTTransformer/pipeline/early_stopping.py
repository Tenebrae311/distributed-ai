import torch

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, verbose=True, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path
        
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss, model):
        # Verbesserung?
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            # bestes Modell speichern
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"✔ Model improved — saving checkpoint to {self.save_path}")

        else:
            self.counter += 1
            if self.verbose:
                print(f"✗ No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
