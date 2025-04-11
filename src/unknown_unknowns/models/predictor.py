import torch
import torch.nn as nn
import pytorch_lightning as pl

class ModelPredictor(pl.LightningModule):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Model layers
        self.individual_layers = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
        )

        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.criterion = nn.MSELoss()

    def forward(self, emb1, emb2):
        proc1 = self.individual_layers(emb1)
        proc2 = self.individual_layers(emb2)
        combined = torch.cat([proc1, proc2], dim=1)
        return self.combined_layers(combined).squeeze()

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        query_emb, target_emb, _ = batch
        return self(query_emb, target_emb)

    def _common_step(self, batch, batch_idx):
        query_emb, target_emb, param_value = batch
        predictions = self(query_emb, target_emb)
        loss = self.criterion(predictions, param_value)
        return loss, predictions, param_value

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)