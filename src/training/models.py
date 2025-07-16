import torch
import torch.nn as nn
import pytorch_lightning as pl


# Base class for common steps
class BasePredictor(pl.LightningModule):
    criterion = nn.MSELoss()

    def forward(self, emb1, emb2):
        # This must be implemented by subclasses
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Assumes batch structure (query_emb, target_emb, optional_target)
        # Subclasses might override if batch structure differs during prediction
        query_emb, target_emb, _ = batch
        return self(query_emb, target_emb)

    def _common_step(self, batch, batch_idx):
        query_emb, target_emb, param_value = batch
        predictions = self(query_emb, target_emb)  # Calls the subclass's forward method
        loss = self.criterion(predictions, param_value)
        return loss, predictions, param_value

    def configure_optimizers(self):
        # Access learning_rate from hparams saved by subclasses
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class FNNPredictor(BasePredictor):  # Inherit from BasePredictor
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()  # Saves embedding_size, hidden_size, learning_rate

        # Model layers (specific to FNN)
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
        # Criterion is inherited from BasePredictor

    def forward(self, emb1, emb2):
        proc1 = self.individual_layers(emb1)
        proc2 = self.individual_layers(emb2)
        combined = torch.cat([proc1, proc2], dim=1)
        return self.combined_layers(combined).squeeze()

    # training_step, validation_step, predict_step, _common_step, configure_optimizers are inherited


class LinearRegressionPredictor(BasePredictor):  # Inherit from BasePredictor
    def __init__(self, embedding_size: int, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        # Simple linear layer operating on the concatenated embeddings
        self.linear = nn.Linear(embedding_size * 2, 1)
        # Criterion is inherited

    def forward(self, emb1, emb2):
        combined = torch.cat([emb1, emb2], dim=1)
        return self.linear(combined).squeeze()

    # training_step, validation_step, predict_step, _common_step, configure_optimizers are inherited


class LinearDistancePredictor(BasePredictor):  # Inherit from BasePredictor
    def __init__(self, embedding_size: int, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        # Linear layer operating on the element-wise squared difference
        self.linear = nn.Linear(embedding_size, 1)
        # Criterion is inherited

    def forward(self, emb1, emb2):
        # Calculate element-wise squared difference
        diff_sq = (emb1 - emb2).pow(2)
        return self.linear(diff_sq).squeeze()

    # training_step, validation_step, predict_step, _common_step, configure_optimizers are inherited
