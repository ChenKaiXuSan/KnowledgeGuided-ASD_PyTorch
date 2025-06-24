import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.CrossEntropyLoss()(self.forward(x), y)
        print(f"[Rank {self.global_rank}] Batch {batch_idx} Loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def get_dataloader():
    x = torch.randn(100, 32)
    y = torch.randint(0, 2, (100,))
    return DataLoader(TensorDataset(x, y), batch_size=16, num_workers=0)

def main():
    model = SimpleModel()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0, 1],
        strategy="auto",  # <- 更安全
        max_epochs=10,
        log_every_n_steps=10,
    )
    trainer.fit(model, get_dataloader())

if __name__ == "__main__":
    main()
