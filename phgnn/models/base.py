"""Shared Lightning logic of the seismic regressors: loss, L2 penalty, optimizer, prediction."""
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


def conv_encoder(in_channels, hidden, kernel_size, stride):
    """Return the two Conv1d layers of the TSER-GCN waveform encoder."""
    return (nn.Conv1d(in_channels, hidden, kernel_size=kernel_size, stride=stride),
            nn.Conv1d(hidden, hidden * 2, kernel_size=kernel_size, stride=stride))


def encoded_size(conv1, conv2, window, in_channels):
    # torch.randn (not zeros) keeps the random stream, and so the initial weights, of the reported runs.
    with torch.no_grad():
        out = conv2(conv1(torch.randn(1, in_channels, window)))
    return out.shape[1] * out.shape[2]


def l2_penalty(reg_const, conv_kernels, graph_layers):
    """Return c * sum(w**2) over the conv and graph kernels (Keras regularizers.l2)."""
    graph_params = [p for layer in graph_layers for p in layer.parameters()]
    return reg_const * sum((w ** 2).sum() for w in [*conv_kernels, *graph_params])


class SeismicRegressor(L.LightningModule):
    """Base class; subclasses implement predict(x, coords) -> (predictions, l2_penalty)."""

    def __init__(self, train_config):
        super().__init__()
        self.lr = train_config["lr"]
        self.reg_const = train_config["reg_const"]
        self.optimizer_name = train_config["optimizer"]
        self.lr_schedule = train_config["lr_schedule"]
        self.weight_decay = train_config["weight_decay"]
        self.pct_start = train_config["pct_start"]
        self.final_div = train_config["final_div"]

    def split_batch(self, batch):
        """(waveforms, station coordinates, targets)."""
        x, coords, y = batch
        return x, coords, y

    def data_loss(self, pred, target):
        """Return the sum over the five outputs of their mean squared errors (as Keras does)."""
        return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 1)).sum()

    def training_step(self, batch, batch_idx):
        x, coords, y = self.split_batch(batch)
        pred, l2 = self.predict(x, coords)
        loss = self.data_loss(pred, y) + l2
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, coords, y = self.split_batch(batch)
        pred, _ = self.predict(x, coords)
        loss = self.data_loss(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def raw_predict(self, batch):
        """Return (predictions, targets) as numpy arrays [B, N, 5]."""
        x, coords, y = (t.to(self.device) for t in self.split_batch(batch))
        pred, _ = self.predict(x, coords)
        return pred.float().cpu().numpy(), y.cpu().numpy()

    def configure_optimizers(self):
        """RMSprop at a constant rate (the published setup), Adam (KIM-GNN), or fastai's Adam with a one-cycle schedule."""
        if self.optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.9)
        elif self.optimizer_name == "adam":       # torch Adam with coupled (L2) weight decay, as in Kim et al.'s code
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "fastai_adam":  # fastai's Adam: mom 0.9, sqr_mom 0.99, eps 1e-5, decoupled weight decay
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-5,
                                          weight_decay=self.weight_decay)
        else:
            raise ValueError(f"optimizer must be rmsprop, adam or fastai_adam, got {self.optimizer_name!r}")
        if self.lr_schedule != "one_cycle":
            return optimizer
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, pct_start=self.pct_start,
                                                       final_div_factor=self.final_div,
                                                       total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [dict(scheduler=schedule, interval="step")]
