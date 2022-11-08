import torch
import pytorch_lightning as pl
from torchmetrics import R2Score, SpearmanCorrCoef


class LightningClassification(pl.LightningModule):

    def __init__(self,
                 model,
                 final_activation,
                 optimizer,
                 optimizer_params,
                 loss_fn,
                 ):
        super(LightningClassification, self).__init__()

        # Get from config
        self._loss_fn = loss_fn
        self._optim = optimizer
        self._optim_params = optimizer_params
        self._final_activiation = final_activation
        self.net =  model

    def available_metrics(self):
        return ['loss', 'acc']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)

    def configure_optimizers(self):
        opt =  self._optim(self.parameters(), **self._optim_params)
        lr_scheduler = dict(
            scheduler = torch.optim.lr_scheduler.StepLR(opt, gamma=0.1, step_size=5000),
            interval = 'step',

        )
        return {
            "optimizer": opt,
            "lr_scheduler": lr_scheduler
        }

    def _process_batch(self, batch, batch_idx):
        x, y, s = batch
        y = torch.squeeze(y, dim=-1)  # get rid of extra dimensions for loss

        logits = self.forward(x)
        if self._final_activiation == 'softmax':
            pred = torch.softmax(logits, dim=1)
        elif self._final_activiation is None:
             pred = logits
        else:
            raise ValueError(f'Activation {self._final_activiation} not yet implemented')
        loss =  self._loss_fn(logits, y)    # cross entropy loss expects logits
        acc = (pred.argmax(dim=-1) == y).float().mean()

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0]
        logits = self(x)
        if self._final_activiation == 'softmax':
            out = torch.softmax(logits, dim=1)
        elif self._final_activiation is None:
            out = logits
        else:
            raise ValueError(f'Activation {self._final_activiation} not yet implemented')
        return out


class LightningRegression(pl.LightningModule):

    def __init__(self,
                 model_cls,
                 input_shape,
                 n_classes,
                 model_params,
                 final_activation,
                 optimizer,
                 optimizer_params,
                 loss_fn,
                 ):

        super(LightningRegression, self).__init__()

        # Get from config
        self._loss_fn = loss_fn
        self._optim = optimizer
        self._optim_params = optimizer_params
        self._final_activiation = final_activation
        self.net = model_cls(input_shape, n_classes, **model_params)

        self._r2_train = R2Score()
        self._r2_val = R2Score()
        self._r2_test = R2Score()
        self._spear_train = SpearmanCorrCoef()
        self._spear_val = SpearmanCorrCoef()
        self._spear_test = SpearmanCorrCoef()

    def available_metrics(self):
        return ['loss', 'r2', 'spearman']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)

    def configure_optimizers(self):
        return self._optim(self.parameters(), **self._optim_params)

    def _process_batch(self, batch, batch_idx):
        x, y, s = batch
        y = torch.squeeze(y, dim=-1)  # get rid of extra dimensions for loss

        logits = self.forward(x)
        if self._final_activiation is None:
            out = logits
        else:
            raise ValueError(f'Activation {self._final_activiation} not yet implemented')
        loss =  self._loss_fn(out, y)

        return loss, out

    def training_step(self, batch, batch_idx):
        loss, out = self._process_batch(batch, batch_idx)
        _, y, _ = batch
        self._r2_train(out, y)
        self._spear_train(out, y)
        self.log('train_loss', loss)
        self.log('train_r2', self._r2_train, prog_bar=True)
        self.log('train_spearman', self._spear_train, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_r2_epoch', self._r2_train)
        self.log('train_spearman_epoch', self._spear_train)

    def validation_step(self, batch, batch_idx):
        loss, out = self._process_batch(batch, batch_idx)
        _, y, _ = batch
        self._r2_val(out, y)
        self._spear_val(out, y)
        self.log('val_loss', loss)
        self.log('train_r2', self._r2_val, prog_bar=True)
        self.log('train_spearman', self._spear_val, prog_bar=True)
        return loss

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_r2_epoch', self._r2_val)
        self.log('val_spearman_epoch', self._spear_val)

    def test_step(self, batch, batch_idx):
        loss, out = self._process_batch(batch, batch_idx)
        _, y, _ = batch
        self._r2_test(out, y)
        self._spear_test(out, y)
        self.log('test_loss', loss)
        self.log('test_r2', self._r2_test, prog_bar=True)
        self.log('test_spearman', self._spear_test, prog_bar=True)
        return loss

    def test_epoch_end(self, outs):
        # log epoch metric
        self.log('test_r2_epoch', self._r2_test)
        self.log('test_spearman_epoch', self._spear_test)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0]
        logits = self(x)
        if self._final_activiation is None:
            out = logits
        else:
            raise ValueError(f'Activation {self._final_activiation} not yet implemented')
        return out
