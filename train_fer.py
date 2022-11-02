from pathlib import Path

import torch
import torch.utils.data
import pytorch_lightning as pl
from torchvision import models
from torchvision import transforms

from torchsummaryX import summary

from src.models.models import AlexNet
from src.models.lightning_models import LightningClassification
from src.data.affectnet_datamodule import AffectNetImageDataModule

pl.seed_everything(42)

MEAN = [0.5697, 0.4462, 0.3913]
STD = [0.2323, 0.2060, 0.1947]

def main(args):

    ## Init params
    label = 'expression'
    batch_size = 256
    val_split = 0.1

    final_activation = 'softmax'
    optim = torch.optim.Adam
    optim_params = dict(
        lr = 0.005
    )
    loss = torch.nn.CrossEntropyLoss()

    ## Setup Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    dm = AffectNetImageDataModule(label_type=label,
                                  data_root=args.dataroot,
                                  val_split=val_split,
                                  batch_size=batch_size,
                                  train_transform=transform,
                                  test_transform=transform,
                                  refresh_cache=args.refresh_cache,
                                  num_workers=12)

    ## Setup Model
    # load alexnet and modify output layer for new number of classes
    model = AlexNet(n_classes=dm.num_classes, pretrained=args.pretrained)
    summary(model, torch.zeros((1, 3, 224, 224)))

    net = LightningClassification(model=model,
                                  final_activation=final_activation,
                                  optimizer=optim,
                                  optimizer_params=optim_params,
                                  loss_fn=loss)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_weights_only=True, mode='min', monitor='val_loss'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step')
    ]
    trainer = pl.Trainer(default_root_dir=args.output / 'ckpts',
                         callbacks=callbacks,
                         max_epochs=args.epochs,
                         gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(net, dm)

    net = LightningClassification.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model=model,
        final_activation=final_activation,
        optimizer=optim,
        optimizer_params=optim_params,
        loss_fn=loss
    )

    eval_results = trainer.test(net, dm)
    print(eval_results)

if __name__ == '__main__':
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--dataroot', required=True, type=Path,
                        help=f'Path to root of data directory')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help=f'Path to store output of training, including checkpoints')
    parser.add_argument('--pretrained', action='store_true',
                        help=f'Used Pretrained AlexNet Weights')
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help=f'Number of epochs to train model.')
    parser.add_argument('--refresh-cache', action='store_true',
                        help=f'Refresh data module cache')

    main(parser.parse_args())