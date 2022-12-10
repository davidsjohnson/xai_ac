from pathlib import Path

import torch
import torch.utils.data
import pytorch_lightning as pl
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import default_collate

from torchsummaryX import summary

from src.models.models import AlexNet
from src.models.lightning_models import LightningClassification

pl.seed_everything(42)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def main(args):

    batch_size = 256
    val_split = 0.1

    final_activation = 'softmax'
    optim = torch.optim.SGD
    optim_params = dict(
        lr = 0.001, momentum=0.9
    )
    loss = torch.nn.CrossEntropyLoss()

    ## Setup Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    ds_train = CIFAR10('unittests/testdata', train=True, download=True, transform=transform)
    ds_test = CIFAR10('unittests/testdata', train=False, download=True, transform=transform)

    # make dataloader here
    val_split_size = int(len(ds_train) * val_split)
    train_split_size = len(ds_train) - val_split_size
    ds_train, ds_val = torch.utils.data.random_split(ds_train, [train_split_size, val_split_size])

    # collate function to add third element to batch
    def dummy_collate(data):
        batch = default_collate(data)
        subs = torch.zeros(len(batch[0]))
        batch.append(subs)
        return batch

    workers = 12

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=dummy_collate, num_workers=workers)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=dummy_collate, num_workers=workers)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=dummy_collate, num_workers=workers)

    num_classes = len(ds_test.classes)

    ## Setup Model
    # load alexnet and modify output layer for new number of classes
    model = AlexNet(n_classes=num_classes, pretrained=args.pretrained)
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
    trainer.fit(net, train_dataloaders=dl_train, val_dataloaders=dl_val)

    net = LightningClassification.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model=model,
        final_activation=final_activation,
        optimizer=optim,
        optimizer_params=optim_params,
        loss_fn=loss
    )

    eval_results = trainer.test(net, dl_test)
    print(eval_results)

if __name__ == '__main__':
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help=f'Path to store output of training, including checkpoints')
    parser.add_argument('--pretrained', action='store_true',
                        help=f'Used Pretrained AlexNet Weights')
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help=f'Number of epochs to train model.')
    parser.add_argument('--refresh-cache', action='store_true',
                        help=f'Refresh data module cache')

    main(parser.parse_args())