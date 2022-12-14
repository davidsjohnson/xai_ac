from pathlib import Path

import torch
import torch.utils.data
import pytorch_lightning as pl
from torchvision import transforms
from torchmetrics import AUROC

from torchsummaryX import summary

from src.models.models import AlexNet, DenseNet, VGGVariant, ResNet18
from src.models.lightning_models import LightningClassification
from src.data.affectnet_datamodule import AffectNetImageDataModule

pl.seed_everything(42)

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

MEAN = [0.5697, 0.4462, 0.3913]
STD = [0.2323, 0.2060, 0.1947]

def main(args):

    ## Init params
    label = 'expression'
    batch_size = 400
    val_split = 0.1

    final_activation = 'softmax'
    loss_cls = torch.nn.CrossEntropyLoss

    mixup = True
    alpha = 0.2

    ## Setup Data
    mean = MEAN if not args.pretrained else IMGNET_MEAN
    std = STD if not args.pretrained else IMGNET_STD

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(scale=(0.02, 0.25)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
        ]),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize((224, 224))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize((224, 224))
    ])
    dm = AffectNetImageDataModule(label_type=label,
                                  data_root=args.dataroot,
                                  val_split=val_split,
                                  batch_size=batch_size,
                                  train_transform=train_transform,
                                  test_transform=test_transform,
                                  refresh_cache=args.refresh_cache,
                                  num_workers=18)

    ## Model Training
    # load alexnet and modify output layer for new number of classes
    if args.model.lower() == 'alexnet':
        optim = torch.optim.SGD
        optim_params = dict(
            lr=0.001, momentum=0.9
        )
        model = AlexNet(n_classes=dm.num_classes, pretrained=args.pretrained)
    elif args.model.lower() == 'densenet':
        optim = torch.optim.SGD
        optim_params = dict(
            lr=0.1,
        )
        model = DenseNet(n_classes=dm.num_classes, pretrained=args.pretrained)
    elif args.model.lower() in ['vgg', 'resnet']:
        optim = torch.optim.Adam
        optim_params = dict(
            lr=0.001,
        )
        if args.model.lower() == 'vgg':
            model = VGGVariant(input_shape=(3, 224, 224), n_classes=dm.num_classes)
        else:
            model = ResNet18(n_classes=dm.num_classes, pretrained=args.pretrained)
    else:
        raise ValueError(f'Invalid model name, {args.model}.  Model name should be one of [densenet, alexnet, vgg, resnet]')
    summary(model, torch.zeros((1, 3, 224, 224)))

    # this needs to be adjusted based on training/val split
    inverse_weights = torch.from_numpy(1.0/dm.class_counts).type(torch.float32)
    # inverse_weights = inverse_weights / dm.class_counts.sum() * dm.num_classes
    loss = loss_cls(weight=inverse_weights)
    net = LightningClassification(model=model,
                                  num_classes=dm.num_classes,
                                  final_activation=final_activation,
                                  optimizer=optim,
                                  optimizer_params=optim_params,
                                  loss_fn=loss,
                                  mixup=mixup,
                                  alpha=alpha)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_weights_only=True, mode='min', monitor='val_loss'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step')
    ]
    trainer = pl.Trainer(default_root_dir=args.output / 'ckpts',
                         callbacks=callbacks,
                         max_epochs=args.epochs,
                         gpus=1 if torch.cuda.is_available() else 0,
                         fast_dev_run=3 if args.debug else False)
    trainer.fit(net, dm)

    ### Evaluation of model
    ckpt_path = 'best' if not args.debug else None
    train_eval_results = trainer.test(net, dm.train_dataloader(), ckpt_path=ckpt_path)
    val_eval_results = trainer.test(net, dm.val_dataloader(), ckpt_path=ckpt_path)
    test_eval_results = trainer.test(net, dm, ckpt_path=ckpt_path)

    print('Train Results:', train_eval_results)
    print('Val Results:', val_eval_results)
    print('Test Results:', test_eval_results)

    # calc AUROC for test data
    auroc_test = AUROC(task='multiclass', num_classes=dm.num_classes, average='weighted')
    auroc_test_perlcass = AUROC(task='multiclass', num_classes=dm.num_classes, average=None)
    y_true = torch.from_numpy(dm.test_dataset.df[label].values)
    y_pred = trainer.predict(net, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_path)
    y_pred = torch.cat(y_pred) # concate all batches into one tensor for scoring

    roc_score = auroc_test(y_pred, y_true)
    roc_scores = auroc_test_perlcass(y_pred, y_true)

    print(f'AUC Score Weighted: {roc_score}')
    print(f'AUC Scores Per Class: {roc_scores}')


if __name__ == '__main__':
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--dataroot', required=True, type=Path,
                        help=f'Path to root of data directory')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help=f'Path to store output of training, including checkpoints')
    parser.add_argument('-m', '--model', default='alexnet', type=str,
                        help='Name of the model to use during training. Should be one of [densenet, alexnet, vgg, resnet]')
    parser.add_argument('--pretrained', action='store_true',
                        help=f'Used Pretrained AlexNet Weights')
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help=f'Number of epochs to train model.')
    parser.add_argument('--refresh-cache', action='store_true',
                        help=f'Refresh data module cache')
    parser.add_argument('--debug', action='store_true',
                        help=f'Run in debug mode')


    main(parser.parse_args())