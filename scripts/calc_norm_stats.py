from pathlib import Path
from tqdm import tqdm

import pytorch_lightning as pl

from src.data.affectnet_datamodule import AffectNetImageDataModule

def calc_norm_stats(train_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for X, y, s in tqdm(train_loader):
        batch_samples = X.size(0)
        X = X.view(batch_samples, X.size(1), -1)
        mean += X.mean(2).sum(0)
        std += X.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

def main(args):
    pl.seed_everything(args.seed)

    dm = AffectNetImageDataModule(label_type='expression',
                                  data_root=args.dataroot,
                                  val_split=args.val_split,
                                  batch_size=args.batch_size,
                                  refresh_cache=args.refresh_cache,
                                  num_workers=0)
    dm.prepare_data()
    dm.setup(stage='fit')

    mean, std = calc_norm_stats(dm.train_dataloader())
    print(mean)
    print(std)

if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--dataroot', required=True, type=Path,
                        help=f'Path to root of data directory')
    parser.add_argument('--val-split', required=True, type=float,
                        help=f'Validation percentage')
    parser.add_argument('--batch-size', required=True, type=int,
                        help=f'Batch Size')
    parser.add_argument('--seed', required=False, type=int, default=42,
                        help=f'Seed value to seed randomization')
    parser.add_argument('--refresh-cache', action='store_true',
                        help=f'Refresh data module cache')

    main(parser.parse_args())
