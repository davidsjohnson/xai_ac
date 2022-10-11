import random
from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from src.data.affectnet_datamodule import AffectNetAUDataModule

class TestAffectNetAUDataModule:

    def test_datamodule(self):

        self.label_type = 'arousal'
        self.label_name = 'aro'
        self.val_split = 0.2
        self.batch_size = 32

        dm = AffectNetAUDataModule(self.label_type, val_split=self.val_split, batch_size=self.batch_size)
        dm.prepare_data()
        dm.setup()

        self.feature_cols = dm.train_dataset.dataset.feature_names    # get feature columns for testing

        #### Check Validation size
        len_train = len(dm.train_dataset.indices)
        len_val = len(dm.val_dataset.indices)
        assert len_val / (len_train + len_val) == self.val_split

        #### Train Loader Testing
        dl_train: DataLoader = dm.train_dataloader()
        self._dl_tester(dl_train, 'train', self.label_name, Path('data/train_set'))

        #### Val Loader Testing
        dl_val: DataLoader = dm.val_dataloader()
        self._dl_tester(dl_val, 'val', self.label_name, Path('data/train_set'))

        #### Test Loader Testing
        dl_test: DataLoader = dm.test_dataloader()
        self._dl_tester(dl_test, 'test', self.label_name, Path('data/val_set'))

    def _dl_tester(self, dataloader: DataLoader, partition: str, label_name: str, datapath: Path):

        data_iter = iter(dataloader)
        x_batch, y_batch, ids_batch = next(data_iter)

        if partition in ['train', 'val']:
            assert x_batch.shape ==  (self.batch_size, 35)
        else:
            assert x_batch.shape == (1, 35)

        # check random data
        batch_idx = random.randint(0, self.batch_size-1) if partition in ['train', 'val'] else 0
        x = x_batch[batch_idx]
        y = y_batch[batch_idx]
        img_id = ids_batch[batch_idx]

        df_aus = pd.read_csv(datapath / 'aus' / f'{img_id}.csv')
        x_exp = df_aus[self.feature_cols]
        y_exp = float(np.load(datapath / 'annotations' / f'{img_id}_{label_name}.npy').item())

        assert np.allclose(x_exp, x)
        assert y == y_exp