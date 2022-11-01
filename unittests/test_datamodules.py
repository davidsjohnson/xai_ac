import random
import pytest
from pathlib import Path
from typing import Union

import numpy as np
import numpy.testing
import pandas as pd
from PIL import Image

import torch.testing
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.affectnet_datamodule import AffectNetAUDataModule, AffectNetImageDataModule

class TestAffectNetImageDataModule:

    def test_datamodule(self):
        """
        Testing data module with no transforms applied

        """
        data_root = Path('unittests/testdata/affectnet/raw/')
        self.label_type = 'arousal'
        self.label_name = 'aro'
        self.val_split = 0.2
        self.batch_size = 4 # small batch size for small test dataset

        dm = AffectNetImageDataModule(self.label_type, data_root=data_root, name='affectnet_img_debug',
                                      val_split=self.val_split, batch_size=self.batch_size,
                                      refresh_cache=True, num_workers=0)
        dm.prepare_data()
        dm.setup()

        #### Check Validation size
        len_train = len(dm.train_dataset.indices)
        len_val = len(dm.val_dataset.indices)
        assert len_val / (len_train + len_val) == self.val_split
        assert len_train == 32
        assert len_val == 8
        assert len(dm.test_dataset) == 20

        #### Train Loader Testing
        dl_train: DataLoader = dm.train_dataloader()
        self._dl_tester(dl_train, 'train', self.label_name, Path('data/train_set'))

        #### Val Loader Testing
        dl_val: DataLoader = dm.val_dataloader()
        self._dl_tester(dl_val, 'val', self.label_name, Path('data/train_set'))

        #### Test Loader Testing
        dl_test: DataLoader = dm.test_dataloader()
        self._dl_tester(dl_test, 'test', self.label_name, Path('data/val_set'))

    def test_fulldataset(self):
        """
        Testing data module with no transforms applied

        """
        data_root = Path('data')
        self.label_type = ['arousal', 'valence', 'expression']
        self.label_name = ['aro', 'val', 'exp']
        self.val_split = 0.2
        self.batch_size = 64

        dm = AffectNetImageDataModule(self.label_type, data_root=data_root,
                                      val_split=self.val_split, batch_size=self.batch_size,
                                      refresh_cache=False, num_workers=0)
        dm.prepare_data()
        dm.setup()

        #### Check Validation size
        len_train = len(dm.train_dataset.indices)
        len_val = len(dm.val_dataset.indices)
        assert len_val / (len_train + len_val) == pytest.approx(self.val_split, 2e-5)
        assert len_train == 230121
        assert len_val == 57530
        assert len(dm.test_dataset) == 3999

        #### Train Loader Testing
        dl_train: DataLoader = dm.train_dataloader()
        self._dl_tester(dl_train, 'train', self.label_name, Path('data/train_set'))

        #### Val Loader Testing
        dl_val: DataLoader = dm.val_dataloader()
        self._dl_tester(dl_val, 'val', self.label_name, Path('data/train_set'))

        #### Test Loader Testing
        dl_test: DataLoader = dm.test_dataloader()
        self._dl_tester(dl_test, 'test', self.label_name, Path('data/val_set'))

    def _dl_tester(self, dataloader: DataLoader, partition: str, label_name: Union[str, list], datapath: Path):

        # convert to lists for testing y values
        if isinstance(label_name, str):
            label_name = [label_name]

        data_iter = iter(dataloader)
        x_batch, y_batch, ids_batch = next(data_iter)

        assert x_batch.shape ==  (self.batch_size, 3, 224, 224)    #images are 3, 224, 224 (pytorch is channels first)

        # check random data
        batch_idx = random.randint(0, self.batch_size-1) if partition in ['train', 'val'] else 0
        x = x_batch[batch_idx]
        y = y_batch[batch_idx]
        img_id = ids_batch[batch_idx]

        # load image and check images are equal
        imagepath = datapath / 'images'
        with open(imagepath / f'{img_id}.jpg', "rb") as f:
            x_exp = Image.open(f)
            x_exp.convert("RGB")
        x_exp = transforms.ToTensor()(x_exp)
        torch.testing.assert_close(x, x_exp)

        # load annotations and check equal
        y_exp = [float(np.load(datapath / 'annotations' / f'{img_id}_{l}.npy').item())
                 for l in label_name]
        np.testing.assert_allclose(y, y_exp)

class TestAffectNetAUDataModule:

    def test_datamodule(self):

        self.label_type = 'arousal'
        self.label_name = 'aro'
        self.val_split = 0.2
        self.batch_size = 32

        dm = AffectNetAUDataModule(self.label_type, val_split=self.val_split, batch_size=self.batch_size, num_workers=0)
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

        assert x_batch.shape ==  (self.batch_size, 35)

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