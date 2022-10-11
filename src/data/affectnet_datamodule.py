from pathlib import Path
from typing import Union

import torch.utils.data
import pytorch_lightning as pl

from src.data.datasets import AffectNetAUDataset

class AffectNetAUDataModule(pl.LightningDataModule):

    #TODO simplify this..
    # default data paths.  best to use symbolic links to regenerate this structure so code doesn't need to change
    TRAIN_AUS_PATH = Path('train_set/aus')
    TRAIN_LABELS_PATH = Path('train_set/annotations')

    VAL_AUS_PATH = Path('val_set/aus')
    VAL_LABELS_PATH = Path('val_set/annotations')

    def __init__(self,
                 label_type: Union[str, list],
                 data_basepth: Path=Path('data'),
                 val_split: float=0.1,
                 batch_size: int = 64):
        '''

        :param label_type:
        :param val_split:
        :param batch_size:
        '''
        super(AffectNetAUDataModule, self).__init__()

        self._train_aus_path = data_basepth / self.TRAIN_AUS_PATH
        self._train_labels_path = data_basepth / self.TRAIN_LABELS_PATH

        self._val_aus_path = data_basepth / self.VAL_AUS_PATH
        self._val_labels_path = data_basepth / self.VAL_LABELS_PATH

        self._label_type = label_type

        self._val_split = val_split
        self._batch_size = batch_size

        self._train_dataset: torch.utils.data.Dataset = None
        self._val_dataset: torch.utils.data.Dataset = None
        self._test_dataset: torch.utils.data.Dataset = None

        self.dims = (35, )
        self.num_classes = 8

        self._feature_names: list = None
        self._expression_labels: list = None


    def prepare_data(self):
        '''

        :return:
        '''

        # Build AU dataset (will load from cache if it exists)
        train_ds = AffectNetAUDataset(self._train_aus_path, self._train_labels_path,
                                     self._label_type, 'affectnet_train_aus', True, True)
        test_ds = AffectNetAUDataset(self._val_aus_path, self._val_labels_path,
                                     self._label_type, 'affectnet_val_aus', True, True)

        self._feature_names = train_ds.feature_names
        self._expression_labels = train_ds.expression_labels

    def setup(self, stage=None):
        '''

        :param stage:
        :return:
        '''

        if stage == 'fit' or stage is None:
            # current AffectNet has no test data, so we will use validation for testing
            self._train_dataset = AffectNetAUDataset(self._train_aus_path, self._train_labels_path,
                                                     self._label_type, 'affectnet_train_aus', False, True)
            # and split validation from train data
            splits = (int(len(self._train_dataset) * (1 - self._val_split)), int(len(self._train_dataset) * self._val_split))
            self._train_dataset, self._val_dataset = torch.utils.data.random_split(self._train_dataset, splits)

        if stage == 'test' or stage == None:
            self._test_dataset = AffectNetAUDataset(self._val_aus_path, self._val_labels_path,
                                                    self._label_type, 'affectnet_val_aus', False, True)

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def expression_labels(self):
        return self._expression_labels

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> torch.utils.data.Dataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self._test_dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=True)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=1,
                                           shuffle=False)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=1,
                                           shuffle=False)