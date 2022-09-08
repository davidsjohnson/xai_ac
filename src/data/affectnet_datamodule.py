from pathlib import Path

import torch.utils.data
import pytorch_lightning as pl

from src.data.datasets import AffectNetAUDataset

class AffectNetAUDataModule(pl.LightningDataModule):

    #TODO simplify this..
    # default data paths.  best to use symbolic links to regenerate this structure so code doesn't need to change
    TRAIN_AUS_PATH = Path('data/train_set/aus')
    TRAIN_LABELS_PATH = Path('data/train_set/annotations')

    VAL_AUS_PATH = Path('data/val_set/aus')
    VAL_LABELS_PATH = Path('data/val_set/annotations')

    def __init__(self,
                 label_type: str,
                 val_split: float=0.1,
                 batch_size: int = 64):
        '''

        :param label_type:
        :param val_split:
        :param batch_size:
        '''
        super(AffectNetAUDataModule, self).__init__()

        self._train_aus_path = self.TRAIN_AUS_PATH
        self._train_labels_path = self.TRAIN_LABELS_PATH

        self._val_aus_path = self.VAL_AUS_PATH
        self._val_labels_path = self.VAL_LABELS_PATH

        self._label_type = label_type

        self._val_split = val_split
        self._batch_size = batch_size

        self._ds_train: torch.utils.data.Dataset = None
        self._ds_val: torch.utils.data.Dataset = None
        self._ds_test: torch.utils.data.Dataset = None

        self.dims = (35, )
        self.num_classes = 8


    def prepare_data(self):
        '''

        :return:
        '''

        # Build AU dataset (will load from cache if it exists)
        AffectNetAUDataset(self._train_aus_path, self._train_labels_path,
                           self._label_type, 'affectnet_train_aus', True, True)
        AffectNetAUDataset(self._val_aus_path, self._val_labels_path,
                           self._label_type, 'affectnet_val_aus', True, True)

    def setup(self, stage=None):
        '''

        :param stage:
        :return:
        '''

        if stage == 'fit' or stage is None:
            # current AffectNet has no test data, so we will use validation for testing
            self._ds_train = AffectNetAUDataset(self._train_aus_path, self._train_labels_path,
                                                self._label_type, 'affectnet_train_aus', False, True)
            # and split validation from train data
            splits = (int(len(self._ds_train) * (1 - self._val_split)), int(len(self._ds_train) * self._val_split))
            self._ds_train, self._ds_val = torch.utils.data.random_split(self._ds_train, splits)

        if stage == 'test' or stage == None:
            self._ds_test = AffectNetAUDataset(self._val_aus_path, self._val_labels_path,
                                               self._label_type, 'affectnet_val_aus', False, True)

    def train_dataloader(self):
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self._ds_train,
                                           batch_size=self._batch_size,
                                           shuffle=True)

    def val_dataloader(self):
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self._ds_val,
                                           batch_size=self._batch_size,
                                           shuffle=True)

    def test_dataloader(self):
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self._ds_test,
                                           batch_size=1,
                                           shuffle=False)

    def predict_dataloader(self):
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self._ds_test,
                                           batch_size=1,
                                           shuffle=False)