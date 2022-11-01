from pathlib import Path
from typing import Union, Callable, Optional

import torch.utils.data
from torchvision import transforms
import pytorch_lightning as pl

from src.data.affectnet_datasets import AffectNetAUDataset, AffectNetImageDataset

class TrainValSplitTransformWrapper(torch.utils.data.Dataset):
    """
    Class to ensure validation data from Torch train/split doesn't have training transform applied
    https://discuss.pytorch.org/t/disable-transform-in-validation-dataset/55572
    """
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super(TrainValSplitTransformWrapper, self).__init__()

        self.dataset = dataset
        # setting all dataset transforms to None so wrapper handles transform instead
        self.dataset.transforms = None
        self.dataset.transform = None
        self.dataset.target_transform = None

        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        new_idx = self.indices[idx]
        sample, target, img_id = self.dataset[new_idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, img_id

class AbstractAffectNetDataModule(pl.LightningDataModule):

    def __init__(self, num_workers):
        super(AbstractAffectNetDataModule, self).__init__()
        # TODO combine inits to here...

        self._num_workers = num_workers

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
                                           shuffle=True,
                                           num_workers=self._num_workers)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=True,
                                           num_workers=self._num_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        '''

        :return:
        '''
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers)


class AffectNetImageDataModule(AbstractAffectNetDataModule):

    def __init__(self,
                 label_type: Union[str, list],
                 data_root: Path = Path('data'),
                 name: str = 'affectnet_img',
                 val_split: float = 0.1,
                 batch_size: int = 64,
                 train_transform: Callable = transforms.ToTensor(),
                 train_target_transform: Optional[Callable] = None,
                 test_transform: Callable = transforms.ToTensor(),
                 test_targret_transform: Optional[Callable] = None,
                 keep_as_pandas: bool = False,
                 refresh_cache: bool = False,
                 num_workers:int = 0):
        '''

        :param label_type:
        :param data_root:
        :param name:
        :param val_split:
        :param batch_size:
        :param refresh_cache:
        '''
        super(AffectNetImageDataModule, self).__init__(num_workers)
        self._train_root = data_root / 'train_set'
        self._val_root = data_root / 'val_set'

        self._name = name
        self._keep_as_pandas = keep_as_pandas
        self._refresh_cache = refresh_cache

        self._label_type = label_type

        self._val_split = val_split
        self._batch_size = batch_size

        self._expression_labels = None
        self._feature_names = None

        self._train_dataset: AffectNetImageDataset = None
        self._val_dataset: AffectNetImageDataset = None
        self._test_dataset: AffectNetImageDataset = None

        self._train_transform = train_transform
        self._train_target_transform = train_target_transform

        self._test_transform = test_transform
        self._test_target_transform = test_targret_transform

    @property
    def dims(self):
        return 3, 224, 224

    @property
    def num_classes(self):
        return 8

    def prepare_data(self):
        '''

        :return:
        '''

        # Build a dummy dataset to get features and labels. Also builds cache if neeeed
        # TODO: Update AUs datamodule too
        train_ds = AffectNetImageDataset(self._train_root, self._label_type, name=f'{self._name}_train',
                                         load_cache=True if not self._refresh_cache else False, save=True,
                                         keep_as_pandas=self._keep_as_pandas)
        _ = AffectNetImageDataset(self._val_root, self._label_type, name=f'{self._name}_val',
                                  load_cache=True if not self._refresh_cache else False, save=True,
                                  keep_as_pandas=self._keep_as_pandas)

        self._expression_labels = train_ds.expression_labels

    def setup(self, stage=None):
        '''

        :param stage:
        :return:
        '''

        if stage == 'fit' or stage is None:
            # current AffectNet has no test data, so we will use validation for testing
            self._train_dataset = AffectNetImageDataset(self._train_root, self._label_type,
                                                        name=f'{self._name}_train',
                                                        transform=self._train_transform,
                                                        target_transform=self._train_target_transform,
                                                        load_cache=True, save=False,
                                                        keep_as_pandas=self._keep_as_pandas)

            # and split validation from train data
            val_split_size = int(len(self._train_dataset) * self._val_split)
            train_split_size = len(self._train_dataset) - val_split_size
            self._train_dataset, self._val_dataset = torch.utils.data.random_split(self._train_dataset, (train_split_size, val_split_size))

            # Wrapping Torch subset dataset to ensure train transforms are not applied to validation data
            self._train_dataset = TrainValSplitTransformWrapper(self._train_dataset.dataset,
                                                                self._train_dataset.indices,
                                                                transform=self._train_transform,
                                                                target_transform=self._train_target_transform)
            self._val_dataset = TrainValSplitTransformWrapper(self._val_dataset.dataset,
                                                              self._val_dataset.indices,
                                                              transform=self._test_transform,
                                                              target_transform=self._test_target_transform)

        if stage == 'test' or stage == None:
            self._test_dataset = AffectNetImageDataset(self._val_root, self._label_type,
                                                       name=f'{self._name}_val',
                                                       transform=self._test_transform,
                                                       target_transform=self._test_target_transform,
                                                       load_cache=True, save=False,
                                                       keep_as_pandas=self._keep_as_pandas)


class AffectNetAUDataModule(AbstractAffectNetDataModule):

    #TODO simplify this..
    # default data paths.  best to use symbolic links to regenerate this structure so code doesn't need to change
    TRAIN_AUS_PATH = Path('train_set/aus')
    TRAIN_LABELS_PATH = Path('train_set/annotations')

    VAL_AUS_PATH = Path('val_set/aus')
    VAL_LABELS_PATH = Path('val_set/annotations')

    def __init__(self,
                 label_type: Union[str, list],
                 data_root: Path=Path('data'),
                 val_split: float=0.1,
                 batch_size: int = 64,
                 num_workers:int = 0):
        '''

        :param label_type:
        :param data_root:
        :param val_split:
        :param batch_size:
        '''
        super(AffectNetAUDataModule, self).__init__(num_workers)

        self._train_aus_path = data_root / self.TRAIN_AUS_PATH
        self._train_labels_path = data_root / self.TRAIN_LABELS_PATH

        self._val_aus_path = data_root / self.VAL_AUS_PATH
        self._val_labels_path = data_root / self.VAL_LABELS_PATH

        self._label_type = label_type

        self._val_split = val_split
        self._batch_size = batch_size

        self._train_dataset: torch.utils.data.Dataset = None
        self._val_dataset: torch.utils.data.Dataset = None
        self._test_dataset: torch.utils.data.Dataset = None

        self._feature_names: list = None
        self._expression_labels: list = None

    @property
    def dims(self):
        return 35,

    @property
    def num_classes(self):
        return 8

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




## Data  Module Loading to Build Cache
def load_img_dm():

    label_type = ['arousal', 'valence', 'expression']

    dm = AffectNetImageDataModule(label_type, refresh_cache=False)
    dm.prepare_data()
    dm.setup()

    return dm

if __name__ == '__main__':

    pl.seed_everything(42)

    dm_img = load_img_dm()
    print('Data Module Loaded')
    print(f'Train Data: {len(dm_img.train_dataset)}')
    print(f'Val Data: {len(dm_img.val_dataset)}')
    print(f'Test Data: {len(dm_img.test_dataset)}')