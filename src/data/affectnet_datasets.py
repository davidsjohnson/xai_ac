from pathlib import Path
from typing import Union, Optional, Callable, Any, Tuple

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch.utils.data
import torchvision
from torchvision.datasets.folder import default_loader

class AffectNetImageDataset(torchvision.datasets.VisionDataset):

    LABEL_TYPES = ['arousal', 'valence', 'expression']
    EXPRESSION_LABELS = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    def __init__(self,
                 root: Union[str, Path],
                 label_column: Union[str, list],
                 name: str,
                 loader: Callable = default_loader,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 save: bool = False,
                 load_cache: bool = True,
                 keep_as_pandas = False
                 ):
        """
        uses pytorch DatasetFolder as a guide but we must be able to support regression labels in addition to classes
        https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#DatasetFolder

        :param root: 
        :param transform:
        :param target_transform:
        :param transforms
        """
        super(AffectNetImageDataset, self).__init__(root,
                                                    transforms=transforms,
                                                    transform=transform,
                                                    target_transform=target_transform)
        self._imgs_root = Path(self.root) / 'images'
        self._annotations_root = Path(self.root) / 'annotations'
        self._name = name
        self._keep_as_pandas = keep_as_pandas

        self._csv_path = self.root / f'{name}.csv'
        if load_cache == True and self._csv_path.exists():
            self._df: pd.DataFrame = pd.read_csv(self._csv_path)
        else:
            self._df: pd.DataFrame = self._make_dataset(savepath=self._csv_path if save else None)

        self._loader = loader

        # handle a single label column or a list of label columns
        if isinstance(label_column, str):
            label_column = [label_column]       # convert single label type to a list so get_item returns series
        for l in label_column:
            if l not in self.LABEL_TYPES:
                raise ValueError(f'Invalid Label Type {l}.  Should be one of {self.LABEL_TYPES}')
        self._label_col = label_column

    @property
    def expression_labels(self):
        return self.EXPRESSION_LABELS

    @property
    def df(self):
        return self._df

    @property
    def dims(self):
        return 3, 224, 224

    @property
    def num_classes(self):
        return 8

    def _make_dataset(self, savepath: Optional[Path]=None):

        img_files = self._imgs_root.rglob('*.jpg')

        dfs = []
        with tqdm(img_files) as pbar:
            for f in pbar:

                img_id = f.stem

                df = pd.DataFrame(data=[f], columns=['filepath'])
                df['img_id'] = img_id

                # load annotations
                f_aro = self._annotations_root / f'{img_id}_aro.npy'
                f_val = self._annotations_root / f'{img_id}_val.npy'
                f_exp = self._annotations_root / f'{img_id}_exp.npy'
                f_lnd = self._annotations_root / f'{img_id}_lnd.npy'

                df['arousal'] = float(np.load(f_aro).item())
                df['valence'] = float(np.load(f_val).item())
                df['expression'] = int(np.load(f_exp).item())

                # load landmarks
                lnds = np.load(f_lnd)
                lnd_cols = [f'lnd_{i // 2 + 1}_{"x" if i % 2 == 0 else "y"}' for i in range(0, len(lnds))]
                df = pd.concat([df, pd.DataFrame([lnds], columns=lnd_cols, index=df.index)], axis=1)

                dfs.append(df)

        if len(dfs) == 0:
            raise ValueError(f'No JPG files found in folder: {self._imgs_root.resolve()}')
        df_imgs = pd.concat(dfs)
        df_imgs.astype({'expression': int})

        if savepath is not None:
            df_imgs.to_csv(savepath)

        return df_imgs

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        imgpath = self._df['filepath'].iloc[idx]
        target = self._df[self._label_col].iloc[idx]
        sample = self._loader(imgpath)

        # convert target to numpy instead of pandas series
        if not self._keep_as_pandas:
            target = target.values

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img_id = self._df['img_id'].iloc[idx]

        return sample, target, img_id


class AffectNetAUDataset(torch.utils.data.Dataset):

    LABEL_TYPES = ['arousal', 'valence', 'expression']
    EXPRESSION_LABELS = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    def __init__(self,
                 datapath: Path,
                 annotationspath: Path,
                 label_column: Union[str, list],
                 name: str,
                 save: bool = False,
                 load_cache: bool = True,
                 return_numpy: bool = True):
        """

        :param datapath:
        :param annotationspath:
        :param label_column:
        :param name:
        :param save:
        :param load_cache:
        :param return_numpy:
        """
        super(AffectNetAUDataset, self).__init__()
        
        self._datapath = datapath
        self._annotationspath = annotationspath
        self._name = name

        self._csv_path = self._datapath / f'{name}.csv'
        if load_cache == True and self._csv_path.exists():
            self._df: pd.DataFrame = pd.read_csv(self._csv_path)
        else:
            self._df: pd.DataFrame = self._load_aus_annotations(datapath, self._csv_path if save else None)

        # handle a single label column or a list of label columns
        if isinstance(label_column, str):
            label_column = [label_column]       # convert single label type to a list so get_item returns series
        for l in label_column:
            if l not in self.LABEL_TYPES:
                raise ValueError(f'Invalid Label Type {l}.  Should be one of {self.LABEL_TYPES}')
        self._label_col = label_column

        self._feature_cols = [col for col in self._df.columns if 'AU' in col]

        self._return_numpy = return_numpy

    @property
    def feature_names(self):
        return self._feature_cols

    @property
    def expression_labels(self):
        return self.EXPRESSION_LABELS

    @property
    def dims(self):
        return 35,

    @property
    def num_classes(self):
        return 8

    @property
    def df(self):
        return self._df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        x = self._df[self._feature_cols].iloc[idx]
        y = self._df[self._label_col].iloc[idx]
        img_id = self._df['img_id'].iloc[idx]

        if self._return_numpy:
            return x.values, y.values, img_id
        else:
            return x, y, img_id


    def _load_aus_annotations(self, path: Path, savepath: Optional[Path]=None):
        '''

        :param path:
        :param savepath:
        :return:
        '''

        dfs = []
        for f in path.rglob('*.csv'):
            if f != self._csv_path: # make sure it isn't the cache file
                df = self._load_single_item(f)
                dfs.append(df)
        df_aus = pd.concat(dfs)     # TODO Fix error.  Add check for files
        df_aus.astype({'expression': int})
        if savepath is not None:
            df_aus.to_csv(savepath)

        return df_aus

    def _load_single_item(self, filepath):
        '''

        :param filepath:
        :return:
        '''
        img_id = filepath.stem

        # load pre-extracted AUs
        df = pd.read_csv(filepath)
        try:
            df.insert(0, 'img_id', img_id)
        except ValueError as e:
            print(img_id)

        # load annotations
        f_aro = self._annotationspath / f'{img_id}_aro.npy'
        f_val = self._annotationspath / f'{img_id}_val.npy'
        f_exp = self._annotationspath / f'{img_id}_exp.npy'
        f_lnd = self._annotationspath / f'{img_id}_lnd.npy'

        df['arousal'] = float(np.load(f_aro).item())
        df['valence'] = float(np.load(f_val).item())
        df['expression'] = int(np.load(f_exp).item())

        # load landmarks
        lnds = np.load(f_lnd)
        lnd_cols = [f'lnd_{i//2 + 1}_{"x" if i % 2 == 0 else "y"}' for i in range(0, len(lnds))]
        df = pd.concat([df, pd.DataFrame([lnds], columns=lnd_cols, index=df.index)], axis=1)

        return df
