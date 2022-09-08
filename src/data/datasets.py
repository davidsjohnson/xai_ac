from pathlib import Path

import numpy as np
import pandas as pd

import torch.utils.data

class AffectNetAUDataset(torch.utils.data.Dataset):

    def __init__(self,
                 datapath: Path,
                 annotationspath: Path,
                 label_column: str,
                 name: str,
                 save: bool=False,
                 load_cache: bool=True):
        '''

        :param datapath:
        :param annotationspath:
        :param label_column:
        :param name:
        :param save:
        :param load_cache:
        '''

        self._datapath = datapath
        self._annotationspath = annotationspath
        self._name = name
        self._csv_path = self._datapath / f'{name}.csv'

        if load_cache == True and self._csv_path.exists():
            self._df: pd.DataFrame = pd.read_csv(self._datapath / f'{name}.csv')
        else:
            self._df: pd.DataFrame = self._load_aus_annotations(datapath, self._csv_path if save else None)

        self._label_col = label_column

        self._feature_cols = [col for col in self._df.columns if 'AU' in col]

    @property
    def feature_columns(self):
        return self._feature_cols

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        x = self._df[self._feature_cols].iloc[idx].values
        y = self._df[self._label_col].iloc[idx]
        img_id = self._df['img_id'].iloc[idx]

        return x, y, img_id


    def _load_aus_annotations(self, path: Path, savepath: Path=None):
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
        df_aus = pd.concat(dfs)
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
        df['expression'] = float(np.load(f_exp).item())

        # load landmarks
        lnds = np.load(f_lnd)
        lnd_cols = [f'lnd_{i//2 + 1}_{"x" if i % 2 == 0 else "y"}' for i in range(0, len(lnds))]
        df = pd.concat([df, pd.DataFrame([lnds], columns=lnd_cols, index=df.index)], axis=1)

        return df
