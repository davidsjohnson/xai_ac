from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets import AffectNetAUDataset


class TestAffectNetAUDataset:

    def test_fromscratch(self):

        trainpath = Path('unittests/testdata/affectnet/processed/train_set/aus')
        annotationspath = Path('unittests/testdata/affectnet/raw/train_set/annotations')
        label_column = 'expression'
        label_name = 'exp'

        ds = AffectNetAUDataset(trainpath,
                                annotationspath,
                                label_column,
                                'affectnet_train_aus',
                                save=True,
                                load_cache=False)

        exp_len = 40
        assert exp_len == len(ds)

        x, y, img_id = ds[10]
        assert len(x) == 35     # 35 AUs from OpenFace

        df_aus = pd.read_csv(trainpath / f'{img_id}.csv')
        x_exp = df_aus[ds.feature_columns]
        y_exp = float(np.load(annotationspath / f'{img_id}_{label_name}.npy').item())

        assert np.allclose(x, x_exp)
        assert y == y_exp

    def test_fromcache(self):

        trainpath = Path('unittests/testdata/affectnet/processed/train_set/aus')
        annotationspath = Path('unittests/testdata/affectnet/raw/train_set/annotations')
        label_column = 'expression'
        label_name = 'exp'

        ds = AffectNetAUDataset(trainpath,
                                annotationspath,
                                label_column,
                                'affectnet_train_aus',
                                save=True,
                                load_cache=True)

        # TODO add tests that check values are correct
        exp_len = 40
        assert exp_len == len(ds)

        x, y, img_id = ds[14]
        assert len(x) == 35

        df_aus = pd.read_csv(trainpath / f'{img_id}.csv')
        x_exp = df_aus[ds.feature_columns]
        y_exp = float(np.load(annotationspath / f'{img_id}_{label_name}.npy').item())

        assert np.allclose(x, x_exp)
        assert y == y_exp