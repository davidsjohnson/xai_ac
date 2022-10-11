from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets import AffectNetAUDataset


class TestAffectNetAUDataset:

    def test_fromscratch(self):

        trainpath = Path('unittests/testdata/affectnet/processed/train_set/aus')
        annotationspath = Path('unittests/testdata/affectnet/raw/train_set/annotations')
        label_column = 'expression'
        label_name = 'exp'      # needed to load original annotation file

        ds = AffectNetAUDataset(trainpath,
                                annotationspath,
                                label_column,
                                'affectnet_train_aus',
                                save=True,
                                load_cache=False,
                                return_numpy=True)

        exp_len = 40
        assert exp_len == len(ds)

        x, y, img_id = ds[10]
        assert len(x) == 35     # 35 AUs from OpenFace

        df_aus = pd.read_csv(trainpath / f'{img_id}.csv')
        x_exp = df_aus[ds.feature_names]
        y_exp = float(np.load(annotationspath / f'{img_id}_{label_name}.npy').item())

        assert np.allclose(x, x_exp)
        assert y == y_exp

    def test_fromcache(self):

        trainpath = Path('unittests/testdata/affectnet/processed/train_set/aus')
        annotationspath = Path('unittests/testdata/affectnet/raw/train_set/annotations')
        label_column = 'expression'
        label_name = 'exp'      # needed to load original annotation file

        ds = AffectNetAUDataset(trainpath,
                                annotationspath,
                                label_column,
                                'affectnet_train_aus',
                                save=True,
                                load_cache=True,
                                return_numpy=True)

        exp_len = 40
        assert exp_len == len(ds)

        x, y, img_id = ds[33]
        assert len(x) == 35

        # get expected values from original feature and annotation files
        df_aus = pd.read_csv(trainpath / f'{img_id}.csv')
        x_exp = df_aus[ds.feature_names]
        y_exp = float(np.load(annotationspath / f'{img_id}_{label_name}.npy').item())

        assert np.allclose(x, x_exp)
        assert y == y_exp

    def test_alllabeltypes(self):

        trainpath = Path('unittests/testdata/affectnet/processed/train_set/aus')
        annotationspath = Path('unittests/testdata/affectnet/raw/train_set/annotations')
        label_columns = ['arousal', 'valence', 'expression']
        label_names = ['aro', 'val', 'exp']

        ds = AffectNetAUDataset(trainpath,
                                annotationspath,
                                label_columns,
                                'affectnet_train_aus',
                                save=True,
                                load_cache=True,
                                return_numpy=False)

        exp_len = 40
        assert exp_len == len(ds)

        x, y, img_id = ds[21]
        assert len(x) == 35

        # get expected values from original feature and annotation files
        df_aus = pd.read_csv(trainpath / f'{img_id}.csv')
        x_exp = df_aus[ds.feature_names]
        y_exp = {c: float(np.load(annotationspath / f'{img_id}_{l}.npy').item())
                 for l, c in zip(label_names, label_columns)}
        y_exp = pd.Series(data=y_exp)

        assert np.allclose(x, x_exp)
        pd.testing.assert_series_equal(y, y_exp, check_names=False)


    def test_partiallabeltypes(self):

        trainpath = Path('unittests/testdata/affectnet/processed/train_set/aus')
        annotationspath = Path('unittests/testdata/affectnet/raw/train_set/annotations')
        label_columns = ['arousal', 'valence']
        label_names = ['aro', 'val']

        ds = AffectNetAUDataset(trainpath,
                                annotationspath,
                                label_columns,
                                'affectnet_train_aus',
                                save=True,
                                load_cache=True,
                                return_numpy=False)

        exp_len = 40
        assert exp_len == len(ds)

        x, y, img_id = ds[4]
        assert len(x) == 35

        # get expected values from original feature and annotation files
        df_aus = pd.read_csv(trainpath / f'{img_id}.csv')
        x_exp = df_aus[ds.feature_names]
        y_exp = {c: float(np.load(annotationspath / f'{img_id}_{l}.npy').item())
                 for l, c in zip(label_names, label_columns)}
        y_exp = pd.Series(data=y_exp)

        assert np.allclose(x, x_exp)
        pd.testing.assert_series_equal(y, y_exp, check_names=False)
