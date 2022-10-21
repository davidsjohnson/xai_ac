from pathlib import Path

import numpy as np
import numpy.testing
import pandas as pd

from PIL import Image

import torch
torch.manual_seed(17)

from torchvision import transforms

from src.data.affectnet_datasets import AffectNetAUDataset, AffectNetImageDataset


class TestAffectNetImageDataset:

    def test_basicloading(self):
        trainpath = Path('unittests/testdata/affectnet/raw/train_set')
        label_col = 'expression'
        ds = AffectNetImageDataset(trainpath, label_col, name='affectnet_train_debug',
                                   load_cache=False, save=False)

        assert ds.df is not None

        exp_len = 40
        assert exp_len == len(ds)

        x, y, img_id = ds[10]
        assert x.size == (224, 224)
        assert x.mode == 'RGB'
        assert y.shape == (1, )
        assert isinstance(img_id, str)


    def test_correct(self):

        trainpath = Path('unittests/testdata/affectnet/raw/train_set')
        imagepath = trainpath / 'images'
        annotationspath = trainpath / 'annotations'

        label_columns = ['arousal', 'valence', 'expression']
        label_names = ['aro', 'val', 'exp']

        ds = AffectNetImageDataset(trainpath, label_columns, name='affectnet_train_debug',
                                   load_cache=False, save=False, keep_as_pandas=True)

        x_act, y_act, img_id = ds[30]

        # load image and check images are equal
        with open(imagepath / f'{img_id}.jpg', "rb") as f:
            x_exp = Image.open(f)
            x_exp.convert("RGB")
        assert list(x_act.getdata()) == list(x_exp.getdata())

        # load annotations and check equal
        y_exp = {c: float(np.load(annotationspath / f'{img_id}_{l}.npy').item())
                 for l, c in zip(label_names, label_columns)}
        y_exp = pd.Series(data=y_exp)
        pd.testing.assert_series_equal(y_act, y_exp, check_names=False)


    def test_transform(self):
        trainpath = Path('unittests/testdata/affectnet/raw/train_set')
        imagepath = trainpath / 'images'
        annotationspath = trainpath / 'annotations'

        label_columns = ['arousal', 'valence', 'expression']
        label_names = ['aro', 'val', 'exp']

        img_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandAugment()]
        )

        target_transform = transforms.Compose([lambda x: x/.05])

        ds = AffectNetImageDataset(trainpath, label_columns, name='affectnet_train_debug',
                                   transform=img_transforms, target_transform=target_transform,
                                    load_cache=False, save=False, keep_as_pandas=False)

        x_act, y_act, img_id = ds[28]

        # load image and check images are equal
        with open(imagepath / f'{img_id}.jpg', "rb") as f:
            x_exp = Image.open(f)
            x_exp.convert("RGB")
        assert list(x_act.getdata()) != list(x_exp.getdata())

        # save for manual review
        x_act.save('unittests/testoutput/x_transformed.jpg')
        x_exp.save('unittests/testoutput/x_original.jpg')

        # load annotations and check equal
        y_exp = [float(np.load(annotationspath / f'{img_id}_{l}.npy').item())  / 0.05
                 for l in label_names]
        np.testing.assert_allclose(y_act, y_exp)


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
        x_exp = df_aus[ds.feature_names].values
        y_exp = float(np.load(annotationspath / f'{img_id}_{label_name}.npy').item())

        np.testing.assert_allclose(x, np.squeeze(x_exp))
        np.testing.assert_allclose(y, y_exp)

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
        x_exp = df_aus[ds.feature_names].values
        y_exp = float(np.load(annotationspath / f'{img_id}_{label_name}.npy').item())

        np.testing.assert_allclose(x, np.squeeze(x_exp))
        np.testing.assert_allclose(y, y_exp)

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

        np.testing.assert_allclose(x, np.squeeze(x_exp))
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
        x_exp = df_aus[ds.feature_names].values
        y_exp = {c: float(np.load(annotationspath / f'{img_id}_{l}.npy').item())
                 for l, c in zip(label_names, label_columns)}
        y_exp = pd.Series(data=y_exp)

        np.testing.assert_allclose(x, np.squeeze(x_exp))
        pd.testing.assert_series_equal(y, y_exp, check_names=False)
