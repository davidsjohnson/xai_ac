import os
from pathlib import Path
import shutil
from src.feature_extractors.feature_extactors import ActionUnitExtractor

class TestAUExtraction:

    def test_extract_affectnet(self):
        datapath = Path('unittests/testdata/affectnet/raw')
        outpath = Path('unittests/testdata/affectnet/processed/')

        # delete folder if it already exists for testing
        if outpath.exists():
            shutil.rmtree(outpath)

        au_extractor = ActionUnitExtractor(extract_only_aus=True,
                                           other_params=None)
        au_extractor(datapath=datapath, outpath=outpath)

        # test values
        img_files = list(datapath.rglob('*.jpg'))
        csv_files = list((outpath).rglob('*.csv'))
        n_files_exp = len(img_files) - 2 # 2 files don't extract with openface for some reason (even when manually running) [val_set/1606, ]
        n_files_act = len(csv_files)

        assert  n_files_act == n_files_exp

        # check files exist in same folder structure
        for root, dirs, files in os.walk(datapath):
            out_root = Path(root.replace(str(datapath), str(outpath)).replace('images', 'aus'))
            for f in files:
                if Path(f).suffix == '.jpg' and Path(f).stem not in ['1606', '2501']:
                    assert (out_root / f.replace('.jpg', '.csv')).exists(), f'Expected file does not exist: {out_root / f.replace(".jpg", ".csv")}'

    def test_extract(self):

        datapath = Path('unittests/testdata/affectnet_val/')
        outpath = Path('unittests/testoutput/affectnet_val/')

        # delete folder if it already exists for testing
        if outpath.exists():
            shutil.rmtree(outpath)

        au_extractor = ActionUnitExtractor(extract_only_aus=True,
                                           other_params=None)
        au_extractor(datapath=datapath,outpath=outpath)

        # test values
        img_files = list(datapath.rglob('*.jpg'))
        csv_files = list(outpath.rglob('*.csv'))
        n_files_exp = len(img_files)
        n_files_act = len(csv_files)

        assert n_files_act == n_files_exp

        # check files and folders
        for root, dirs, files in os.walk(datapath):
            out_root = Path(root.replace('testdata', 'testoutput'))
            for f in files:
                assert (out_root / f.replace('.jpg', '.csv')).exists(), f'Expected file does not exist: {out_root / f.replace(".jpg", ".csv")}'