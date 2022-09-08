import os
import subprocess
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm

import pandas as pd

class ActionUnitExtractor:

    def __init__(self, extract_only_aus: bool=True, other_params: Optional[List]=None):
        '''

        :param extract_only_aus:
        :param dynamic:
        :param other_params:
        '''

        # Feature Extraction using OpenFace
        self._executable = "FaceLandmarkImg"
        self._cl_args = [self._executable]

        # add params
        self._cl_args += ['-aus'] if extract_only_aus else []
        self._cl_args += other_params if other_params is not None else []

    def __call__(self, datapath: Path, outpath: Path):
        return self.extract(datapath, outpath)

    def extract(self, datapath: Path, outpath: Path) -> pd.DataFrame:
        '''

        :param datapath:
        :param outpath:
        :return:
        '''

        if not datapath.exists():
            raise FileExistsError(f'Data path {datapath} does not exist.')

        with tqdm(os.walk(datapath)) as pbar_outer:
            for root, dirs, files in pbar_outer:
                pbar_outer.set_description(f'Extracting features in {root}')
                with tqdm(dirs, leave=False) as pbar_inner:
                    for subdir in pbar_inner:
                        pbar_inner.set_description(f'- subfolder {subdir}')

                        imgsrc = Path(root) / subdir
                        featdst = Path(str(imgsrc).replace(str(datapath), str(outpath)).replace('images', 'aus'))   # TODO can I generalize this more?

                        #check to see if there are files in the directory to process otherwise skip OpenFace
                        src_files = [Path(f) for f in os.scandir(imgsrc) if Path(f).is_file() and Path(f).suffix == '.jpg']
                        if len(src_files) > 0:
                            if not featdst.exists():
                                featdst.mkdir(parents=True)

                            cl = self._cl_args + ['-fdir', Path(root) / subdir, '-out_dir', featdst]  # setup final command line
                            proc = subprocess.run(cl, capture_output=True)

                            # TODO how to check success? OpenFace doesn't return error codes very well
                            if proc.returncode != 0:
                                raise RuntimeError("An error occured while processing video: ", proc.stdout, proc.stderr)


#### Simple Class Drivers ####
def run_image_unit_extractor():
    import shutil

    datapath = Path('unittests/testdata/affectnet/')
    outpath = Path('unittests/test/extract_unittest')

    # delete folder if it already exists for testing
    if outpath.exists():
        shutil.rmtree(outpath)

    au_extractor = ActionUnitExtractor(extract_only_aus=True,
                                       other_params=None)
    au_extractor(datapath=datapath, outpath=outpath)

if __name__ == '__main__':
    run_image_unit_extractor()