import argparse as ap
from pathlib import Path

from src.feature_extractors.feature_extactors import ActionUnitExtractor

def setup_argparser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--datafolder', type=Path, required=True,
                        help='Specify the location of the input image data.')
    parser.add_argument('-o', '--outputfolder', type=Path, required=True,
                        help='Specify the location where the OpenFace CSV files should be stored.',)
    parser.add_argument('--aus', action='store_true',
                        help='Specify if only AUs should be extracted with OpenFace')
    return parser

def main(args):

    au_extractor = ActionUnitExtractor(extract_only_aus=args.aus)
    au_extractor(args.datafolder, args.outputfolder)

if __name__ == '__main__':
    parser = setup_argparser()
    main(parser.parse_args())