from pathlib import Path

from src.data.affectnet_datasets import AffectNetImageDataset


def calc_classweights(ds):
    counts = ds.df['expression'].value_counts(normalize=True)
    return counts

def main():
    dataroot = Path('data/train_set')
    ds = AffectNetImageDataset(dataroot, 'expression', name='train_data',
                               load_cache=True, save=False, keep_as_pandas=True)
    counts = calc_classweights(ds)
    print(counts)

if __name__ == '__main__':
    main()
    