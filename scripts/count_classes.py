from pathlib import Path

from src.data.affectnet_datasets import AffectNetImageDataset


def calc_classweights(ds):
    counts = ds.df['expression'].value_counts(normalize=False, sort=False)
    return counts

def main():
    dataroot = Path('data/train_set')
    ds = AffectNetImageDataset(dataroot, 'expression', name='affectnet_img_train',
                               load_cache=True, save=False, keep_as_pandas=True)
    counts = calc_classweights(ds)
    pct_list = []
    for idx, val in counts.sort_index().iteritems():
        pct_list.append(val)
        print(f'{ds.expression_labels[int(idx)]}: {val}')

    print(pct_list)



if __name__ == '__main__':
    main()
