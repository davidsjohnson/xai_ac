import shutil
from pathlib import Path

DATAPATH = Path('/data/affectnet_2017_ext_img_mix_wild/raw/')
TESTPATH = Path('/home/djohnson/projects/xai_fe/unittests/testdata/affectnet')


def main():

    # setup train data for unittesting
    train_img_path = DATAPATH / 'train_set' / 'images'
    train_labels_path = DATAPATH / 'train_set' / 'annotations'

    train_img_dst = TESTPATH / 'train_set' / 'images'
    train_img_dst.mkdir(exist_ok=True, parents=True)
    train_labels_dst = TESTPATH / 'train_set' / 'annotations'
    train_labels_dst.mkdir(exist_ok=True, parents=True)


    imgpaths = train_img_path.rglob('*.jpg')
    imgiter = iter(imgpaths)

    for i in range(40):
        imgpath = next(imgiter)
        shutil.copy(imgpath, train_img_dst)

        img_id = imgpath.stem
        f_aro = f'{img_id}_aro.npy'
        f_val = f'{img_id}_val.npy'
        f_exp = f'{img_id}_exp.npy'
        f_lnd = f'{img_id}_lnd.npy'

        shutil.copy(train_labels_path / f_aro, train_labels_dst)
        shutil.copy(train_labels_path / f_val, train_labels_dst)
        shutil.copy(train_labels_path / f_exp, train_labels_dst)
        shutil.copy(train_labels_path / f_lnd, train_labels_dst)


    val_img_path = DATAPATH / 'val_set' / 'images'
    val_labels_path = DATAPATH / 'val_set' / 'annotations'

    val_img_dst = TESTPATH / 'val_set' / 'images'
    val_img_dst.mkdir(exist_ok=True, parents=True)
    val_labels_dst = TESTPATH / 'val_set' / 'annotations'
    val_labels_dst.mkdir(exist_ok=True, parents=True)

    imgpaths = val_img_path.rglob('*.jpg')
    imgiter = iter(imgpaths)

    for i in range(20):
        imgpath = next(imgiter)
        shutil.copy(imgpath, val_img_dst)

        img_id = imgpath.stem
        f_aro = f'{img_id}_aro.npy'
        f_val = f'{img_id}_val.npy'
        f_exp = f'{img_id}_exp.npy'
        f_lnd = f'{img_id}_lnd.npy'

        shutil.copy(val_labels_path / f_aro, val_labels_dst)
        shutil.copy(val_labels_path / f_val, val_labels_dst)
        shutil.copy(val_labels_path / f_exp, val_labels_dst)
        shutil.copy(val_labels_path / f_lnd, val_labels_dst)


if __name__ == '__main__':
    main()