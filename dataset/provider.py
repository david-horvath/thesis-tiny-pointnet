import os
import glob
import zipfile
import h5py
import numpy as np

from sklearn.model_selection import train_test_split


VKITTI3D_DATASET_KEY = 'vkitti3d'
S3DIS_DATASET_KEY = 's3dis'

VKITTI3D_CLASSES = [
    ('Terrain', [200, 90, 0]),
    ('Tree', [0, 128, 50]),
    ('Vegetation', [0, 220, 0]),
    ('Building', [255, 0, 0]),
    ('Road', [100, 100, 100]),
    ('Guard rail', [200, 200, 200]),
    ('Traffic sign', [255, 0, 255]),
    ('Traffic light', [255, 255, 0]),
    ('Pole', [128, 0, 255]),
    ('Misc', [255, 200, 150]),
    ('Truck', [0, 128, 255]),
    ('Car', [0, 200, 255]),
    ('Van', [255, 128, 0]),
    ('Don\'t care', [0, 0, 0])
]

S3DIS_CLASSES = [
    ('Ceiling', [233, 229, 107]),
    ('Floor', [95, 156, 196]),
    ('Wall', [179, 116, 81]),
    ('Beam', [241, 149, 131]),
    ('Column', [81, 163, 148]),
    ('Window', [77, 174, 84]),
    ('Door', [108, 135, 75]),
    ('Chair', [41, 49, 101]),
    ('Table', [79, 79, 76]),
    ('Bookcase', [223, 52, 52]),
    ('Sofa', [89, 47, 95]),
    ('Board', [81, 109, 114]),
    ('Clutter', [233, 233, 229]),
    ('Don\'t care', [0, 0, 0])
]


_BASE = os.path.dirname(os.path.realpath(__file__))
_PARENT = os.path.dirname(_BASE)
_DATAS = _PARENT + '/datas'


def dataset(dataset_name='vkitti3d'):
    print(f'Loading {dataset_name} dataset...')

    path = _DATAS + f'/{dataset_name}_dataset'

    if not os.path.exists(path):
        os.mkdir(path)

    zip_ref = zipfile.ZipFile(_DATAS + f'/{dataset_name}_h5.zip', 'r')
    zip_ref.extractall(path)
    zip_ref.close()

    train_data = []
    train_label = []

    files = glob.glob(path + '/*')

    for file in files:
        f = h5py.File(file, 'r')
        train_data.append(f['normalized_points'][:, :, 0:3])
        train_label.append(f['labels'])

    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)

    print(f'Load {dataset_name} dataset success')

    train_data, val_data, train_label, val_label = train_test_split(train_data,
                                                                    train_label,
                                                                    test_size=0.4,
                                                                    random_state=42)

    val_data, test_data, val_label, test_label = train_test_split(val_data,
                                                                  val_label,
                                                                  test_size=0.5,
                                                                  random_state=42)

    print(f'train data: {train_data.shape}, train label: {train_label.shape}')
    print(f'validation data: {val_data.shape}, validation label: {val_label.shape}')
    print(f'test data: {test_data.shape}, test label: {test_label.shape}')

    return train_data, train_label, val_data, val_label, test_data, test_label
