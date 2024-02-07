import argparse
import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

# Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_root', type=str, default='MICCAI_BraTS2020_TrainingData',
                    help="Root folder of the training data.")
parser.add_argument('--output_dir', type=str, default='data',
                    help="Root folder of the output data.")
args = parser.parse_args()

if not os.path.exists(args.train_data_root):
    raise ValueError(f'Invalid training data path: {args.train_data_root}')

# Creating Output Directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Constants
OUTPUT_DIR = args.output_dir
TRAIN_PATH = args.train_data_root
DATA_TYPES = ['flair', 't1', 't1ce', 't2', 'seg']
N_FOLDS = 5

# Data Distribution
train_data_paths = {
    data_type: sorted(
        glob(f'{TRAIN_PATH}/**/*_{data_type}.nii')
    ) for data_type in DATA_TYPES
}
train_data_paths['seg'].append(f'{TRAIN_PATH}/BraTS20_Training_355/W39_1998.09.19_Segm.nii')
train_data_paths['seg'] = sorted(train_data_paths['seg'])

for k, v in train_data_paths.items():
    print(f'[TRAIN] Number of {k} images: {len(v)}')


# Data Loading and Preprocessing
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return (volume * 255).astype(np.uint8)


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Rotate by -90 degrees
    volume = ndimage.rotate(volume, -90, axes=(0, 1), reshape=False, order=1)

    return volume


# Extracting Half Volumes

# Creating CSV file
df = pd.DataFrame(columns=['volume_id', 'path', 'label', 'half'])

# iterate over train_data_paths in parallel
for i, (paths) in enumerate(zip(*train_data_paths.values())):
    # load mask and split in half to get the labels for each half
    mask = read_nifti_file(paths[-1])
    mask = ndimage.rotate(mask, -90, axes=(0, 1), reshape=False, order=1)
    mask[mask > 0] = 1
    half_volume_size = mask.shape[1] // 2
    left_mask = mask[:, :half_volume_size, :]
    right_mask = mask[:, half_volume_size:, :]

    left_label = int(np.any(left_mask > 0))
    right_label = int(np.any(right_mask > 0))
    hv_id = paths[-1].split('/')[-1].split('.')[0]
    if hv_id == 'W39_1998':
        hv_id = 'BraTS20_Training_355_seg'
    vol_dir_path = '_'.join(hv_id.split('_')[:-1])
    vol_dir_path = f'{OUTPUT_DIR}/{vol_dir_path}'

    for half in ['left', 'right']:
        os.makedirs(f'{vol_dir_path}/{half}', exist_ok=True)
        np.savez_compressed(f'{vol_dir_path}/{half}/{hv_id}_{half}.npz', data=left_mask.astype(
            np.uint8) if half == 'left' else right_mask.astype(np.uint8))
        df = df.append({
            'volume_id': i + 1,
            'path': f'{vol_dir_path}/{half}',
            'label': left_label if half == 'left' else right_label,
            'half': half
        }, ignore_index=True)

    # load all the volumes, split in half and save them
    # as numpy arrays, add also entries to the dataframe
    left_volume = []
    right_volume = []
    for path in paths[:-1]:
        volume = process_scan(path)
        left_volume.append(volume[:, :half_volume_size, :])
        right_volume.append(volume[:, half_volume_size:, :])

    hv_id = '_'.join(hv_id.split('_')[:-1])
    for half in ['left', 'right']:
        np.savez_compressed(f'{vol_dir_path}/{half}/{hv_id}_{half}.npz',
                            data=np.stack(left_volume, axis=-1) if half == 'left' else np.stack(right_volume, axis=-1))

# Generating Folds
indices_0 = df[df['label'] == 0].index.values.tolist()
indices_1 = df[df['label'] == 1].index.values.tolist()
training_indices = []
test_indices = []

# randomize the indices of the data with label 1
np.random.shuffle(indices_0)
np.random.shuffle(indices_1)

for i in range(N_FOLDS):
    # keep 1/5 of the data in the test set
    test_indices.append(
        indices_0[i * len(indices_0) // N_FOLDS:(i + 1) * len(indices_0) // N_FOLDS] +
        indices_1[i * len(indices_1) // N_FOLDS:(i + 1) * len(indices_1) // N_FOLDS])
    # keep the rest in the training set
    training_indices.append(
        indices_0[:i * len(indices_0) // N_FOLDS] + indices_0[(i + 1) * len(indices_0) // N_FOLDS:] +
        indices_1[:i * len(indices_1) // N_FOLDS] + indices_1[(i + 1) * len(indices_1) // N_FOLDS:])

# add relevant data to the df
df['fold'] = -1
for i in range(N_FOLDS):
    df.loc[df.index.isin(test_indices[i]), 'fold'] = i

# Folds in Numbers
n_test_samples_per_fold = len(test_indices[0])
n_training_samples_per_fold = len(training_indices[0])

print(f'{n_test_samples_per_fold} test samples, {n_training_samples_per_fold} training samples per fold.')
print(
    f'Each test fold contributes to {(n_test_samples_per_fold / (n_test_samples_per_fold + n_training_samples_per_fold)) * 100:.2f}% of the whole dataset.')

# class ratio in the training set
n_0 = len(df[(df['fold'] != 0) & (df['label'] == 0)])
n_1 = len(df[(df['fold'] != 0) & (df['label'] == 1)])

print(f'No tumor: {n_0} ({n_0 / (n_0 + n_1) * 100:.2f}%).')
print(f'Tumor: {n_1} ({n_1 / (n_0 + n_1) * 100:.2f}%).')

# Saving the Data
df.to_csv('metadata.csv', index=False)
