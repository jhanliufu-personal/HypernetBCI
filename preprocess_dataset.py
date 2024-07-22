from braindecode.datasets import MOABBDataset
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
import numpy as np
import os

preprocessed_dir = 'data/Schirrmeister2017_preprocessed'
all_subject_id_lst = list(range(1, 14))

if os.path.exists(preprocessed_dir) and os.listdir(preprocessed_dir):
    print('Preprocessed dataset exists')
    # If a preprocessed dataset exists
    dataset_loaded = load_concat_dataset(
        path = preprocessed_dir,
        preload = True,
        ids_to_load = all_subject_id_lst,
        target_name = None,
    )
    print(f'Preprocessed dataset loaded from {preprocessed_dir}')

else:
    dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=all_subject_id_lst)

    ### ----------------------------------- PREPROCESSING -----------------------------------
    # low cut frequency for filtering
    low_cut_hz = 4.0  
    # high cut frequency for filtering
    high_cut_hz = 38.0  
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    transforms = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(
            lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
            factor=1e6,
        ),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(
            exponential_moving_standardize,  # Exponential moving standardization
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ]

    # Transform the data
    preprocess(dataset, transforms, n_jobs=1)
    print('Dataset preprocessed')

    # Save preprocessed dataset

    dataset.save(
        path=preprocessed_dir,
        overwrite=True,
    )
    print(f'Dataset saved to {preprocessed_dir}')