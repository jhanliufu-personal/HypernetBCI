from my_moabb import my_MOABBDataset
from braindecode.datasets import MOABBDataset
import os
from moabb.datasets.base import CacheConfig
from mne import get_config, set_config

subject_id = 2
dataset_name = 'BNCI2014_001'
# assuming current working directory is Hypernet/notebooks
# repo_path = os.path.dirname(os.getcwd())
repo_path = os.getcwd()
dir_data = os.path.join(repo_path, 'data')
print(dir_data)
os.environ[f'MNE_DATASETS_{dataset_name}_PATH'] = dir_data
set_config("MNE_DATA", dir_data)
print("get_config('MNE_DATA'):" + get_config('MNE_DATA'))

# print('Test CacheConfig')
# dict_cache_config = {'path': dir_data}
# myCacheConfig = CacheConfig.make(dict_cache_config)
# print(myCacheConfig.path)

# braindecode MOABBDataset class, can't specify download path
default_path_dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id])

# my_moabb class, should be able to specify download path
# my_path_dataset = my_MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id], download_path=dir_data)

# print(default_path_dataset.description)

print('Data loaded successfully')